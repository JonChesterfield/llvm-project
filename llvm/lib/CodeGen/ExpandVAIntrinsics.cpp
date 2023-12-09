//===-- ExpandVAIntrinsicsPass.cpp --------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Can expand variadic functions, their calls, va_arg and the intrinsics.
//
// The lowering replaces the variadic argument (...) with a Int8Ty*, moves
// function arguments into a alloca struct and passes that address instead.
// The struct alignment is the natural stack alignment, or greater if arguments
// have greater ABI alignment.
//
// Order of operations is chosen to keep the IR semantically well formed.
// 1/ Expand intrinsics that don't involve function parameters
// 2/ Declare new functions with the ... arg replaced with void*
// 3/ Replace call instructions to the variadics with calls to the declarations
// 4/ Splice the body of original functions into the new ones
// 5/ Delete the remaining parts of the original functions
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ExpandVAIntrinsics.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <cstdio>

#define DEBUG_TYPE "expand-va-intrinsics"

using namespace llvm;
using namespace PatternMatch;

static cl::opt<bool>
    RewriteABI(DEBUG_TYPE "-abi", cl::init(false),
               cl::desc("Use valist as the vaarg calling convention"),
               cl::Hidden);

static cl::opt<bool> SplitFunctions(DEBUG_TYPE "-split", cl::init(true),
                                    cl::desc("Split variadic functions"),
                                    cl::Hidden);

static cl::opt<bool> ReplaceCalls(DEBUG_TYPE "-calls", cl::init(true),
                                  cl::desc("Rewrite variadic calls"),
                                  cl::Hidden);

static cl::opt<bool>
    ReplaceOperations(DEBUG_TYPE "-operations", cl::init(false),
                      cl::desc("Rewrite va_arg, va_end, va_copy"), cl::Hidden);

// TODO: May not want to ship this, but it's helpful during dev
// Notably when the pass is broken, gives an easy way to build offloading
static cl::opt<bool> DisablePass(DEBUG_TYPE "-disable", cl::init(false),
                                 cl::desc("Disable variadic expansion pass"),
                                 cl::Hidden);

namespace {

class ExpandVAIntrinsics : public ModulePass {
public:
  static char ID;
  const bool AllTransformsEnabled;
  Triple Trip;
  DenseMap<Function *, Function *> VariadicToReplacement;

  ExpandVAIntrinsics(bool A = false)
      : ModulePass(ID), AllTransformsEnabled(A) {}

  bool splitFunctions()
  {
    return SplitFunctions | AllTransformsEnabled;
  }
  bool replaceCalls()
  {
    return ReplaceCalls | AllTransformsEnabled;
  }
  bool replaceOperations()
  {
    return ReplaceOperations | AllTransformsEnabled;
  }
  bool rewriteABI()
  {
    return  RewriteABI | AllTransformsEnabled;
  }
  
  Type *vaListParameterType(LLVMContext &Ctx) {
    // True for everything at present, but probably not true for aarch
    return PointerType::getUnqual(Ctx);
  }

  bool implementedArchitecture(Module &M)
  {
    return vaListType(M.getContext()) != nullptr;
  }

  bool isX64WindowsABI()
  {
    // Trying to guess which x64 ABI is in use
    // TODO: seek help on this
    return Trip.isWindowsMSVCEnvironment() || Trip.isOSWindows();
  }

  struct slotAlignTy {
    uint32_t min;
    uint32_t max;
  };

  struct archinfo {
    Type *vaListType;
    slotAlignTy slotAlign;
  };

  archinfo getArchInfo(LLVMContext &Ctx) {
    switch(Trip.getArch()) {
    case Triple::r600:
    case Triple::amdgcn:
      return {PointerType::getUnqual(Ctx), {4, 4}};
    case Triple::nvptx:
    case Triple::nvptx64:
      return {PointerType::getUnqual(Ctx), {4, 0}};

    case Triple::x86:
      return {nullptr, {0, 0}};

    case Triple::x86_64:
      {
        if (isX64WindowsABI()) {
          return {nullptr, {0, 0}};
        }

        // An array of length one of a {i32, i32, ptr, ptr}
        // Not liking the top level stack up of these things,

        auto I32 = Type::getInt32Ty(Ctx);
        auto Ptr = PointerType::getUnqual(Ctx);
        return {ArrayType::get(StructType::get(Ctx,
                                               {
                                                   I32,
                                                   I32,
                                                   Ptr,
                                                   Ptr,
                                               }),
                               1),
                {8, 0}};
    }
    
    default:
      return {nullptr, {0, 0}};
    }
  }

  Type *vaListType(LLVMContext &Ctx) { return getArchInfo(Ctx).vaListType; }

  slotAlignTy archSlotAlign(LLVMContext &Ctx) {
    return getArchInfo(Ctx).slotAlign;
  }

  Value *createValistArgumentFromBuffer(Module &M, LLVMContext &Ctx,
                                        IRBuilder<> &Builder, Value *buffer) {
    const DataLayout &DL = M.getDataLayout();

    PointerType *voidptr = PointerType::getUnqual(Ctx);
    Value *voidBuffer =
        Builder.CreatePointerBitCastOrAddrSpaceCast(buffer, voidptr); 
    Type *va_list_ty = vaListType(Ctx);

#if 1
    // considering stack aligning the va_list instance
    // though it's probably better to drop the ptr align on the parameter
    Value *va_list_instance =
        Builder.Insert(new AllocaInst(va_list_ty, DL.getAllocaAddrSpace(),
                                      nullptr, assumedStructAlignment(DL)),
                       "va_list");
#else
    Value *va_list_instance =
      Builder.CreateAlloca(va_list_ty, nullptr, "va_list");
#endif

    if (va_list_ty == voidptr) {
      Builder.CreateStore(voidBuffer, va_list_instance);
      return Builder.CreatePointerBitCastOrAddrSpaceCast(va_list_instance,
                                                         voidptr);
    }

    if (Trip.getArch() == Triple::x86_64) {
      // TODO: Justify this

      Type *I32 = Type::getInt32Ty(Ctx);
      Type *I64 = Type::getInt64Ty(Ctx);

      Value *Idxs[3] = {
          ConstantInt::get(I64, 0),
          ConstantInt::get(I32, 0),
          nullptr,
      };

      Idxs[2] = ConstantInt::get(I32, 0);
      Builder.CreateStore(ConstantInt::get(I32, 48),
                          Builder.CreateInBoundsGEP(
                              va_list_ty, va_list_instance, Idxs, "gp_offset"));

      Idxs[2] = ConstantInt::get(I32, 1);
      Builder.CreateStore(ConstantInt::get(I32, 6 * 8 + 8 * 16),
                          Builder.CreateInBoundsGEP(
                              va_list_ty, va_list_instance, Idxs, "fp_offset"));

      Idxs[2] = ConstantInt::get(I32, 2);
      Builder.CreateStore(
          voidBuffer, Builder.CreateInBoundsGEP(va_list_ty, va_list_instance,
                                                Idxs, "overfow_arg_area"));

      Idxs[2] = ConstantInt::get(I32, 3);
      Builder.CreateStore(ConstantPointerNull::get(voidptr),
                          Builder.CreateInBoundsGEP(va_list_ty,
                                                    va_list_instance, Idxs,
                                                    "reg_save_area"));

      return va_list_instance;
    }

    report_fatal_error("create va list argument unknown arch");
  }

  void ExpandVACopy(const DataLayout &DL, VACopyInst *Inst) {
    IRBuilder<> Builder(Inst);
    auto &Ctx = Builder.getContext();
    Type *va_list_ty = vaListType(Ctx);
    uint64_t size = DL.getTypeAllocSize(va_list_ty).getFixedValue();
    // todo: on amdgcn this should be in terms of addrspace 5
    Builder.CreateMemCpyInline(Inst->getDest(), {}, Inst->getSrc(), {},
                               ConstantInt::get(Type::getInt32Ty(Ctx), size));
    Inst->eraseFromParent();
  }

  static void ExpandVAEnd(VAEndInst *Inst) {
    // A no-op on all the architectures implemented so far
    Inst->eraseFromParent();
  }


  bool runOnModule(Module &M) override {
    if (DisablePass) {
      return false;
    }

    #if 0
    /*
    Branch funnels do something weird here. Codegen looks like:

define hidden void @__typeid_typeid1_0_branch_funnel(ptr nest %0, ...) {
  musttail call void (...) @llvm.icall.branch.funnel(ptr %0, ptr @vt1_1, ptr @vf1_1, ptr @vt1_2, ptr @vf1_2, ...)
  ret void
}

    with call sites like
    
%1 = call i32 @__typeid_typeid1_0_branch_funnel(ptr nest %vtable, ptr %obj, i32 1)

    this is using variadic function types to represent some different thing. 

     Dead argument elimination manages to not trip over these by skipping HasMustTailCallers
     or HasMustTailCalls,  It crawls all uses of the function and if the use is not a callbase
     or various other things, it leaves it alone
     

    */
    #endif
    
    // M.dump();
    
    // Can get at triple, still need to distinguish x86 variants
    // Generally mistrusting of the constructor call time given the two pass managers
    Trip = Triple(M.getTargetTriple());

    if (!implementedArchitecture(M)) { return false; }

    if (false) fprintf(stderr, "ExpandVA: split %u, calls %u, op %u, abi %u\n",
                   splitFunctions(),
                   replaceCalls(),
                   replaceOperations(),
                   rewriteABI());
    
    bool Changed = false;
    for (Function &F : llvm::make_early_inc_range(M)){
      Changed |= runOnFunction(M, &F);
    }

    if (rewriteABI()) {
      for (Function &F : llvm::make_early_inc_range(M)){
        if (F.isDeclaration()) continue;
        for (BasicBlock &BB : F) {
          for (Instruction &I : llvm::make_early_inc_range(BB)) {
            if (CallBase *CB = dyn_cast<CallBase>(&I)) {
              if (CB->isIndirectCall()) {
                FunctionType* FTy = CB->getFunctionType();
                if (FTy->isVarArg()) {
                  // Alright, we've got an indirect call to a variadic function
                  report_fatal_error("Hit not yet handled indirect call case");

                }
              }
            }
          }
        }
      }

    }
    
    
    return Changed;
  }

  bool runOnFunction(Module &M, Function *F) {
    bool changed = false;

    //fprintf(stderr, "Run on %s\n", F->getName().str().c_str());

    if (F->isIntrinsic() ||
        F->hasFnAttribute(Attribute::Naked)) {
      return false;
    }

    if (replaceOperations()) {
      // Replaces va_copy, va_end in non-variadic functions as well
      changed = replaceOperationsInFunction(M, F);
    }

    // The remaining operations are only meaningful on variadic functions
    // though the rewrite abi is going to have to find remaining call instructions
    if (!F->isVarArg()) {
      return false;
    }
    
    if (!rewriteABI()) {
      // If changing the ABI is unavailable be more conservative about what functions can
      // be  changed.
      for (const Use &U : F->uses()) {
        // TODO: This more conservative path needs to be taken on the optimisation road
        // also this still isn't enough to not mangle the branch funnel thing
        bool bad = false;
        const auto *CB = dyn_cast<CallBase>(U.getUser());
        if (!CB || !CB->isCallee(&U) /*might not care about address escaping */ ||
            CB->getFunctionType() != F->getFunctionType()) {
          bad = true;
        }

        if (CB && CB->isMustTailCall()) {
          // can't do anything with musttail
          bad = true;
        }

        // there's an argument for ignoring functions which do nothing with the ...
        // i.e. leave it for dead argument elimination
        
        // might also want to check for tailcalls in the function itself
        if (bad) { return false; }
      }
    }

    bool usefulToSplit = splitFunctions() &&
      (!F->isDeclaration() || rewriteABI());
    
    if (usefulToSplit) {
      Function *NF = FindReplacement(M, F);
      if (!NF) {
        NF = DeriveReplacementFunctionFromVariadic(M, *F);
        assert(NF);
        changed = true;

        if (replaceOperations()) {
          // The newly created function will otherwise miss this expansion
          replaceOperationsInFunction(M, NF);
        }

        VariadicToReplacement[F] = NF;        
      }

      if (false) {
        if (!F->isDeclaration()) {
          if (NF != LookForPreExistingReplacement(M, F)) {
            printf("Warning, replaced function does not hit pattern\n");
          }
        }
      }
    }

    if (replaceCalls()) {
      Function *NF = FindReplacement(M, F);
      if (!NF) {
        return changed;
      }

      assert (F->isDeclaration() == NF->isDeclaration());

      assert(rewriteABI() || !F->isDeclaration());
      
      for (User *U : llvm::make_early_inc_range(F->users()))
        if (CallBase *CB = dyn_cast<CallBase>(U)) {
          // TODO: A test where the call instruction takes a variadic function as a parameter other than the one it is calling
          Value * calledOperand = CB->getCalledOperand();
          if (F == calledOperand) {
            ExpandCall(M, CB, F, NF);
            changed = true;
          }
        }
    }

    if (rewriteABI()) {
      // RewriteABI essentally requires the previous passes to have succeeded
      // first
      Function *NF = FindReplacement(M, F);
      if (!NF) {
        report_fatal_error("ExpandVA abi requires replacement function\n");
      }
      for (User *U : llvm::make_early_inc_range(F->users()))
        if (CallBase *CB = dyn_cast<CallBase>(U))
          report_fatal_error(
              "ExpandVA abi requires eliminating call uses first\n");

      // No direct calls exist to F, remaining uses are things like address escaping
      NF->setLinkage(F->getLinkage());
      NF->setVisibility(F->getVisibility());
      NF->takeName(F);

      // Indirect calls still need to be patched up
      // DAE bitcasts it, todo: check block addresses 
      F->replaceAllUsesWith(NF);
      F->eraseFromParent();
    }

    return true;
  }

  bool replaceOperationsInFunction(Module &M, Function *F) {
    // VAArg is lowered in clang for now
    bool Changed = false;
    if (F->isDeclaration()) { return Changed; }
    const DataLayout &DL = M.getDataLayout();
    for (BasicBlock &BB : *F) {
      for (Instruction &I : llvm::make_early_inc_range(BB)) {
        if (VAEndInst *II = dyn_cast<VAEndInst>(&I)) {
          Changed = true;
          ExpandVAEnd(II);
          continue;
        }
        if (VACopyInst *II = dyn_cast<VACopyInst>(&I)) {
          Changed = true;
          ExpandVACopy(DL, II);
          continue;
        }
      }
    }

    return Changed;
  }

  Function *FindReplacement(Module &M, Function *F) {
    // Cheapest to hit in the cache
    auto it = VariadicToReplacement.find(F);
    if (it != VariadicToReplacement.end()) {
      return it->second;
    }

    if(F->isDeclaration()) { return nullptr;}
    // Lowering by multiple calls may mean the cache is missing a value
    Function *maybe = LookForPreExistingReplacement(M, F);
    if (maybe) {
      VariadicToReplacement[F] = maybe;
      return maybe;
    }

    return nullptr;
  }

  static Function *LookForPreExistingReplacement(Module &, Function *F) {
    assert(F->isVarArg());
    assert(!F->isDeclaration());

    // Recognise functions that look exactly like the ones build by this pass
    // Will need some work to handle addrspace cast noise for amdgpu, that might
    // make patternmatch more worthwhile

    BasicBlock &BB = F->getEntryBlock();

    if (!isa<ReturnInst>(BB.getTerminator())) {
      return nullptr;
    }

    SmallVector<Instruction *, 5> seq;
    for (Instruction &inst : BB) {
      seq.push_back(&inst);
      if (seq.size() == 6) {
        break;
      }
    }
    if (seq.size() != 5)
      return nullptr;

    AllocaInst *alloca = dyn_cast<AllocaInst>(seq[0]);
    VAStartInst *start = dyn_cast<VAStartInst>(seq[1]);
    CallInst *call = dyn_cast<CallInst>(seq[2]);
    VAEndInst *end = dyn_cast<VAEndInst>(seq[3]);
    ReturnInst *ret = dyn_cast<ReturnInst>(seq[4]);

    if (!(alloca && start && call && end && ret)) {
      return nullptr;
    }

    // start and end acting on the alloca
    if ((start->getArgList() != alloca) || (end->getArgList() != alloca)) {
      return nullptr;
    }

    SmallVector<Value *> FuncArgs;
    for (Argument &A : F->args())
      FuncArgs.push_back(&A);

    SmallVector<Value *> CallArgs;
    for (Use &A : call->args())
      CallArgs.push_back(A);

    size_t Fixed = FuncArgs.size();
    if (Fixed + 1 != CallArgs.size()) {
      return nullptr;
    }

    for (size_t i = 0; i < Fixed; i++) {
      if (FuncArgs[i] != CallArgs[i]) {
        return nullptr;
      }
    }

    if (CallArgs[Fixed] != alloca) {
      return nullptr;
    }

    Value *maybeReturnValue = ret->getReturnValue();
    if (call->getType()->isVoidTy()) {
      if (maybeReturnValue) {
        return nullptr;
      } else {
        return call->getCalledFunction();
      }
    } else {
      if (maybeReturnValue != call) {
        return nullptr;
      } else {
        return call->getCalledFunction();
      }
    }
  }

 Function *DeriveReplacementFunctionFromVariadic(Module &M,
                                                 Function &F) {
    // todo, other sanity checks
    assert(F.isVarArg());

    auto &Ctx = M.getContext();
    const DataLayout &DL = M.getDataLayout();

    // Returned value isDeclaration() is equal to F.isDeclaration()
    // but that invariant is not satisfied throughout this function
    const bool FunctionIsDefinition = !F.isDeclaration();
    
    IRBuilder<> Builder(Ctx);
    FunctionType *FTy = F.getFunctionType();
    SmallVector<Type *> ArgTypes(FTy->param_begin(), FTy->param_end());
    ArgTypes.push_back(vaListParameterType(Ctx));

    FunctionType *NFTy =
        FunctionType::get(FTy->getReturnType(), ArgTypes, /*IsVarArgs*/ false);
    Function *NF = Function::Create(NFTy, F.getLinkage(), F.getAddressSpace());

    // Note - same attribute handling as DeadArgumentElimination
    NF->copyAttributesFrom(&F);
    NF->setComdat(F.getComdat()); // beware weak
    F.getParent()->getFunctionList().insert(F.getIterator(), NF);
    NF->setName(F.getName() + ".valist");

    // New function is default visibility and internal
    // Need to set visibility before linkage to avoid an assert in setVisibility
    NF->setVisibility(GlobalValue::DefaultVisibility);
    // NF->setLinkage(GlobalValue::InternalLinkage);

    AttrBuilder ParamAttrs(Ctx);
    ParamAttrs.addAttribute(Attribute::NoAlias);
    ParamAttrs.addAlignmentAttr(assumedStructAlignment(DL));

    AttributeList Attrs = NF->getAttributes();
    Attrs = Attrs.addParamAttributes(Ctx, NFTy->getNumParams() - 1, ParamAttrs);
    NF->setAttributes(Attrs);

    // Splice the implementation into the new function with minimal changes
    if(FunctionIsDefinition)
    {
      NF->splice(NF->begin(), &F);

      auto NewArg = NF->arg_begin();
      for (Argument &Arg : F.args()) {
        Arg.replaceAllUsesWith(NewArg);
        NewArg->setName(Arg.getName()); // takeName without killing the old one
        ++NewArg;
      }
      NewArg->setName("varargs");

      // Replace vastart of the ... with a vacopy of the new va_list argument
      for (BasicBlock &BB : *NF)
        for (Instruction &I : llvm::make_early_inc_range(BB))
          if (VAStartInst *II = dyn_cast<VAStartInst>(&I)) {
            Builder.SetInsertPoint(II);
            Value *start_arg = II->getArgList();
            auto * C = cast<VACopyInst>(Builder.CreateIntrinsic(Intrinsic::vacopy, {}, {start_arg, NewArg}));
            II->eraseFromParent();
            if (replaceOperations()) ExpandVACopy(DL, C);
          }
    }

    SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
    F.getAllMetadata(MDs);
    for (auto [KindID, Node] : MDs)
      NF->addMetadata(KindID, *Node);

    if(FunctionIsDefinition) {
    
      Type *va_list_ty = vaListType(Ctx);

      auto *BB = BasicBlock::Create(Ctx, "entry", &F);
      Builder.SetInsertPoint(BB);

      Value *va_list_instance =
        Builder.CreateAlloca(va_list_ty, nullptr, "va_list");

      // alloca puts it in the default stack addrspace and vastart doesn't
      // currently accept that

      // va start takes a void*, currently with no address space
      // this will convert a void* AS(5) to void*, and it also converts
      // an alloca of a x64 abi array of struct to a void*
      va_list_instance = Builder.CreatePointerBitCastOrAddrSpaceCast(
                                                                     va_list_instance, PointerType::getUnqual(Ctx));

      Builder.CreateIntrinsic(Intrinsic::vastart, {}, {va_list_instance});

      SmallVector<Value *> args;
      for (Argument &arg : F.args()) {
        args.push_back(&arg);
      }
      args.push_back(va_list_instance);

      Value *Result = Builder.CreateCall(NF, args);
      auto *C = cast<VAEndInst>(Builder.CreateIntrinsic(Intrinsic::vaend, {}, {va_list_instance}));
      if (replaceOperations()) ExpandVAEnd(C); 

      if (Result->getType()->isVoidTy())
        Builder.CreateRetVoid();
      else
        Builder.CreateRet(Result);
    }

    if (F.isDeclaration() != NF->isDeclaration()) {
      fprintf(stdout,"F decl %u, NF decl %u\n",F.isDeclaration(), NF->isDeclaration());

      fprintf(stdout,"F:\n");
      F.dump();
      fprintf(stdout,"NF:\n");
      NF->dump();
      fflush(stdout);
    }
    assert(F.isDeclaration() == NF->isDeclaration());
    return NF;
  }

  // Serious hazard around indirect calls here
  // They need to be expanded in the ABI changing case and need to not be expanded in the
  // not ABI changing case
  void ExpandCall(Module &M, CallBase *CB, Function*VarargF, Function *NF) {
    const DataLayout &DL = M.getDataLayout();

    if (CallInst *CI = dyn_cast<CallInst>(CB)) {
      if (CI->isMustTailCall()) {
        // Cannot expand musttail calls
        if (rewriteABI()) {
          report_fatal_error("Cannot rewrite musttail variadic call");
        } else {          
          return; // optimising, ignore this call          
        }
      }
    }

    // This is something of a problem because the call instruction's idea of the function type
    // doesn't necessarily match reality, before or after this pass
    // Since the plan here is to build a new instruction there is no particular
    // benefit to trying to preserve an incorrect initial type
    //
    // If the types don't match and we aren't changing ABI, leave it alone
    // in case someone is deliberately doing dubious type punning through a varargs
    FunctionType *FuncType = CB->getFunctionType();
    if (FuncType != VarargF->getFunctionType()) {
      if (!rewriteABI()) {return;}      
      FuncType = VarargF->getFunctionType();
    }

    auto &Ctx = CB->getContext();
    unsigned NumArgs = FuncType->getNumParams();

    SmallVector<Value *> Args;
    Args.assign(CB->arg_begin(), CB->arg_begin() + NumArgs);

    // ByVal is dealt with by putting the indirected type in the struct
    // and emitting a memcpy to write to it, but this doesn't appear to
    // bear much relation to what x64 is going to do with byval, bad alignment
    // handling?
    // ARM might be useful for testing indirect passing, I don't think X64 ever does.

    // TODO: This structure is a mess, fix it
    // from arg to index in localvartypes
    SmallVector<std::pair<Value *, uint64_t>> Varargs;
    SmallVector<Type *> LocalVarTypes;
    SmallVector<bool> isbyval;

    Align MaxFieldAlign(1);
    uint64_t CurrentOffset = 0;
    slotAlignTy slotAlign = archSlotAlign(Ctx);

    // X64 behaviour.
    // Slots are at least eight byte aligned and at most 16 byte aligned.
    // If the type needs more than sixteen byte alignment, it still only gets
    // that much alignment on the stack.

    for (unsigned I = FuncType->getNumParams(), E = CB->arg_size(); I < E;
         ++I) {
      Value *ArgVal = CB->getArgOperand(I);
      bool isByVal = CB->paramHasAttr(I, Attribute::ByVal);
      Type *ArgType = isByVal ? CB->getParamByValType(I) : ArgVal->getType();
      Align DataAlign = DL.getABITypeAlign(ArgType);
      MaxFieldAlign = std::max(MaxFieldAlign, DataAlign); // should be based on adjusted align

      uint64_t DataAlignV = DataAlign.value();
      // Fun thing. x64 ABI implies there's a maximum slot align of 16,
      // but clang va_arg lowering assumes the types are natively aligned
      if (slotAlign.min && DataAlignV < slotAlign.min)
        DataAlignV = slotAlign.min;
      if (slotAlign.max && DataAlignV > slotAlign.max)
        DataAlignV = slotAlign.max;

      if (uint64_t Rem = CurrentOffset % DataAlignV) {
        // Inject explicit padding to deal with alignment requirements
        uint64_t Padding = DataAlignV - Rem;
        Type *ATy = ArrayType::get(Type::getInt8Ty(Ctx), Padding);
        LocalVarTypes.push_back(ATy);
        isbyval.push_back(false);
        CurrentOffset += Padding;
      }

      Varargs.push_back({ArgVal, LocalVarTypes.size()});
      LocalVarTypes.push_back(ArgType);
      isbyval.push_back(isByVal);

      CurrentOffset += DL.getTypeAllocSize(ArgType).getFixedValue();
    }

    if (Varargs.empty()) {
      // todo, pass nullptr instead?
      LocalVarTypes.push_back(Type::getInt32Ty(Ctx));
      isbyval.push_back(false);
    }

    const bool isPacked = true;
    StructType *VarargsTy = StructType::create(
        Ctx, LocalVarTypes, (Twine(NF->getName()) + ".vararg").str(), isPacked);

    Function *CBF = CB->getParent()->getParent();
    BasicBlock &BB = CBF->getEntryBlock();
    IRBuilder<> Builder(&*BB.getFirstInsertionPt());

    auto alloced = Builder.Insert(
        new AllocaInst(VarargsTy, DL.getAllocaAddrSpace(), nullptr,
                       std::max(MaxFieldAlign, assumedStructAlignment(DL))),
        "vararg_buffer");

    // TODO: Lifetime annotate it
    Builder.SetInsertPoint(CB);
    for (size_t i = 0; i < Varargs.size(); i++) {
      auto r = Builder.CreateStructGEP(VarargsTy, alloced, Varargs[i].second);
      if (isbyval[Varargs[i].second]) {
        Type *ByValType = LocalVarTypes[Varargs[i].second];
        Builder.CreateMemCpy(r, {}, Varargs[i].first, {},
                             DL.getTypeAllocSize(ByValType).getFixedValue());
      } else {
        Builder.CreateStore(Varargs[i].first,
                            r);
      }
    }

    Args.push_back(createValistArgumentFromBuffer(M, Ctx, Builder, alloced));

    // Attributes excluding any on the vararg arguments
    AttributeList PAL = CB->getAttributes();
    if (!PAL.isEmpty()) {
      SmallVector<AttributeSet, 8> ArgAttrs;
      for (unsigned ArgNo = 0; ArgNo < NumArgs; ArgNo++)
        ArgAttrs.push_back(PAL.getParamAttrs(ArgNo));
      PAL = AttributeList::get(Ctx, PAL.getFnAttrs(), PAL.getRetAttrs(),
                               ArgAttrs);
    }

    SmallVector<OperandBundleDef, 1> OpBundles;
    CB->getOperandBundlesAsDefs(OpBundles);

    CallBase *NewCB = nullptr;
    // TODO, other instructions
    if (InvokeInst *II = dyn_cast<InvokeInst>(CB)) {
      NewCB = InvokeInst::Create(NF, II->getNormalDest(), II->getUnwindDest(),
                                 Args, OpBundles, "", CB);
    } else {
      NewCB = CallInst::Create(NF, Args, OpBundles, "", CB);

      CallInst::TailCallKind TCK = cast<CallInst>(CB)->getTailCallKind();
      assert(TCK != CallInst::TCK_MustTail); // guarded at prologue

      // It doesn't get to be a tail call any more
      // might want to guard this with arch, x64 and aarch64 document that
      // varargs can't be tail called anyway
      // Not totally convinced this is necessary but dead store elimination
      // decides to discard the stores to the alloca and pass uninitialised
      // memory along instead if the function is marked tailcall
      if (TCK == CallInst::TCK_Tail) {
        TCK = CallInst::TCK_None;
      }
      cast<CallInst>(NewCB)->setTailCallKind(TCK);
    }

    NewCB->setAttributes(PAL);
    NewCB->takeName(CB);
    NewCB->setCallingConv(CB->getCallingConv());
    NewCB->copyMetadata(*CB, {LLVMContext::MD_prof, LLVMContext::MD_dbg});
    
    if (!CB->use_empty()) // dead branch?
      {
        CB->replaceAllUsesWith(NewCB);
      }
    CB->eraseFromParent();
  }

  static Align assumedStructAlignment(const DataLayout &DL) {
    // TODO: Change the DataLayout API so there's an easier way to test
    // whether the stack alignment is known. Nvptx doesn't always have S
    // in the data layout string, this sidesteps an assertion there.
    Align ExcessiveAlignment = Align(UINT64_C(1) << 63u);
    bool knownNaturalStackAlignment =
        DL.exceedsNaturalStackAlignment(ExcessiveAlignment);
    if (knownNaturalStackAlignment) {
      return DL.getStackAlignment();
    } else {
      return {};
    }
  }
};
} // namespace

char ExpandVAIntrinsics::ID = 0;

INITIALIZE_PASS(ExpandVAIntrinsics, DEBUG_TYPE, "Expand VA intrinsics", false,
                false)

ModulePass *llvm::createExpandVAIntrinsicsPass(bool ApplicableToAllFunctions) {
  return new ExpandVAIntrinsics(ApplicableToAllFunctions);
}

PreservedAnalyses ExpandVAIntrinsicsPass::run(Module &M,
                                              ModuleAnalysisManager &) {
  return ExpandVAIntrinsics(false).runOnModule(M) ? PreservedAnalyses::none()
                                                  : PreservedAnalyses::all();
}
