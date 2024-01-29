//===-- ExpandVAIntrinsicsPass.cpp --------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a codegen lowering and optimisation pass for variadic functions.
//
// For codegen it eliminates all traces of variadic functions from the module.
// For optimisation, it replaces some calls to variadic functions with calls to
// equivalent non-variadic functions which unblocks other optimisations.
//
// The mechanism is to replace the trailing ... parameter with a va_list value
// passed as the ABI usually would, then pack the arguments into a struct at
// the call site to pass via a va_list value. For example, replace a fprintf
// call with a vprintf call if the usual fprintf definition is in the module.
//
// Where this transform is consistent with the ABI, e.g. AMDGPU or NVPTX, or
// where the ABI can be chosen to align with this transform, variadic functions
// are essentially zero cost. In particular the output of this pass composes
// with function inlining and similar.
//
//
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
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TargetParser/Triple.h"
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

// va_arg is not yet implemented here
static cl::opt<bool> ReplaceOperations(DEBUG_TYPE "-operations", cl::init(true),
                                       cl::desc("Rewrite va_end, va_copy"),
                                       cl::Hidden);

// TODO: May not want to ship this, but it's helpful during dev
// Notably when the pass is broken, gives an easy way to build offloading
static cl::opt<bool> DisablePass(DEBUG_TYPE "-disable", cl::init(false),
                                 cl::desc("Disable variadic expansion pass"),
                                 cl::Hidden);

namespace {
namespace ABI {

// calling convention for passing as valist object, same as it would be in C
// aarch64 uses byval
enum class valistCC { value, pointer, /*byval*/ };

// Values may be passed indirectly
// As far as va_arg is concerned, that's indistinguishable from a void*
// Values may also be passed as byval in which case the memcpy is needed

struct Interface {
protected:
  Interface() {}

public:
  virtual ~Interface() {}

  // Most ABIs use a void* or char* for va_list, others can specialise
  virtual Type *vaListType(LLVMContext &Ctx) {
    return PointerType::getUnqual(Ctx);
  }

  // How the vaListType is passed
  virtual valistCC vaListCC() { return valistCC::value; }

  // The valist might need to be stack allocated.
  virtual bool valistOnStack() { return false; }

  virtual void initializeVAList(LLVMContext &Ctx, IRBuilder<> &Builder,
                                AllocaInst *va_list, Value * /*buffer*/) {
    // Function needs to be implemented if valist is on the stack
    assert(!valistOnStack());
    assert(va_list == nullptr);
  }

  virtual bool allow_higher_align() = 0;
  virtual uint32_t minimum_slot_align() = 0;
  virtual uint32_t maximum_slot_align() = 0;

  // Could make these virtual, fair chance that's free since all
  // classes choose not to override them at present

#if 0
  uint64_t vaListTypeSize(Module &M) {
    auto &Ctx = M.getContext();
    const DataLayout &DL = M.getDataLayout();
    Type * Ty = vaListType(Ctx);
    return DL.getTypeAllocSize(Ty).getFixedValue();
  }
#endif

  // All targets currently implemented use a ptr for the valist parameter
  Type *vaListParameterType(LLVMContext &Ctx) {
    return PointerType::getUnqual(Ctx);
  }

  bool VAEndIsNop() { return true; }

  bool VACopyIsMemcpy() { return true; }
};

struct X64Linux final : public Interface {
  Type *vaListType(LLVMContext &Ctx) override {
    auto I32 = Type::getInt32Ty(Ctx);
    auto Ptr = PointerType::getUnqual(Ctx);
    return ArrayType::get(StructType::get(Ctx,
                                          {
                                              I32,
                                              I32,
                                              Ptr,
                                              Ptr,
                                          }),
                          1);
  }
  valistCC vaListCC() override { return valistCC::pointer; }

  bool valistOnStack() override { return true; }

  void initializeVAList(LLVMContext &Ctx, IRBuilder<> &Builder,
                        AllocaInst *va_list, Value *voidBuffer) override {
    assert(valistOnStack());
    assert(va_list != nullptr);
    assert(va_list->getAllocatedType() == vaListType(Ctx));

    Type *va_list_ty = vaListType(Ctx);

    Type *I32 = Type::getInt32Ty(Ctx);
    Type *I64 = Type::getInt64Ty(Ctx);

    Value *Idxs[3] = {
        ConstantInt::get(I64, 0),
        ConstantInt::get(I32, 0),
        nullptr,
    };

    Idxs[2] = ConstantInt::get(I32, 0);
    Builder.CreateStore(
        ConstantInt::get(I32, 48),
        Builder.CreateInBoundsGEP(va_list_ty, va_list, Idxs, "gp_offset"));

    Idxs[2] = ConstantInt::get(I32, 1);
    Builder.CreateStore(
        ConstantInt::get(I32, 6 * 8 + 8 * 16),
        Builder.CreateInBoundsGEP(va_list_ty, va_list, Idxs, "fp_offset"));

    Idxs[2] = ConstantInt::get(I32, 2);
    Builder.CreateStore(voidBuffer,
                        Builder.CreateInBoundsGEP(va_list_ty, va_list, Idxs,
                                                  "overfow_arg_area"));

    Idxs[2] = ConstantInt::get(I32, 3);
    Builder.CreateStore(
        ConstantPointerNull::get(PointerType::getUnqual(Ctx)),
        Builder.CreateInBoundsGEP(va_list_ty, va_list, Idxs, "reg_save_area"));
  }

  bool allow_higher_align() override { return true; }
  uint32_t minimum_slot_align() override { return 8; }
  uint32_t maximum_slot_align() override { return 0; }
};

struct X64Windows final : public Interface {
  bool allow_higher_align() override { return false; }
  uint32_t minimum_slot_align() override { return 8; }
  uint32_t maximum_slot_align() override { return 8; }
};

struct X86 final : public Interface {
  bool allow_higher_align() override { return true; }
  uint32_t minimum_slot_align() override { return 4; }
  uint32_t maximum_slot_align() override { return 0; }
};

struct AMDGPU final : public Interface {
  bool allow_higher_align() override { return true; }
  uint32_t minimum_slot_align() override { return 1; }
  uint32_t maximum_slot_align() override { return 0; }
};

struct NVPTX final : public Interface {
  bool allow_higher_align() override { return true; }
  uint32_t minimum_slot_align() override { return 4; }
  uint32_t maximum_slot_align() override { return 0; }
};

std::unique_ptr<Interface> create(Module &M) {
  Triple Trip = Triple(M.getTargetTriple());

  switch (Trip.getArch()) {
  default:
    return nullptr;

  case Triple::r600:
  case Triple::amdgcn: {
    return std::make_unique<ABI::AMDGPU>();
  }

  case Triple::nvptx:
  case Triple::nvptx64: {
    return std::make_unique<ABI::NVPTX>();
  }

  case Triple::x86: {
    return std::make_unique<ABI::X86>();
  }

  case Triple::x86_64: {
    if (Trip.isWindowsMSVCEnvironment() || Trip.isOSWindows()) {
      return std::make_unique<ABI::X64Windows>();
    }

    // todo, don't default to this
    return std::make_unique<ABI::X64Linux>();
  }
  }
}

} // namespace ABI

class ExpandVAIntrinsics : public ModulePass {
public:
  static char ID;
  const bool AllTransformsEnabled;
  Triple Trip;
  std::unique_ptr<ABI::Interface> ABI;
  DenseMap<Function *, Function *> wrapperToVAListEquivalent;

  // if the target stack allocates valist, use the same one on each call.
  // DenseMap<Function *, Value*> functionTovaListAlloca;

  ExpandVAIntrinsics(bool A = false)
      : ModulePass(ID), AllTransformsEnabled(A) {}

  bool splitFunctions() { return SplitFunctions | rewriteABI(); }
  bool replaceCalls() { return ReplaceCalls | rewriteABI(); }
  bool replaceOperations() { return ReplaceOperations | rewriteABI(); }
  bool rewriteABI() { return RewriteABI | AllTransformsEnabled; }

  bool conservative() { return !rewriteABI(); }

  bool isX64WindowsABI() {
    // Trying to guess which x64 ABI is in use
    // TODO: test this, looks like there are only two triples involved (msvc and
    // gnu)
    return Trip.isWindowsMSVCEnvironment() || Trip.isOSWindows();
  }

  void MemcpyVAListPointers(const DataLayout &DL, IRBuilder<> &Builder,
                            Value *dst, Value *src) {
    auto &Ctx = Builder.getContext();
    Type *va_list_ty = ABI->vaListType(Ctx);
    uint64_t size = DL.getTypeAllocSize(va_list_ty).getFixedValue();
    // todo: on amdgcn this should be in terms of addrspace 5
    Builder.CreateMemCpyInline(dst, {}, src, {},
                               ConstantInt::get(Type::getInt32Ty(Ctx), size));
  }

  bool ExpandVACopy(const DataLayout &DL, VACopyInst *Inst) {
    IRBuilder<> Builder(Inst);
    MemcpyVAListPointers(DL, Builder, Inst->getDest(), Inst->getSrc());
    Inst->eraseFromParent();
    return true;
  }

  static bool ExpandVAEnd(VAEndInst *Inst) {
    // A no-op on all the architectures implemented so far
    Inst->eraseFromParent();
    return true;
  }

  bool runOnModule(Module &M) override {
    if (DisablePass) {
      return false;
    }

    // Can get at triple, still need to distinguish x86 variants
    // Generally mistrusting of the constructor call time given the two pass
    // managers
    const DataLayout &DL = M.getDataLayout();
    Trip = Triple(M.getTargetTriple());

    // Tempting to specialise the pass on this instead of doing the runtime
    // vtable.
    ABI = ABI::create(M);

    if (!ABI) {
      return false;
    }

    if (0)
      fprintf(stderr, "ExpandVA: split %u, calls %u, op %u, abi %u\n",
              splitFunctions(), replaceCalls(), replaceOperations(),
              rewriteABI());

    bool Changed = false;

    for (Function &F : llvm::make_early_inc_range(M)) {
      Changed |= runOnFunction(M, &F);
    }

    if (replaceOperations()) {
      Function *vaend = Intrinsic::getDeclaration(&M, Intrinsic::vaend);
      for (User *U : vaend->users())
        if (auto *I = dyn_cast<VAEndInst>(U))
          Changed |= ExpandVAEnd(I);

      Function *vacopy = Intrinsic::getDeclaration(&M, Intrinsic::vacopy);
      for (User *U : vacopy->users())
        if (auto *I = dyn_cast<VACopyInst>(U))
          Changed |= ExpandVACopy(DL, I);
    }

    if (rewriteABI()) {
      for (Function &F : llvm::make_early_inc_range(M)) {
        if (F.isDeclaration())
          continue;

        // Now need to track down indirect calls. Can't find those
        // by walking uses of variadic functions, need to crawl the instruction
        // stream. Fortunately this is only necessary for the ABI rewrite case.
        for (BasicBlock &BB : F) {
          for (Instruction &I : llvm::make_early_inc_range(BB)) {
            if (CallBase *CB = dyn_cast<CallBase>(&I)) {
              if (CB->isIndirectCall()) {
                FunctionType *FTy = CB->getFunctionType();
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

  // this should be ABI-specific
  bool expansionApplicableToFunction(Module &M, Function *F) {
    if (F->isIntrinsic() || !F->isVarArg() ||
        F->hasFnAttribute(Attribute::Naked)) {
      return false;
    }

    if (F->getCallingConv() != CallingConv::C) {
      return false;
    }

    if (conservative()) {
      // linkonceodr / internal etc can be transformed as the external
      // calling convention is preserved.
      // weak can't be split unless changing ABI is allowed
      if (GlobalValue::isInterposableLinkage(F->getLinkage())) {
        return false;
      }
    }

    if (conservative()) {
      // If optimising, err on the side of leaving things alone
      for (const Use &U : F->uses()) {
        const auto *CB = dyn_cast<CallBase>(U.getUser());
        if (!CB ||
            !CB->isCallee(&U) /*might not care about address escaping */ ||
            CB->getFunctionType() != F->getFunctionType()) {
          return false;
        }

        if (CB && CB->isMustTailCall()) {
          // can't do anything with musttail
          return false;
        }

        // there's an argument for ignoring functions which do nothing with the
        // ... i.e. leave it for dead argument elimination

        // might also want to check for tailcalls in the function itself
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

      // branch funnel intrinsics are called from functions that look variadic
      // but mean something else
      // todo: there should be a reasonable way to check for an intrinsic
      // without inserting a prototype
      Function *funnel =
          Intrinsic::getDeclaration(&M, Intrinsic::icall_branch_funnel);
      for (const User *U : funnel->users()) {
        if (auto *I = dyn_cast<CallBase>(U)) {
          if (F == I->getFunction()) {
            return false;
          }
        }
      }
      if (funnel->use_empty())
        funnel->eraseFromParent();
    }

    return true;
  }

  bool runOnFunction(Module &M, Function *F) {
    bool changed = false;

    // fprintf(stderr, "Called on %s\n", F->getName().str().c_str());

    // This check might be too coarse - there are probably cases where
    // splitting a function is bad but it's usable without splitting
    if (!expansionApplicableToFunction(M, F))
      return false;

    // TODO: Leave "thunk" attribute functions alone

    bool usefulToSplit =
        splitFunctions() && (!F->isDeclaration() || rewriteABI());

    // F may already be a single basic block calling a known function
    // that takes a va_list, in which case it doens't need to be split.
    Function *Equivalent = isWrapperAroundVAListEquivalent(M, F);

    if (usefulToSplit && !Equivalent) {
      Equivalent = DeriveWrappedFunctionFromVariadic(M, *F);
      assert(Equivalent);
      changed = true;
      wrapperToVAListEquivalent[F] = Equivalent;
    }

    if (rewriteABI() && !Equivalent) {
      report_fatal_error("ExpandVA abi requires replacement function\n");
    }

    if (replaceCalls()) {
      if (!Equivalent) {
        assert(!rewriteABI());
        return changed;
      }

      for (User *U : llvm::make_early_inc_range(F->users()))
        // Need to handle invoke etc, or at least deliberately ignore them on
        // the optimise path
        // TODO: A test where the call instruction takes a variadic function as
        // a parameter other than the one it is calling
        if (CallBase *CB = dyn_cast<CallBase>(U)) {
          Value *calledOperand = CB->getCalledOperand();
          if (F == calledOperand) {
            ExpandCall(M, CB, F, Equivalent);
            changed = true;
          }
        }
    }

    if (rewriteABI()) {
      assert(Equivalent);
      // RewriteABI essentally requires the previous passes to have succeeded
      // first. Failures to lower are fatal if rewriting is requested - it's a
      // backend pass with no alternative plan for dealing with variadics.

      // No direct calls remain to F, remaining uses are things like address
      // escaping, modulo errors in this implementation.
      for (User *U : llvm::make_early_inc_range(F->users()))
        if (CallBase *CB = dyn_cast<CallBase>(U))
          report_fatal_error(
              "ExpandVA abi requires eliminating call uses first\n");

      // Converting the original variadic function in-place into the equivalent
      // one.
      Equivalent->setLinkage(F->getLinkage());
      Equivalent->setVisibility(F->getVisibility());
      Equivalent->takeName(F);

      // Indirect calls still need to be patched up
      // DAE bitcasts it, todo: check block addresses
      F->replaceAllUsesWith(Equivalent);
      F->eraseFromParent();
    }

    return true;
  }

  // This should probably only skip if the size and pointer values are right
  template <Intrinsic::ID ID>
  static BasicBlock::iterator skipSpecificIntrinsic(BasicBlock::iterator Iter) {
    if (auto *Intrinsic = dyn_cast<IntrinsicInst>(&*Iter))
      if (Intrinsic->getIntrinsicID() == ID)
        Iter++;
    return Iter;
  }

  // A predicate in that return nullptr means F is not a wrapper
  // Returns the function it is a wrapper around on success
  Function *isWrapperAroundVAListEquivalent(Module &M, Function *F) {
    assert(F->isVarArg());
    assert(expansionApplicableToFunction(M, F));

    {
      auto it = wrapperToVAListEquivalent.find(F);
      if (it != wrapperToVAListEquivalent.end())
        return it->second;
    }

    if (F->isDeclaration())
      return nullptr;

    const bool verbose = true;

    auto VALOG2 = [&](int line) -> Function * {
      (void)verbose;
      // fprintf(stdout, "L%d\n", line);
      return nullptr;
    };
#define VALOG() VALOG2(__LINE__)

    // Recognise functions that look similar enough to the ones built by this
    // pass
    // Will need some work to handle addrspace cast noise for amdgpu

    // Matching functions are a single basic block.
    BasicBlock &BB = F->getEntryBlock();
    if (!isa<ReturnInst>(BB.getTerminator())) {
      return VALOG();
    }

    BasicBlock::iterator it = BB.begin();

    AllocaInst *alloca = dyn_cast<AllocaInst>(&*it++);
    if (!alloca)
      return VALOG();

    Value *valist_argument = alloca;

    it = skipSpecificIntrinsic<Intrinsic::lifetime_start>(it);

    VAStartInst *start = dyn_cast<VAStartInst>(&*it++);
    if (!start || start->getArgList() != valist_argument) {
      return VALOG();
    }

    // The va_list instance is stack allocated
    // The ... replacement is a va_list passed "by value"
    // That involves a load for some ABIs and passing the pointer for others
    Value *valist_trailing_argument = nullptr;
    switch (ABI->vaListCC()) {
    case ABI::valistCC::value: {
      // If it's being passed by value, need a load
      // TODO: Check it's loading the right thing
      auto *load = dyn_cast<LoadInst>(&*it);
      if (!load)
        return VALOG();
      valist_trailing_argument = load;
      it++;
      break;
    }
    case ABI::valistCC::pointer: {
      // If it's being passed by pointer, going to use the alloca directly
      valist_trailing_argument = valist_argument;
      break;
    }
    }

    CallInst *call = dyn_cast<CallInst>(&*it++);
    if (!call)
      return VALOG();

    if (auto *end = dyn_cast<VAEndInst>(&*it)) {
      if (end->getArgList() != valist_argument)
        return VALOG();
      it++;
    } else {
      // If vaend does nothing anyway, ignore it missing
      if (!ABI->VAEndIsNop())
        return VALOG();
    }

    it = skipSpecificIntrinsic<Intrinsic::lifetime_end>(it);

    ReturnInst *ret = dyn_cast<ReturnInst>(&*it++);
    if (!ret || it != BB.end())
      return VALOG();

    // The function call is expected to take the fixed arguments then the alloca
    // TODO: Drop the vectors here
    SmallVector<Value *> FuncArgs;
    for (Argument &A : F->args())
      FuncArgs.push_back(&A);

    SmallVector<Value *> CallArgs;
    for (Use &A : call->args())
      CallArgs.push_back(A);

    size_t Fixed = FuncArgs.size();
    if (Fixed + 1 != CallArgs.size())
      return VALOG();

    for (size_t i = 0; i < Fixed; i++)
      if (FuncArgs[i] != CallArgs[i])
        return VALOG();

    if (CallArgs[Fixed] != valist_trailing_argument)
      return VALOG();

    // TODO: This is messy
    bool callIsVoid = call->getType()->isVoidTy();
    Value *maybeReturnValue = ret->getReturnValue();
    if (callIsVoid && (maybeReturnValue != nullptr))
      return VALOG();
    if (!callIsVoid && (maybeReturnValue != call))
      return VALOG();

    Function *Equivalent = call->getCalledFunction();
    wrapperToVAListEquivalent[F] = Equivalent;
    return Equivalent;
  }

  // Given a variadic function, creates an equivalent function which takes a
  // va_list instead of a .., and mutates F to be a single block that calls
  // into the equivalent function.
  Function *DeriveWrappedFunctionFromVariadic(Module &M, Function &F) {
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
    ArgTypes.push_back(ABI->vaListParameterType(Ctx));

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
    NF->setLinkage(GlobalValue::InternalLinkage);

    AttrBuilder ParamAttrs(Ctx);
    ParamAttrs.addAttribute(Attribute::NoAlias);
    // TODO: Reintroduce alignment here where possible, it cleans up codegen a
    // lot ParamAttrs.addAlignmentAttr(assumedStructAlignment(DL));

    AttributeList Attrs = NF->getAttributes();
    Attrs = Attrs.addParamAttributes(Ctx, NFTy->getNumParams() - 1, ParamAttrs);
    NF->setAttributes(Attrs);

    // Splice the implementation into the new function with minimal changes
    if (FunctionIsDefinition) {
      NF->splice(NF->begin(), &F);

      auto NewArg = NF->arg_begin();
      for (Argument &Arg : F.args()) {
        Arg.replaceAllUsesWith(NewArg);
        NewArg->setName(Arg.getName()); // takeName without killing the old one
        ++NewArg;
      }
      NewArg->setName("varargs");

      // TODO: Don't crawl all the instructions looking for an intrinsic
      // Replace vastart of the ... with a memcpy from the new va_list argument
      // This will need to be adjusted if a target uses a va_copy/va_end pair
      // which are not replacable with a memcpy.
      for (BasicBlock &BB : *NF)
        for (Instruction &I : llvm::make_early_inc_range(BB))
          if (VAStartInst *II = dyn_cast<VAStartInst>(&I)) {
            Builder.SetInsertPoint(II);

            // va_start takes a pointer to a va_list, e.g. one on the stack.
            // Retrieve it:
            Value *va_start_arg = II->getArgList();

            // The last argument is a vaListParameterType
            Value *passed_va_list = NewArg;

            switch (ABI->vaListCC()) {
            case ABI::valistCC::value: {
              // Got a va_list in an ssa register
              Builder.CreateStore(passed_va_list, va_start_arg);
              break;
            }
            case ABI::valistCC::pointer: {
              // src and dst are both pointers
              MemcpyVAListPointers(DL, Builder, va_start_arg, passed_va_list);
              break;
            }
            }

            II->eraseFromParent();
          }
    }

    SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
    F.getAllMetadata(MDs);
    for (auto [KindID, Node] : MDs)
      NF->addMetadata(KindID, *Node);

    if (FunctionIsDefinition) {
      Type *va_list_ty = ABI->vaListType(Ctx);

      auto *BB = BasicBlock::Create(Ctx, "entry", &F);
      Builder.SetInsertPoint(BB);

      Value *va_list_instance =
          Builder.CreateAlloca(va_list_ty, nullptr, "va_list");

      // alloca puts it in the default stack addrspace and vastart doesn't
      // currently accept that

      // this will convert a void* AS(5) to void*, and it also converts
      // an alloca of a x64 abi array of struct to a void*
      va_list_instance = Builder.CreatePointerBitCastOrAddrSpaceCast(
          va_list_instance, PointerType::getUnqual(Ctx));

      Builder.CreateIntrinsic(Intrinsic::vastart, {}, {va_list_instance});

      SmallVector<Value *> args;
      for (Argument &arg : F.args())
        args.push_back(&arg);

      args.push_back(va_list_instance);

      CallInst *Result = Builder.CreateCall(NF, args);
      Result->setTailCallKind(
          CallInst::TCK_Tail); // todo: look for notail marker

      // vaend is a no-op on implemented targets, no cleanup to prevent the tail
      // call

      if (Result->getType()->isVoidTy())
        Builder.CreateRetVoid();
      else
        Builder.CreateRet(Result);
    }

    assert(F.isDeclaration() == NF->isDeclaration());
    return NF;
  }

  // Serious hazard around indirect calls here
  // They need to be expanded in the ABI changing case and need to not be
  // expanded in the not ABI changing case
  void ExpandCall(Module &M, CallBase *CB, Function *VarargF, Function *NF) {
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

    // This is something of a problem because the call instructions' idea of the
    // function type doesn't necessarily match reality, before or after this
    // pass
    // Since the plan here is to build a new instruction there is no
    // particular benefit to trying to preserve an incorrect initial type
    // If the types don't match and we aren't changing ABI, leave it alone
    // in case someone is deliberately doing dubious type punning through a
    // varargs
    FunctionType *FuncType = CB->getFunctionType();
    if (FuncType != VarargF->getFunctionType()) {
      if (!rewriteABI()) {
        return;
      }
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
    // ARM might be useful for testing indirect passing, I don't think X64 ever
    // does.
    // nvptx uses indirect.

    // TODO: This structure is a mess, fix it. Noise is from indexing over
    // padding. from arg to index in localvartypes
    SmallVector<std::pair<Value *, uint64_t>> Varargs;

    // localvartypes and isbyval are equal length (I think), varargs is <= their
    // length
    SmallVector<Type *> LocalVarTypes;
    SmallVector<bool> isbyval;

    struct slotAlignTy {
      uint32_t min;
      uint32_t max;
    };

    slotAlignTy slotAlign;
    slotAlign.min = ABI->minimum_slot_align();
    slotAlign.max = ABI->maximum_slot_align();

    // Align the struct on slotAlign.min to start with
    // Some targets will increase that alignment
    Align MaxFieldAlign(slotAlign.min);
    uint64_t CurrentOffset = 0;

    // X64 documented behaviour:
    // Slots are at least eight byte aligned and at most 16 byte aligned.
    // If the type needs more than sixteen byte alignment, it still only gets
    // that much alignment on the stack.
    // X64 behaviour in clang:
    // Slots are at least eight byte aligned and at most naturally aligned
    // Going with clang here.

    for (unsigned I = FuncType->getNumParams(), E = CB->arg_size(); I < E;
         ++I) {
      Value *ArgVal = CB->getArgOperand(I);
      bool isByVal = CB->paramHasAttr(I, Attribute::ByVal);
      Type *ArgType = isByVal ? CB->getParamByValType(I) : ArgVal->getType();
      Align DataAlign = DL.getABITypeAlign(ArgType);

      uint64_t DataAlignV = DataAlign.value();

      // Currently using 0 as a sentinel to mean ignored
      if (slotAlign.min && DataAlignV < slotAlign.min)
        DataAlignV = slotAlign.min;
      if (slotAlign.max && DataAlignV > slotAlign.max)
        DataAlignV = slotAlign.max;

      DataAlign = Align(DataAlignV);
      MaxFieldAlign = std::max(MaxFieldAlign, DataAlign);

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
      // todo, pass nullptr instead? something platform-specific?
      LocalVarTypes.push_back(Type::getInt32Ty(Ctx));
      isbyval.push_back(false);
    }

    const bool isPacked = true;
    StructType *VarargsTy = StructType::create(
        Ctx, LocalVarTypes, (Twine(NF->getName()) + ".vararg").str(), isPacked);

    Function *CBF = CB->getParent()->getParent();
    BasicBlock &BB = CBF->getEntryBlock();
    IRBuilder<> Builder(&*BB.getFirstInsertionPt());

    // Clumsy call here is to set a specific alignment on the struct instance
    auto alloced =
        Builder.Insert(new AllocaInst(VarargsTy, DL.getAllocaAddrSpace(),
                                      nullptr, MaxFieldAlign),
                       "vararg_buffer");

    // Initialise the fields in the struct
    // TODO: Lifetime annotate it and alloca in entry
    // Needs to start life shortly before these copies and end immediately after
    // the new call instruction
    Builder.SetInsertPoint(CB);
    for (size_t i = 0; i < Varargs.size(); i++) {
      auto r = Builder.CreateStructGEP(VarargsTy, alloced, Varargs[i].second);
      if (isbyval[Varargs[i].second]) {
        Type *ByValType = LocalVarTypes[Varargs[i].second];
        Builder.CreateMemCpy(r, {}, Varargs[i].first, {},
                             DL.getTypeAllocSize(ByValType).getFixedValue());
      } else {
        Builder.CreateStore(Varargs[i].first, r);
      }
    }

    // Initialise a va_list pointing to that struct and pass it as the last
    // argument
    {
      PointerType *voidptr = PointerType::getUnqual(Ctx);
      Value *voidBuffer =
          Builder.CreatePointerBitCastOrAddrSpaceCast(alloced, voidptr);

      if (ABI->valistOnStack()) {
        // or byval? not really worth memcpy'ing it since it dies immediately
        // after the call
        assert(ABI->vaListCC() == ABI::valistCC::pointer);
        Type *va_list_ty = ABI->vaListType(Ctx);

        // todo: one va_list alloca per function, also lifetime annotate
        AllocaInst *va_list =
            Builder.CreateAlloca(va_list_ty, nullptr, "va_list");

        ABI->initializeVAList(Ctx, Builder, va_list, voidBuffer);
        Args.push_back(va_list);
      } else {
        assert(ABI->vaListCC() == ABI::valistCC::value);
        Args.push_back(voidBuffer);
      }
    }

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
      // memory along instead when the function is marked tailcall
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
    // This is a hold over from a previous design that wanted to be target
    // independent at this point, lets just assume the alignment is consistent
    // with the architecture
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
