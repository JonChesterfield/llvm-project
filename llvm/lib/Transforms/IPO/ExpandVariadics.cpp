//===-- ExpandVariadicsPass.cpp --------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is an optimisation pass for variadic functions. If called from codegen,
// it can serve as the implementation of variadic functions for a given target.
//
// The target-dependent parts are in namespace VariadicABIInfo. Enabling a new
// target means adding a case to VariadicABIInfo::create() along with tests.
//
// The module pass using that information is class ExpandVariadics.
//
// The strategy is:
// 1. Test whether a variadic function is sufficiently simple
// 2. If it was, calls to it can be replaced with calls to a different function
// 3. If it wasn't, try to split it into a simple function and a remainder
// 4. Optionally rewrite the varadic function calling convention as well
//
// This pass considers "sufficiently simple" to mean a variadic function that
// calls into a different function taking a va_list to do the real work. For
// example, libc might implement fprintf as a single basic block calling into
// vfprintf. This pass can then rewrite call to the variadic into some code
// to construct a target-specific value to use for the va_list and a call
// into the non-variadic implementation function. There's a test for that.
//
// Most other variadic functions whose definition is known can be converted into
// that form. Create a new internal function taking a va_list where the original
// took a ... parameter. Move the blocks across. Create a new block containing a
// va_start that calls into the new function. This is nearly target independent.
//
// Where this transform is consistent with the ABI, e.g. AMDGPU or NVPTX, or
// where the ABI can be chosen to align with this transform, the function
// interface can be rewritten along with calls to unknown variadic functions.
//
// The aggregate effect is to unblock other transforms, most critically the
// general purpose inliner. Known calls to variadic functions become zero cost.
//
// This pass does define some target specific information which is partially
// redundant with other parts of the compiler. In particular, the call frame
// it builds must be the exact complement of the va_arg lowering performed
// by clang. The va_list construction is similar to work done by the backend
// for targets that lower variadics there, though distinct in that this pass
// constructs the pieces using alloca instead of relative to stack pointers.
//
// Consistency with clang is primarily tested by emitting va_arg using clang
// then expanding the variadic functions using this pass, followed by trying
// to constant fold the functions to no-ops.
//
// Target specific behaviour is tested in IR - mainly checking that values are
// put into positions in call frames that make sense for that particular target.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/ExpandVariadics.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TargetParser/Triple.h"

#include <cstdio>

#define DEBUG_TYPE "expand-variadics"

using namespace llvm;

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
namespace VariadicABIInfo {

// calling convention for passing as valist object, same as it would be in C
// aarch64 uses byval
enum class ValistCc { value, pointer, /*byval*/ };

struct Interface {
protected:
  Interface(uint32_t MinAlign, uint32_t MaxAlign)
      : MinAlign(MinAlign), MaxAlign(MaxAlign) {}

public:
  virtual ~Interface() {}
  const uint32_t MinAlign;
  const uint32_t MaxAlign;

  // Most ABIs use a void* or char* for va_list, others can specialise
  virtual Type *vaListType(LLVMContext &Ctx) {
    return PointerType::getUnqual(Ctx);
  }

  // Lots of targets use a void* pointed at a buffer for va_list.
  // Some use more complicated iterator constructs.
  // This interface seeks to express both.
  // Ideally it would be a compile time error for a derived class
  // to override only one of valistOnStack, initializeVAList.

  // How the vaListType is passed
  virtual ValistCc valistCc() { return ValistCc::value; }

  // The valist might need to be stack allocated.
  virtual bool valistOnStack() { return false; }

  virtual void initializeVAList(LLVMContext &Ctx, IRBuilder<> &Builder,
                                AllocaInst * /*va_list*/, Value * /*buffer*/) {
    // Function needs to be implemented iff valist is on the stack.
    assert(!valistOnStack());
    llvm_unreachable("Only called if valistOnStack() returns true");
  }

  // All targets currently implemented use a ptr for the valist parameter
  Type *vaListParameterType(LLVMContext &Ctx) {
    return PointerType::getUnqual(Ctx);
  }

  bool vaEndIsNop() { return true; }

  bool vaCopyIsMemcpy() { return true; }
};

struct X64SystemV final : public Interface {
  // X64 documented behaviour:
  // Slots are at least eight byte aligned and at most 16 byte aligned.
  // If the type needs more than sixteen byte alignment, it still only gets
  // that much alignment on the stack.
  // X64 behaviour in clang:
  // Slots are at least eight byte aligned and at most naturally aligned
  // This matches clang, not the ABI docs.
  X64SystemV() : Interface(8, 0) {}

  Type *vaListType(LLVMContext &Ctx) override {
    auto I32 = Type::getInt32Ty(Ctx);
    auto Ptr = PointerType::getUnqual(Ctx);
    return ArrayType::get(StructType::get(Ctx, {I32, I32, Ptr, Ptr}), 1);
  }
  ValistCc valistCc() override { return ValistCc::pointer; }

  bool valistOnStack() override { return true; }

  void initializeVAList(LLVMContext &Ctx, IRBuilder<> &Builder,
                        AllocaInst *VaList, Value *VoidBuffer) override {
    assert(valistOnStack());
    assert(VaList != nullptr);
    assert(VaList->getAllocatedType() == vaListType(Ctx));

    Type *VaListTy = vaListType(Ctx);

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
        Builder.CreateInBoundsGEP(VaListTy, VaList, Idxs, "gp_offset"));

    Idxs[2] = ConstantInt::get(I32, 1);
    Builder.CreateStore(
        ConstantInt::get(I32, 6 * 8 + 8 * 16),
        Builder.CreateInBoundsGEP(VaListTy, VaList, Idxs, "fp_offset"));

    Idxs[2] = ConstantInt::get(I32, 2);
    Builder.CreateStore(
        VoidBuffer,
        Builder.CreateInBoundsGEP(VaListTy, VaList, Idxs, "overfow_arg_area"));

    Idxs[2] = ConstantInt::get(I32, 3);
    Builder.CreateStore(
        ConstantPointerNull::get(PointerType::getUnqual(Ctx)),
        Builder.CreateInBoundsGEP(VaListTy, VaList, Idxs, "reg_save_area"));
  }
};

std::unique_ptr<Interface> create(Module &M) {
  llvm::Triple Triple(M.getTargetTriple());
  const bool IsLinuxABI = Triple.isOSLinux() || Triple.isOSCygMing();

  switch (Triple.getArch()) {

  case Triple::r600:
  case Triple::amdgcn: {
    struct AMDGPU final : public Interface {
      AMDGPU() : Interface(1, 0) {}
    };
    return std::make_unique<AMDGPU>();
  }

  case Triple::nvptx:
  case Triple::nvptx64: {
    struct NVPTX final : public Interface {
      NVPTX() : Interface(4, 0) {}
    };
    return std::make_unique<NVPTX>();
  }

  case Triple::x86: {
    // These seem to all fall out the same, despite getTypeStackAlign
    // implying otherwise.
    if (Triple.isOSDarwin()) {
      struct X86Darwin final : public Interface {
        // X86_32ABIInfo::getTypeStackAlignInBytes is misleading for this.
        // The slotSize(4) implies a minimum alignment
        // The AllowHigherAlign = true means there is no maximum alignment.
        X86Darwin() : Interface(4, 0) {}
      };

      return std::make_unique<X86Darwin>();
    }
    if (Triple.getOS() == llvm::Triple::Win32) {
      struct X86Windows final : public Interface {
        X86Windows() : Interface(4, 0) {}
      };
      return std::make_unique<X86Windows>();
    }

    if (IsLinuxABI) {
      struct X86Linux final : public Interface {
        X86Linux() : Interface(4, 0) {}
      };
      return std::make_unique<X86Linux>();
    }
    break;
  }

  case Triple::x86_64: {
    if (Triple.isWindowsMSVCEnvironment() || Triple.isOSWindows()) {
      struct X64Windows final : public Interface {
        X64Windows() : Interface(8, 8) {}
      };
      // x64 msvc emit vaarg passes > 8 byte values by pointer
      // however the variadic call instruction created does not, e.g.
      // a <4 x f32> will be passed as itself, not as a pointer or byval.
      // Postponing resolution of that for now.
      return nullptr;
    }

    if (Triple.isOSDarwin()) {
      return std::make_unique<VariadicABIInfo::X64SystemV>();
    }

    if (IsLinuxABI) {
      return std::make_unique<VariadicABIInfo::X64SystemV>();
    }

    break;
  }

  default:
    return nullptr;
  }

  return nullptr;
}

} // namespace VariadicABIInfo

class ExpandVariadics : public ModulePass {
public:
  static char ID;
  const bool AllTransformsEnabled;
  std::unique_ptr<VariadicABIInfo::Interface> ABI;
  DenseMap<Function *, Function *> functionToInliningTarget;

  ExpandVariadics(bool A = false) : ModulePass(ID), AllTransformsEnabled(A) {}
  StringRef getPassName() const override { return "Expand variadic functions"; }

  // A predicate in that return nullptr means false
  // Returns the function target to use when inlining on success
  Function *isFunctionInlinable(Module &M, Function *F);

  // Rewrite a call site.
  void expandCall(Module &M, CallBase *CB, FunctionType *, Function *NF);

  // For variadic functions that fail isFunctionInlinable, split into two
  // functions, one of which can be inlined using the other.
  Function *DeriveInlinableVariadicFunctionPair(Module &M, Function &F);

  bool splitFunctions() { return SplitFunctions | rewriteABI(); }
  bool replaceCalls() { return ReplaceCalls | rewriteABI(); }
  bool replaceOperations() { return ReplaceOperations | rewriteABI(); }
  bool rewriteABI() { return RewriteABI | AllTransformsEnabled; }

  bool conservative() { return !rewriteABI(); }

  void MemcpyVAListPointers(const DataLayout &DL, IRBuilder<> &Builder,
                            Value *dst, Value *src) {
    auto &Ctx = Builder.getContext();
    Type *VaListTy = ABI->vaListType(Ctx);
    uint64_t size = DL.getTypeAllocSize(VaListTy).getFixedValue();
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

  FunctionType * inlinableVariadicFunctionType(Module &M,
                                               FunctionType *FTy) {
    auto &Ctx = M.getContext();
    SmallVector<Type *> ArgTypes(FTy->param_begin(), FTy->param_end());
    ArgTypes.push_back(ABI->vaListParameterType(Ctx));
    return
      FunctionType::get(FTy->getReturnType(), ArgTypes, /*IsVarArgs*/ false);
  }
  
  // this could be partially target specific
  bool expansionApplicableToFunction(Module &M, Function *F) {
    if (F->isIntrinsic() || !F->isVarArg() ||
        F->hasFnAttribute(Attribute::Naked)) {
      return false;
    }

    if (F->getCallingConv() != CallingConv::C)
      return false;

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

        if (!CB)
          return false;

        if (CB->isMustTailCall())
          return false;

        if (!CB->isCallee(&U) || CB->getFunctionType() != F->getFunctionType()) {
          return false;
        }

      }

    // Branch funnels look like variadic functions but aren't:
    //
    // define hidden void @__typeid_typeid1_0_branch_funnel(ptr nest %0, ...) {
    //  musttail call void (...) @llvm.icall.branch.funnel(ptr %0, ptr @vt1_1,
    //  ptr @vf1_1, ...) ret void
    // }
    //
    // %1 = call i32 @__typeid_typeid1_0_branch_funnel(ptr nest %vtable, ptr
    // %obj, i32 1)


      // TODO: there should be a reasonable way to check for an intrinsic
      // without inserting a prototype that then needs to be removed
      Function *Funnel =
          Intrinsic::getDeclaration(&M, Intrinsic::icall_branch_funnel);
      for (const User *U : Funnel->users()) {
        if (auto *I = dyn_cast<CallBase>(U)) {
          if (F == I->getFunction()) {
            return false;
          }
        }
      }
      if (Funnel->use_empty())
        Funnel->eraseFromParent();
    }

    return true;
  }

  // This should probably only skip if the size and pointer values are right
  template <Intrinsic::ID ID>
  static BasicBlock::iterator
  skipIfInstructionIsSpecificIntrinsic(BasicBlock::iterator Iter) {
    if (auto *Intrinsic = dyn_cast<IntrinsicInst>(&*Iter))
      if (Intrinsic->getIntrinsicID() == ID)
        Iter++;
    return Iter;
  }

  bool callinstRewritable(CallBase *CB) {

    if (CallInst *CI = dyn_cast<CallInst>(CB)) {
      if (CI->isMustTailCall()) {
        // Cannot expand musttail calls
        if (rewriteABI()) {
          report_fatal_error("Cannot lower musttail variadic call");
        } else {
          return false;
        }
      }
    }

    return true;
  }

  bool runOnFunction(Module &M, Function *F) {
    bool Changed = false;

    // fprintf(stderr, "Called on %s\n", F->getName().str().c_str());

    // This check might be too coarse - there are probably cases where
    // splitting a function is bad but it's usable without splitting
    if (!expansionApplicableToFunction(M, F))
      return false;

    // TODO: Leave "thunk" attribute functions alone?

    bool usefulToSplit =
        splitFunctions() && (!F->isDeclaration() || rewriteABI());

    // F may already be a single basic block calling a known function
    // that takes a va_list, in which case it doens't need to be split.
    Function *Equivalent = isFunctionInlinable(M, F);

    if (usefulToSplit && !Equivalent) {
      Equivalent = DeriveInlinableVariadicFunctionPair(M, *F);
      assert(Equivalent);
      Changed = true;
      functionToInliningTarget[F] = Equivalent;
    }

    if (rewriteABI() && !Equivalent) {
      report_fatal_error("ExpandVA abi requires replacement function\n");
    }

    if (replaceCalls()) {
      if (!Equivalent) {
        assert(!rewriteABI());
        return Changed;
      }

      for (User *U : llvm::make_early_inc_range(F->users()))
        // Need to handle invoke etc, or at least deliberately ignore them on
        // the optimise path
        // TODO: A test where the call instruction takes a variadic function as
        // a parameter other than the one it is calling
        if (CallBase *CB = dyn_cast<CallBase>(U)) {
          Value *calledOperand = CB->getCalledOperand();
          if (F == calledOperand) {
            expandCall(M, CB, F->getFunctionType(), Equivalent);
            Changed = true;
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

  bool runOnModule(Module &M) override {
    if (DisablePass)
      return false;

    // Can get at triple, still need to distinguish x86 variants
    // Generally mistrusting of the constructor call time given the two pass
    // managers
    const DataLayout &DL = M.getDataLayout();

    ABI = VariadicABIInfo::create(M);
    if (!ABI)
      return false;

    if (0)
      fprintf(stderr, "ExpandVA: split %u, calls %u, op %u, abi %u\n",
              splitFunctions(), replaceCalls(), replaceOperations(),
              rewriteABI());

    bool Changed = false;

    // TODO: Should call on functions that contain va_start with abi changes
    // disabled
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
                  expandCall(M, CB, FTy, 0);
                }
              }
            }
          }
        }
      }
    }

    return Changed;
  }
};

Function *ExpandVariadics::isFunctionInlinable(Module &M, Function *F) {
  assert(F->isVarArg());
  assert(expansionApplicableToFunction(M, F));

  {
    // Already in the cache (todo: maybe drop for the pure inlining branch)
    auto it = functionToInliningTarget.find(F);
    if (it != functionToInliningTarget.end())
      return it->second;
  }

  if (F->isDeclaration())
    return nullptr;

  // A variadic function is inlinable if it is sufficiently simple.
  // Specifically, if it is a single basic block which is functionally
  // equivalent to packing the variadic arguments into a va_list which is
  // passed to another function. The inlining strategy is to build a va_list
  // in the caller and then call said inner function.

  // Single basic block.
  BasicBlock &BB = F->getEntryBlock();
  if (!isa<ReturnInst>(BB.getTerminator()))
    return nullptr;

  // Walk the block in order checking for specific instructions, some of them
  // optional.
  BasicBlock::iterator Iter = BB.begin();

  AllocaInst *Alloca = dyn_cast<AllocaInst>(&*Iter++);
  if (!Alloca)
    return nullptr;

  Value *ValistArgument = Alloca;

  Iter = skipIfInstructionIsSpecificIntrinsic<Intrinsic::lifetime_start>(Iter);

  VAStartInst *Start = dyn_cast<VAStartInst>(&*Iter++);
  if (!Start || Start->getArgList() != ValistArgument) {
    return nullptr;
  }

  // The va_list instance is stack allocated
  // The ... replacement is a va_list passed "by value"
  // That involves a load for some ABIs and passing the pointer for others
  Value *ValistTrailingArgument = nullptr;
  switch (ABI->valistCc()) {
  case VariadicABIInfo::ValistCc::value: {
    // If it's being passed by value, need a load
    // TODO: Check it's loading the right thing
    auto *load = dyn_cast<LoadInst>(&*Iter);
    if (!load)
      return nullptr;
    ValistTrailingArgument = load;
    Iter++;
    break;
  }
  case VariadicABIInfo::ValistCc::pointer: {
    // If it's being passed by pointer, going to use the alloca directly
    ValistTrailingArgument = ValistArgument;
    break;
  }
  }

  CallInst *Call = dyn_cast<CallInst>(&*Iter++);
  if (!Call)
    return nullptr;

  if (auto *end = dyn_cast<VAEndInst>(&*Iter)) {
    if (end->getArgList() != ValistArgument)
      return nullptr;
    Iter++;
  } else {
    // Only fail on a missing va_end if it wasn't a no-op
    if (!ABI->vaEndIsNop())
      return nullptr;
  }

  Iter = skipIfInstructionIsSpecificIntrinsic<Intrinsic::lifetime_end>(Iter);

  ReturnInst *Ret = dyn_cast<ReturnInst>(&*Iter++);
  if (!Ret || Iter != BB.end())
    return nullptr;

  // The function call is expected to take the fixed arguments then the alloca
  // TODO: Drop the vectors here, iterate over them both together instead.
  SmallVector<Value *> FuncArgs;
  for (Argument &A : F->args())
    FuncArgs.push_back(&A);

  SmallVector<Value *> CallArgs;
  for (Use &A : Call->args())
    CallArgs.push_back(A);

  size_t Fixed = FuncArgs.size();
  if (Fixed + 1 != CallArgs.size())
    return nullptr;

  for (size_t i = 0; i < Fixed; i++)
    if (FuncArgs[i] != CallArgs[i])
      return nullptr;

  if (CallArgs[Fixed] != ValistTrailingArgument)
    return nullptr;

  // Check the varadic function returns the result of the inner call
  Value *MaybeReturnValue = Ret->getReturnValue();
  if (Call->getType()->isVoidTy()) {
    if (MaybeReturnValue != nullptr)
      return nullptr;
  } else {
    if (MaybeReturnValue != Call)
      return nullptr;
  }

  // All checks passed. Found a va_list taking function we can use.
  Function *Equivalent = Call->getCalledFunction();
  if (Equivalent)
    functionToInliningTarget[F] = Equivalent;
  return Equivalent;
}


  
Function *ExpandVariadics::DeriveInlinableVariadicFunctionPair(Module &M,
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
  ArgTypes.push_back(ABI->vaListParameterType(Ctx));

  FunctionType *NFTy = inlinableVariadicFunctionType(M, F.getFunctionType());
  Function *NF = Function::Create(NFTy, F.getLinkage(), F.getAddressSpace());

  // Note - same attribute handling as DeadArgumentElimination
  NF->copyAttributesFrom(&F);
  NF->setComdat(F.getComdat()); // beware weak
  F.getParent()->getFunctionList().insert(F.getIterator(), NF);
  NF->setName(F.getName() + ".valist");
  NF->IsNewDbgInfoFormat = F.IsNewDbgInfoFormat;

  // New function is default visibility and internal
  // Need to set visibility before linkage to avoid an assert in setVisibility
  NF->setVisibility(GlobalValue::DefaultVisibility);
  NF->setLinkage(GlobalValue::InternalLinkage);

  AttrBuilder ParamAttrs(Ctx);
  ParamAttrs.addAttribute(Attribute::NoAlias);

  // TODO: When can the va_list argument have addAlignmentAttr called on it?
  // It improves codegen lot in the non-inlined case. Probably target
  // specific.

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
    // Is it better to make this a va_copy and deal with that later? Involves
    // a spurious Alloca
    for (BasicBlock &BB : *NF)
      for (Instruction &I : llvm::make_early_inc_range(BB))
        if (VAStartInst *II = dyn_cast<VAStartInst>(&I)) {
          Builder.SetInsertPoint(II);

          // va_start takes a pointer to a va_list, e.g. one on the stack.
          // Retrieve it:
          Value *va_start_arg = II->getArgList();

          // The last argument is a vaListParameterType
          Value *passed_va_list = NewArg;

          switch (ABI->valistCc()) {
          case VariadicABIInfo::ValistCc::value: {
            // Got a va_list in an ssa register
            Builder.CreateStore(passed_va_list, va_start_arg);
            break;
          }
          case VariadicABIInfo::ValistCc::pointer: {
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
    assert(F.isDeclaration()); // The blocks have been stolen, it looks like a
                               // declaration now
    Type *VaListTy = ABI->vaListType(Ctx);

    auto *BB = BasicBlock::Create(Ctx, "entry", &F);
    Builder.SetInsertPoint(BB);

    Value *va_list_instance =
        Builder.CreateAlloca(VaListTy, nullptr, "va_list");

    // Alloca puts it in the default stack addrspace and vastart doesn't
    // currently accept that.
    va_list_instance = Builder.CreatePointerBitCastOrAddrSpaceCast(
        va_list_instance, PointerType::getUnqual(Ctx));

    Builder.CreateIntrinsic(Intrinsic::vastart, {}, {va_list_instance});

    SmallVector<Value *> args;
    for (Argument &arg : F.args())
      args.push_back(&arg);

    args.push_back(va_list_instance);

    CallInst *Result = Builder.CreateCall(NF, args);
    Result->setTailCallKind(CallInst::TCK_Tail);

    assert(ABI->vaEndIsNop()); // If this changes, insert a va_end here

    if (Result->getType()->isVoidTy())
      Builder.CreateRetVoid();
    else
      Builder.CreateRet(Result);
  }

  assert(F.isDeclaration() == NF->isDeclaration());
  return NF;
}


void ExpandVariadics::expandCall(Module &M, CallBase *CB, FunctionType *VarargFunctionType,
                                 Function *NF) {
  const DataLayout &DL = M.getDataLayout();

  if (!callinstRewritable(CB)) {
    return;
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
  if (FuncType != VarargFunctionType) {
    if (!rewriteABI()) {
      return;
    }
    FuncType = VarargFunctionType;
  }

  auto &Ctx = CB->getContext();

  // Align the struct on ABI->MinAlign to start with
  Align MaxFieldAlign(ABI->MinAlign ? ABI->MinAlign : 1);

  // The strategy here is to allocate a call frame containing the variadic
  // arguments laid out such that a target specific va_list can be initialised
  // with it, such that target specific va_arg instructions will correctly
  // iterate over it. Primarily this means getting the alignment right.

  class {
    // The awkward memory layout is to allow access to a contiguous array of
    // types
    enum { N = 4 };
    SmallVector<Type *, N> FieldTypes;
    SmallVector<std::pair<Value *, bool>, N> maybeValueIsByval;

  public:
    void append(Type *T, Value *V, bool IsByVal) {
      FieldTypes.push_back(T);
      maybeValueIsByval.push_back({V, IsByVal});
    }

    void padding(LLVMContext &Ctx, uint64_t By) {
      append(ArrayType::get(Type::getInt8Ty(Ctx), By), nullptr, false);
    }

    size_t size() { return FieldTypes.size(); }
    bool empty() { return FieldTypes.empty(); }

    StructType *asStruct(LLVMContext &Ctx, StringRef Name) {
      const bool IsPacked = true;
      return StructType::create(Ctx, FieldTypes,
                                (Twine(Name) + ".vararg").str(), IsPacked);
    }

    void initialiseStructAlloca(const DataLayout &DL, IRBuilder<> &Builder,
                                AllocaInst *Alloced) {

      StructType *VarargsTy = cast<StructType>(Alloced->getAllocatedType());

      for (size_t i = 0; i < size(); i++) {
        auto [v, IsByVal] = maybeValueIsByval[i];
        if (!v)
          continue;

        auto r = Builder.CreateStructGEP(VarargsTy, Alloced, i);
        if (IsByVal) {
          Type *ByValType = FieldTypes[i];
          Builder.CreateMemCpy(r, {}, v, {},
                               DL.getTypeAllocSize(ByValType).getFixedValue());
        } else {
          Builder.CreateStore(v, r);
        }
      }
    }
  } Frame;

  uint64_t CurrentOffset = 0;
  for (unsigned I = FuncType->getNumParams(), E = CB->arg_size(); I < E; ++I) {
    Value *ArgVal = CB->getArgOperand(I);
    bool IsByVal = CB->paramHasAttr(I, Attribute::ByVal);
    Type *ArgType = IsByVal ? CB->getParamByValType(I) : ArgVal->getType();
    Align DataAlign = DL.getABITypeAlign(ArgType);

    uint64_t DataAlignV = DataAlign.value();

    // Currently using 0 as a sentinel to mean ignored
    if (ABI->MinAlign && DataAlignV < ABI->MinAlign)
      DataAlignV = ABI->MinAlign;
    if (ABI->MaxAlign && DataAlignV > ABI->MaxAlign)
      DataAlignV = ABI->MaxAlign;

    DataAlign = Align(DataAlignV);
    MaxFieldAlign = std::max(MaxFieldAlign, DataAlign);

    if (uint64_t Rem = CurrentOffset % DataAlignV) {
      // Inject explicit padding to deal with alignment requirements
      uint64_t Padding = DataAlignV - Rem;
      Frame.padding(Ctx, Padding);
      CurrentOffset += Padding;
    }

    Frame.append(ArgType, ArgVal, IsByVal);
    CurrentOffset += DL.getTypeAllocSize(ArgType).getFixedValue();
  }

  if (Frame.empty()) {
    // Not passing anything, hopefully va_arg won't try to dereference it
    // Might be a target specific thing whether one can pass nullptr instead
    // of undef i32
    Frame.append(Type::getInt32Ty(Ctx), nullptr, false);
  }

  Function *CBF = CB->getParent()->getParent();

  StructType *VarargsTy = Frame.asStruct(Ctx, CBF->getName());

  BasicBlock &BB = CBF->getEntryBlock();
  IRBuilder<> Builder(&*BB.getFirstInsertionPt());

  // Clumsy call here is to set a specific alignment on the struct instance
  AllocaInst *Alloced =
      Builder.Insert(new AllocaInst(VarargsTy, DL.getAllocaAddrSpace(), nullptr,
                                    MaxFieldAlign),
                     "vararg_buffer");
  assert(Alloced->getAllocatedType() == VarargsTy);

  // Initialise the fields in the struct
  // TODO: Lifetime annotate it and alloca in entry
  // Needs to start life shortly before these copies and end immediately after
  // the new call instruction
  Builder.SetInsertPoint(CB);

  Frame.initialiseStructAlloca(DL, Builder, Alloced);

  unsigned NumArgs = FuncType->getNumParams();

  SmallVector<Value *> Args;
  Args.assign(CB->arg_begin(), CB->arg_begin() + NumArgs);

  // Initialise a va_list pointing to that struct and pass it as the last
  // argument
  {
    PointerType *Voidptr = PointerType::getUnqual(Ctx);
    Value *VoidBuffer =
        Builder.CreatePointerBitCastOrAddrSpaceCast(Alloced, Voidptr);

    if (ABI->valistOnStack()) {
      assert(ABI->valistCc() == VariadicABIInfo::ValistCc::pointer);
      Type *VaListTy = ABI->vaListType(Ctx);

      // TODO: one va_list alloca per function, also lifetime annotate
      AllocaInst *VaList = Builder.CreateAlloca(VaListTy, nullptr, "va_list");

      ABI->initializeVAList(Ctx, Builder, VaList, VoidBuffer);
      Args.push_back(VaList);
    } else {
      assert(ABI->valistCc() == VariadicABIInfo::ValistCc::value);
      Args.push_back(VoidBuffer);
    }
  }

  // Attributes excluding any on the vararg arguments
  AttributeList PAL = CB->getAttributes();
  if (!PAL.isEmpty()) {
    SmallVector<AttributeSet, 8> ArgAttrs;
    for (unsigned ArgNo = 0; ArgNo < NumArgs; ArgNo++)
      ArgAttrs.push_back(PAL.getParamAttrs(ArgNo));
    PAL =
        AttributeList::get(Ctx, PAL.getFnAttrs(), PAL.getRetAttrs(), ArgAttrs);
  }

  SmallVector<OperandBundleDef, 1> OpBundles;
  CB->getOperandBundlesAsDefs(OpBundles);
  
  CallBase *NewCB = nullptr;
  // TODO, other instructions
  if (CallInst *CI = dyn_cast<CallInst>(CB)) {

    Value * dst = NF ? NF : CI->getCalledOperand();
    FunctionType *NFTy = inlinableVariadicFunctionType(M, VarargFunctionType);
    NewCB = CallInst::Create(NFTy, dst, Args, OpBundles, "", CI);

    CallInst::TailCallKind TCK = CI->getTailCallKind();
    assert(TCK != CallInst::TCK_MustTail); // guarded at prologue

    // It doesn't get to be a tail call any more
    // might want to guard this with arch, x64 and aarch64 document that
    // varargs can't be tail called anyway
    // Not totally convinced this is necessary but dead store elimination
    // decides to discard the stores to the Alloca and pass uninitialised
    // memory along instead when the function is marked tailcall
    if (TCK == CallInst::TCK_Tail) {
      TCK = CallInst::TCK_None;
    }
    CI->setTailCallKind(TCK);

    
  } else if (InvokeInst *II = dyn_cast<InvokeInst>(CB)) {
    assert(NF);
    NewCB = InvokeInst::Create(NF, II->getNormalDest(), II->getUnwindDest(),
                               Args, OpBundles, "", CB);
  } else {

    assert(0);
    
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

} // namespace

char ExpandVariadics::ID = 0;

INITIALIZE_PASS(ExpandVariadics, DEBUG_TYPE, "Expand variadic functions", false,
                false)

ModulePass *llvm::createExpandVariadicsPass(bool RewriteABI) {
  return new ExpandVariadics(RewriteABI);
}

PreservedAnalyses ExpandVariadicsPass::run(Module &M, ModuleAnalysisManager &) {
  return ExpandVariadics(false).runOnModule(M) ? PreservedAnalyses::none()
                                               : PreservedAnalyses::all();
}
