//===-- ExpandVAIntrinsicsPass.cpp --------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ExpandVAIntrinsics.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <cstdio>

#define DEBUG_TYPE "expand-va-intrinsics"

using namespace llvm;

namespace {

class ExpandVAIntrinsics : public ModulePass {
public:
  static char ID;

  ExpandVAIntrinsics() : ModulePass(ID) {
    // may not need this initialize call here
    initializeExpandVAIntrinsicsPass(*PassRegistry::getPassRegistry());
  }


  enum {verbose = false};
  
  static void ExpandVAArg(VAArgInst *Inst, const DataLayout &DL) {
    // todo: review
    // the align sequence seems messy but does coinstant fold
    // the getType vs getValueType needs checking
    Type *IntPtrTy = DL.getIntPtrType(Inst->getContext());
    auto *One = ConstantInt::get(IntPtrTy, 1);
    IRBuilder<> IRB(Inst);
  
    if (verbose) {  printf("Inst\n");
      Inst->dump();}
    

    // Probably no longer necessary. The instruction is va_arg ptr %v, i64 or similar
  auto *ArgList = IRB.CreateBitCast(
      Inst->getPointerOperand(),
      Inst->getType()->getPointerTo()->getPointerTo(), "arglist");

      if (verbose) {
  printf("ArgList\n");
  ArgList->dump();}
  
  // The caller spilled all of the va_args onto the stack in an unpacked
  // struct. Each va_arg load from that struct needs to realign the element to
  // its target-appropriate alignment in the struct in order to jump over
  // padding that may have been in-between arguments. Do this with ConstantExpr
  // to ensure good code gets generated, following the same approach as
  // Support/MathExtras.h:alignAddr:
  //   ((uintptr_t)Addr + Alignment - 1) & ~(uintptr_t)(Alignment - 1)
  // This assumes the alignment of the type is a power of 2 (or 1, in which case
  // no realignment occurs).
  
  auto *Ptr = IRB.CreateLoad(ArgList->getType(), ArgList, "arglist_current");
      if (verbose) {
  printf("ptr\n");
  Ptr->dump();}
  
  auto *AlignOf = ConstantExpr::getIntegerCast(
      ConstantExpr::getAlignOf(Inst->getType()), IntPtrTy, /*isSigned=*/false);
  auto *AlignMinus1 = ConstantExpr::getNUWSub(AlignOf, One);
  auto *NotAlignMinus1 = IRB.CreateNot(AlignMinus1);
  auto *CurrentPtr = IRB.CreateIntToPtr(
      IRB.CreateAnd(
          IRB.CreateNUWAdd(IRB.CreatePtrToInt(Ptr, IntPtrTy), AlignMinus1),
          NotAlignMinus1),
      Ptr->getType());

  auto *Result = IRB.CreateLoad(Inst->getType(), CurrentPtr, "va_arg");
  Result->takeName(Inst);

  // Update the va_list to point to the next argument.
  Value *Indexes[] = {One};
  auto *Next = IRB.CreateInBoundsGEP(CurrentPtr->getType(), CurrentPtr, Indexes, "arglist_next");
  IRB.CreateStore(Next, ArgList);

      if (verbose) {
  printf("result\n");
  Result->dump();
  Result->getType()->dump();
  
  Next->dump();
  Next->getType()->dump();
  fflush(stdout);
      }
      
  Inst->replaceAllUsesWith(Result);
  Inst->eraseFromParent();
}

  static void ExpandVAStart(VAStartInst *Inst, Argument *StructPtr) {
    IRBuilder<> Builder(Inst);
    Builder.CreateStore(StructPtr, Inst->getArgList());
    Inst->eraseFromParent();
  }

  static void ExpandVACopy(VACopyInst *Inst) {
    IRBuilder<> Builder(Inst);
    Value *dst = Inst->getDest();
    Value *src = Inst->getSrc();
    Value *ld = Builder.CreateLoad(src->getType(), src, "vacopy");
    Builder.CreateStore(ld, dst);
    Inst->eraseFromParent();
  }

  static void ExpandVAEnd(VAEndInst *Inst) {
    Inst->eraseFromParent();
  }


  bool runOnModule(Module &M) override {
    // va_list is a void* or similar
    // it is initialised by va_start, or a copy of some previous va_list
    // nothing in this pass changes the definition of va_list
    
    bool Changed = false;
    const DataLayout &DL = M.getDataLayout();
    for (Function &F : llvm::make_early_inc_range(M)) {
      for (BasicBlock &BB : F) {
        for (Instruction &I : llvm::make_early_inc_range(BB)) {
          // Expand the operations that can occur in any function
          if (VAArgInst *II = dyn_cast<VAArgInst>(&I)) {
            Changed = true;
            ExpandVAArg(II,DL);
            continue;
          }
          if (VAEndInst *II = dyn_cast<VAEndInst>(&I)) {
            Changed = true;
            ExpandVAEnd(II);
            continue;
          }
          if (VACopyInst *II = dyn_cast<VACopyInst>(&I)) {
            Changed = true;
            ExpandVACopy(II);
            continue;
          }
        }
      }
      
      if (F.isVarArg()) {
        // Rewrites declarations, definitions, calls, va_start.
        runOnVarargFunction(M, F);
        Changed = true;
      }
      
    }
    return Changed;
  }
  
  static void ExpandCall(Module &M, CallBase* CB, Function *NF, unsigned NumFixedArgs) {
    const DataLayout &DL = M.getDataLayout();

    // Given a CallBase to a variadic function, replace it with a call to a non-variadic
    // function.

    FunctionType *FuncType = CB->getFunctionType();
    
    auto &Ctx = CB->getContext();
        
      // TODO: Deal with attributes on the varargs part, see DAE
      // Need to make the struct and stash things in it

    unsigned NumArgs = FuncType->getNumParams();
    
      std::vector<Value *> Args;
      Args.assign(CB->arg_begin(), CB->arg_begin() + NumArgs);

      std::vector<Value *> Varargs;
      std::vector<Type *> LocalVarTypes;



      
      for (unsigned I = FuncType->getNumParams(), E = CB->arg_size();
           I < E; ++I) {
        Value* ArgVal = CB->getArgOperand(I);
        Varargs.push_back(ArgVal);

        // byval a hazard here?
        LocalVarTypes.push_back(ArgVal->getType());

      }


      
      if (LocalVarTypes.empty()) {
        // oh dear, todo
        // pass nullptr?        
      }

      // Not obvious how to set a minimum alignment on this struct
      // but that is necessary to assume it is at least stack aligned
      // Might involve setting alignment on first field
      // Also, this create thing probably doesn't align fields natively,
      // had to DIY that in lower lds
      // assumedAlignment(DL)
      StructType *VarargsTy = StructType::create(Ctx, LocalVarTypes, "todo");
      
      
      Function* CBF = CB->getParent()->getParent();
      BasicBlock &BB = CBF->getEntryBlock();
      IRBuilder<> Builder(&*BB.getFirstInsertionPt());

      auto alloced =
          Builder.Insert(new AllocaInst(VarargsTy, DL.getAllocaAddrSpace(),
                                        nullptr, assumedAlignment(DL)),
                         "vararg_buffer");

      // TODO: Lifetime annotate it, if that works on AS(5) now
      Builder.SetInsertPoint(CB);
        for (size_t i = 0; i < Varargs.size(); i++) {
          // todo: byval here?                                             
          auto r = Builder.CreateStructGEP(VarargsTy, alloced, i);
          Builder.CreateStore(Varargs[i], r); // alignment info could be better
      }

      auto asvoid = Builder.CreatePointerBitCastOrAddrSpaceCast(
          alloced, Type::getInt8PtrTy(Ctx));

      Args.push_back(asvoid);

      // Attributes excluding any on the vararg arguments
      AttributeList PAL = CB->getAttributes();
      if (!PAL.isEmpty()) {
        SmallVector<AttributeSet, 8> ArgAttrs;
        for (unsigned ArgNo = 0; ArgNo < NumArgs; ArgNo++)
          ArgAttrs.push_back(PAL.getParamAttrs(ArgNo));
        PAL = AttributeList::get(Ctx, PAL.getFnAttrs(),
                                 PAL.getRetAttrs(), ArgAttrs);
      }
      
      SmallVector<OperandBundleDef, 1> OpBundles;
      CB->getOperandBundlesAsDefs(OpBundles);

      // Make a new call instruction
      CallBase *NewCB = nullptr;
      if (InvokeInst *II = dyn_cast<InvokeInst>(CB)) {
        NewCB = InvokeInst::Create(NF, II->getNormalDest(), II->getUnwindDest(),
                                   Args, OpBundles, "", CB);
      } else {
        NewCB = CallInst::Create(NF, Args, OpBundles, "", CB);
        cast<CallInst>(NewCB)->setTailCallKind(
            cast<CallInst>(CB)->getTailCallKind());
      }
      NewCB->setAttributes(PAL);
      NewCB->takeName(CB);
      NewCB->setCallingConv(CB->getCallingConv());
      NewCB->copyMetadata(*CB, {LLVMContext::MD_prof, LLVMContext::MD_dbg});


      if (!CB->use_empty())
        CB->replaceAllUsesWith(NewCB);
      CB->eraseFromParent();


  }

  static Align assumedAlignment(const DataLayout &DL) {
    // TODO: Change the DataLayout API so there's an easier way to test
    // whether the stack alignment is known. Nvptx doesn't always have S
    // in the data layout string.
    Align ExcessiveAlignment = Align(UINT64_C(1) << 63u);
    bool knownNaturalStackAlignment =
        DL.exceedsNaturalStackAlignment(ExcessiveAlignment);
    if (knownNaturalStackAlignment) {
      return DL.getStackAlignment();
    } else {
      return {};
    }
  }

  static void runOnVarargFunction(Module &M, Function &F) {
    auto &Ctx = M.getContext();
    const DataLayout &DL = M.getDataLayout();

    // Can't use runOnFunction because ModuleToFunctionPassAdapater::run skips
    // over declarations.

    // Order of operations is:
    // Declare functions with the ... arg replaced with a void*,size_t pair
    // Replace call instructions to the variadics with calls to the new ones
    // Splice the body of original functions into the new ones
    // Lower intrinsics with respect to the additional arguments
    // Delete the remaining parts of the original functions
    //
    // derived from DAE mostly
    IRBuilder<> Builder(Ctx);


    // Get type of replacement function, remember how many fixed args it took
    FunctionType *FTy = F.getFunctionType();
    
    std::vector<Type *> ArgTypes (FTy->param_begin(), FTy->param_end());
    unsigned NumFixedArgs = FTy->getNumParams();
    ArgTypes.push_back(Type::getInt8PtrTy(Ctx));

    FunctionType *NFTy = FunctionType::get(FTy->getReturnType(),
                                          ArgTypes, /*IsVarArgs*/ false);

    Function *NF = Function::Create(NFTy, F.getLinkage(), F.getAddressSpace());

    // Note - same strategy as DeadArgumentElimination, so if there are problems
    // with attributes on the now-dead ... argument, need to fix there too
    NF->copyAttributesFrom(&F);
    NF->setComdat(F.getComdat());
    F.getParent()->getFunctionList().insert(F.getIterator(), NF); // don't reorder IR
    NF->takeName(&F);

    // struct doesn't alias anything else
    AttrBuilder ParamAttrs(Ctx);
    ParamAttrs.addAttribute(Attribute::NoAlias);
    // TODO: still need to arrange for the struct to have at least stack alignment
    // DL.getStackAlignment() <- this asserts, value not defined
    // Pointer pref alignment is 4 on gcn, seems unlikely to be the right value
    ParamAttrs.addAlignmentAttr(assumedAlignment(DL));

    //ParamAttres.addDereferenceableAttr(Size); // probably can't have this one

    AttributeList Attrs = NF->getAttributes();
    Attrs = Attrs.addParamAttributes(Ctx, NFTy->getNumParams()-1, ParamAttrs);
    NF->setAttributes(Attrs);
    

    // Declared the new function, can now create calls to it
    for (User *U : llvm::make_early_inc_range(F.users()))
      if (CallBase *CB = dyn_cast<CallBase>(U))
        ExpandCall(M, CB, NF, NumFixedArgs);


    // If it's a definition, move the implementation across
    if (!F.isDeclaration()) {
      NF->splice(NF->begin(), &F);

      auto NewArg = NF->arg_begin();
      for (Argument &Arg : F.args()) {
        Arg.replaceAllUsesWith(NewArg);
        NewArg->takeName(&Arg);
        ++NewArg;
      }
      NewArg->setName("varargs");

      for (BasicBlock &BB : *NF) 
        for (Instruction &I : llvm::make_early_inc_range(BB)) 
          if (VAStartInst *II = dyn_cast<VAStartInst>(&I))
            ExpandVAStart(II, NewArg);
    }


    
    SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
    F.getAllMetadata(MDs);
    for (auto [KindID, Node] : MDs)
      NF->addMetadata(KindID, *Node);

    

    // DAE bitcasts it, todo: check block addresses
    // This fails to update call instructions, unfortunately
    // It may therefore also fail to update globals
    F.replaceAllUsesWith(NF);

    F.eraseFromParent();
  }
};
} // namespace

char ExpandVAIntrinsics::ID = 0;

// char &llvm::ExpandVAIntrinsicsID = ExpandVAIntrinsics::ID;

INITIALIZE_PASS(ExpandVAIntrinsics, DEBUG_TYPE, "Expand VA intrinsics", false,
                false)

ModulePass *llvm::createExpandVAIntrinsicsPass() {
  return new ExpandVAIntrinsics();
}

PreservedAnalyses ExpandVAIntrinsicsPass::run(Module &M,
                                              ModuleAnalysisManager &) {
  return ExpandVAIntrinsics().runOnModule(M) ? PreservedAnalyses::none()
                                             : PreservedAnalyses::all();
}
