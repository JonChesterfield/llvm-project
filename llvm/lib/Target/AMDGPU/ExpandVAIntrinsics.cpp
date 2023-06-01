//===-- ExpandVAIntrinsicsPass.cpp --------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/IR/Constants.h"

#include "AMDGPU.h" // wherever initializeExpandVAIntrinsicsPass is

#include <cstdio>

#define DEBUG_TYPE "expand-va-intrinsics"

using namespace llvm;

namespace {

class ExpandVAIntrinsics : public ModulePass {
public:
  static char ID;

  ExpandVAIntrinsics() : ModulePass(ID) {
    initializeExpandVAIntrinsicsPass(*PassRegistry::getPassRegistry());
  }

  Function *cloneWithoutVararg(Function *F) {
    // see Attributor::internalizeFunctions
    
    auto &Ctx = F->getContext();
    // Intrinsic::ID ID = F->getIntrinsicID();
    ClonedCodeInfo CodeInfo;

    std::vector<Type *> ArgTypes;

    for (const Argument &I : F->args())
      ArgTypes.push_back(I.getType());

    // Append a void*, size_t pair (todo, drop the 64 assumption)
    ArgTypes.push_back(Type::getInt8PtrTy(Ctx));
    ArgTypes.push_back(Type::getInt64Ty(Ctx));

    FunctionType *FTy = FunctionType::get(F->getFunctionType()->getReturnType(),
                                          ArgTypes, /*IsVarArgs*/ false);

    // Create the new function...
    Function *NewF =
        Function::Create(FTy, F->getLinkage(), F->getAddressSpace(),
                         F->getName(), F->getParent());
    
    // Loop over the arguments, copying the names of the mapped arguments
    // over...
    Function::arg_iterator DestI = NewF->arg_begin();
    for (const Argument &I : F->args())
      DestI->setName(I.getName()); // Copy the name over...

    ValueToValueMapTy VMap;
    SmallVector<ReturnInst *, 8> Returns; // Ignore returns cloned.
    CloneFunctionInto(NewF, F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                      Returns, "", &CodeInfo);

    return NewF;
  }

  bool runOnModule(Module &M) override {
    auto &Ctx = M.getContext();
    // Plan.
    // 
    
    
    // Can't use runOnFunction because ModuleToFunctionPassAdapater::run skips
    // over declarations.
    
    bool Changed = false;


    // Find calls to varargs functions and hack with them, and then change the varargs functions
    // means the IR is valid in the intermediate phase, might expose that for testing.

    // derived from DAE mostly
    // patch every call site to a variadic function
    for (Function &F : M.functions()) {
      if (!F.isVarArg())
        continue;
      FunctionType *FTy = F.getFunctionType();
      std::vector<Type *> Params(FTy->param_begin(), FTy->param_end());
      unsigned NumArgs = Params.size();

      // Append a void*, size_t pair (todo, drop the 64 assumption)
      Params.push_back(Type::getInt8PtrTy(Ctx));
      Params.push_back(Type::getInt64Ty(Ctx));

      for (User *U : llvm::make_early_inc_range(F.users())) {
        CallBase *CB = dyn_cast<CallBase>(U);
        if (!CB)
          continue;

        
        fprintf(stderr, "Call inst %s\n", CB->getName().str().c_str());
        CB->dump();
        
        // TODO: Deal with attributes on the varargs part, see DAE
        std::vector<Value *> Args;
        Args.assign(CB->arg_begin(), CB->arg_begin() + NumArgs);        

        // Need to make the struct and stash things in it, passing a null for now

        Args.push_back(ConstantPointerNull::get(Type::getInt8PtrTy(Ctx)));
        Args.push_back(ConstantInt::get(Type::getInt64Ty(Ctx), 42));
          

        SmallVector<OperandBundleDef, 1> OpBundles;
        CB->getOperandBundlesAsDefs(OpBundles);

        // Make a new call instruction
        CallBase *NewCB = nullptr;
        if (InvokeInst *II = dyn_cast<InvokeInst>(CB)) {
          NewCB = InvokeInst::Create(&F, II->getNormalDest(), II->getUnwindDest(),
                                     Args, OpBundles, "", CB);
        } else {
          NewCB = CallInst::Create(&F, Args, OpBundles, "", CB);
          cast<CallInst>(NewCB)->setTailCallKind(
                                                 cast<CallInst>(CB)->getTailCallKind());
        }
        NewCB->setCallingConv(CB->getCallingConv());
        NewCB->copyMetadata(*CB, {LLVMContext::MD_prof, LLVMContext::MD_dbg});

        // attributes

        Args.clear();
        if (!CB->use_empty())
          CB->replaceAllUsesWith(NewCB);
        NewCB->takeName(CB);
        CB->eraseFromParent();
        
        fprintf(stderr, "Replacement %s\n", NewCB->getName().str().c_str());
        NewCB->dump();

    }
    }
    

    for (Function &F : M.functions()) {
      if (!F.isVarArg()) {
        continue;
      }

      if (F.isDeclaration()) {
        fprintf(stderr, "Run on declaration %s\n",  F.getName().str().c_str());

            std::vector<Type *> ArgTypes;

            for (const Argument &I : F.args())
              ArgTypes.push_back(I.getType());

            ArgTypes.push_back(Type::getInt8PtrTy(Ctx));
            ArgTypes.push_back(Type::getInt64Ty(Ctx));

            FunctionType *FTy = FunctionType::get(F.getFunctionType()->getReturnType(),
                                                  ArgTypes, /*IsVarArgs*/ false);



            Function *NewF =
              Function::Create(FTy, F.getLinkage(), F.getAddressSpace(),
                               F.getName(), F.getParent());


            fprintf(stderr, "prev\n");
            F.dump();
            fprintf(stderr, "repl\n");
            NewF->dump();

            // Need to copy more stuff across and replaceall.
            F.replaceAllUsesWith(NewF);
            NewF->takeName(&F);
            
            fprintf(stderr, "prev.2\n");
            F.dump();
            fprintf(stderr, "repl.2\n");
            NewF->dump();

            
      }
      
      if (!F.isDeclaration()) {
        fprintf(stderr, "Run on definition %s\n",  F.getName().str().c_str());
      }
    

      // The vararg intrinsics are found in vararg instructions, skip the
      // walk over instructions for others.
      
      if (!F.isVarArg()) {
        // vararg intrinsics are only meaningful in vararg functions
        return false;
      }

    fprintf(stdout, "It's variadic\n");
    // F.dump();


      for (auto &Arg : F.args()) {
        Arg.dump();
      }


      printf("walkies\n");
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          
          if (VAStartInst *II = dyn_cast<VAStartInst>(&I)) {
            printf("start\n");
            II->dump();
            Value *args = II->getArgList();
            args->dump();
            continue;
          }
          
          if (VAEndInst *II = dyn_cast<VAEndInst>(&I)) {
            printf("end\n");
            II->dump();
            Value *args = II->getArgList();
            args->dump();
            continue;
          }
          
          if (VACopyInst *II = dyn_cast<VACopyInst>(&I)) {
            printf("copy\n");
            II->dump();
            Value *dst = II->getDest();
            Value *src = II->getSrc();
            dst->dump();
            src->dump();
            continue;
          }

          
        }
      }

      exit (1);
#if 0
      // Takes dead argument elimination about 1000 lines to get this done which
      // seems excessive
      printf("nonvararg equivalent\n");
      Type *retType = F.getFunctionType()->getReturnType();
      ArrayRef<Type *> params = F.getFunctionType()->params();
      FunctionType *nonvararg = FunctionType::get(retType, params, false);
      nonvararg->dump();


      printf("clone without vararg\n");
      Function *n = cloneWithoutVararg(&F);
      n->dump();

      printf("uses\n");
      for (Use &U : n->uses()) {
        U->dump();
      }
#endif     
    
  }
    return Changed;
        
  }
};
} // namespace
char ExpandVAIntrinsics::ID = 0;

char &llvm::ExpandVAIntrinsicsID = ExpandVAIntrinsics::ID;

INITIALIZE_PASS(ExpandVAIntrinsics, DEBUG_TYPE,
                "Expand VA intrinsics", false,
                false)

ModulePass *llvm::createExpandVAIntrinsicsPass() {
  return new ExpandVAIntrinsics();
}

PreservedAnalyses ExpandVAIntrinsicsPass::run(Module &M,
                                              ModuleAnalysisManager &) {
  return ExpandVAIntrinsics().runOnModule(M) ? PreservedAnalyses::none()
                                               : PreservedAnalyses::all();
}
