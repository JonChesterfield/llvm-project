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
    // Function pass doesn't get called on declarations
    // F : M doesn't seem to either
    bool Changed = false;

    M.dump();
    
    fprintf(stderr, "Got global:\n");
    for (auto &GV : M.globals()) {
    fprintf(stderr, "Got a global %s\n",  GV.getName().str().c_str());
    }

    fprintf(stderr, "Got const func:\n");
    for (const Function &GV : M) {
      fprintf(stderr, "Got a const function %s\n",  GV.getName().str().c_str());
    }

    fprintf(stderr, "Got mutable function:\n");
    for (Function &GV : M) {
      fprintf(stderr, "Got mutable function %s\n",  GV.getName().str().c_str());
    }

    
    fprintf(stderr, "Got ifuncs:\n");
    for (auto &X : M.ifuncs()) {
    fprintf(stderr, "Got a ifunc %s\n",  X.getName().str().c_str());
    }


    fprintf(stderr, "Got functions:\n");
    for (Function &F : M.functions()) {
    
    fprintf(stderr, "Run on function %s\n",  F.getName().str().c_str());

    if (F.isDeclaration()) {
      fprintf(stderr, "Run on declaration\n");
    } else {
      fprintf(stderr, "Run on definition\n");
    }
    
    
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
            Value *args = II->getArgList();
            II->dump();
            args->dump();
            continue;
          }
          
          if (VAEndInst *II = dyn_cast<VAEndInst>(&I)) {
            printf("end\n");
            Value *args = II->getArgList();
            II->dump();
            args->dump();
            continue;
          }
          
          if (VACopyInst *II = dyn_cast<VACopyInst>(&I)) {
            printf("copy\n");
            Value *dst = II->getDest();
            Value *src = II->getSrc();
            II->dump();
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
