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

class ExpandVAIntrinsics : public FunctionPass {
public:
  static char ID;

  ExpandVAIntrinsics() : FunctionPass(ID) {
    initializeExpandVAIntrinsicsPass(*PassRegistry::getPassRegistry());
  }

  Function *cloneWithoutVararg(Function *F) {
    auto &Ctx = F->getContext();
    // Intrinsic::ID ID = F->getIntrinsicID();
    ValueToValueMapTy VMap;
    ClonedCodeInfo CodeInfo;

    std::vector<Type *> ArgTypes;

    for (const Argument &I : F->args())
      ArgTypes.push_back(I.getType());

    ArgTypes.push_back(Type::getInt8PtrTy(Ctx));
    ArgTypes.push_back(Type::getInt64Ty(Ctx));

    FunctionType *FTy = FunctionType::get(F->getFunctionType()->getReturnType(),
                                          ArgTypes, false);

    // Create the new function...
    Function *NewF =
        Function::Create(FTy, F->getLinkage(), F->getAddressSpace(),
                         F->getName(), F->getParent());

    // Loop over the arguments, copying the names of the mapped arguments
    // over...
    Function::arg_iterator DestI = NewF->arg_begin();
    for (const Argument &I : F->args())
      DestI->setName(I.getName()); // Copy the name over...

    SmallVector<ReturnInst *, 8> Returns; // Ignore returns cloned.
    CloneFunctionInto(NewF, F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                      Returns, "", &CodeInfo);

    return NewF;
  }

  bool runOnFunction(Function &F) override {
    bool Changed = false;
    fprintf(stderr, "Pass exists\n");

    if (F.isVarArg()) {
      fprintf(stdout, "It's variadic\n");
      F.dump();

      fprintf(stdout, "arg size %zu\n", F.arg_size());
      F.getFunctionType()->dump();

      Type *retType = F.getFunctionType()->getReturnType();
      retType->dump();

      ArrayRef<Type *> params = F.getFunctionType()->params();

      // Takes dead argument elimination about 1000 lines to get this done which
      // seems excessive
      printf("nonvararg equivalent\n");
      FunctionType *nonvararg = FunctionType::get(retType, params, false);
      nonvararg->dump();

      for (auto &Arg : F.args()) {
        Arg.dump();
      }

      printf("clone without vararg\n");
      Function *n = cloneWithoutVararg(&F);
      n->dump();

      printf("walkies\n");
      for (BasicBlock &BB : *n) {
        for (Instruction &I : BB) {
          if (VAStartInst *II = dyn_cast<VAStartInst>(&I)) {
            printf("start\n");
            Value *args = II->getArgList();
            II->dump();
            args->dump();
          }
          if (VAEndInst *II = dyn_cast<VAEndInst>(&I)) {
            printf("end\n");
            Value *args = II->getArgList();
            II->dump();
            args->dump();
          }
          if (VACopyInst *II = dyn_cast<VACopyInst>(&I)) {
            printf("copy\n");
            Value *dst = II->getDest();
            Value *src = II->getSrc();
            II->dump();
            dst->dump();
            src->dump();
          }
        }
      }

      printf("uses\n");
      for (Use &U : n->uses()) {
        U->dump();
      }
      
      exit(42);
    } else {

      fprintf(stderr, "It's not variadic\n");
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

FunctionPass *llvm::createExpandVAIntrinsicsPass() {
  return new ExpandVAIntrinsics();
}

PreservedAnalyses ExpandVAIntrinsicsPass::run(Function &F,
                                              FunctionAnalysisManager &) {
  return ExpandVAIntrinsics().runOnFunction(F) ? PreservedAnalyses::none()
                                               : PreservedAnalyses::all();
}
