//===-- ExpandVAIntrinsicsPass.cpp --------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Constants.h"
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

  bool runOnModule(Module &M) override {
    auto &Ctx = M.getContext();
    // Plan.
    //

    // Can't use runOnFunction because ModuleToFunctionPassAdapater::run skips
    // over declarations.

    bool Changed = false;

    // Find calls to varargs functions and hack with them, and then change the
    // varargs functions means the IR is valid in the intermediate phase, might
    // expose that for testing.

    // derived from DAE mostly
    // patch every call site to a variadic function

    fprintf(stderr, "Rewrite call instructions\n");
    for (Function &F : M.functions()) {
      if (!F.isVarArg())
        continue;

      fprintf(stderr, "Rewrite calls to %s\n", F.getName().str().c_str());

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
        // Need to make the struct and stash things in it, passing a null for
        // now

        std::vector<Value *> Args;
        Args.assign(CB->arg_begin(), CB->arg_begin() + NumArgs);

        Args.push_back(ConstantPointerNull::get(Type::getInt8PtrTy(Ctx)));
        Args.push_back(ConstantInt::get(Type::getInt64Ty(Ctx), 42));

        SmallVector<OperandBundleDef, 1> OpBundles;
        CB->getOperandBundlesAsDefs(OpBundles);

        // Make a new call instruction
        CallBase *NewCB = nullptr;
        if (InvokeInst *II = dyn_cast<InvokeInst>(CB)) {
          NewCB =
              InvokeInst::Create(&F, II->getNormalDest(), II->getUnwindDest(),
                                 Args, OpBundles, "", CB);
        } else {
          NewCB = CallInst::Create(&F, Args, OpBundles, "", CB);
          cast<CallInst>(NewCB)->setTailCallKind(
              cast<CallInst>(CB)->getTailCallKind());
        }
        NewCB->setCallingConv(CB->getCallingConv());
        NewCB->copyMetadata(*CB, {LLVMContext::MD_prof, LLVMContext::MD_dbg});

        // todo: attributes

        Args.clear();
        if (!CB->use_empty())
          CB->replaceAllUsesWith(NewCB);
        NewCB->takeName(CB);
        CB->eraseFromParent();

        fprintf(stderr, "Replacement %s\n", NewCB->getName().str().c_str());
        NewCB->dump();
      }
    }

    fprintf(stderr, "Rewrite functions\n");
    for (Function &F : llvm::make_early_inc_range(M)) {
      if (!F.isVarArg()) {
        continue;
      }

      fprintf(stderr, "Rewrite decl/defn of %s\n", F.getName().str().c_str());

      std::vector<Type *> ArgTypes;

      for (const Argument &I : F.args())
        ArgTypes.push_back(I.getType());

      ArgTypes.push_back(Type::getInt8PtrTy(Ctx));
      ArgTypes.push_back(Type::getInt64Ty(Ctx));

      FunctionType *FTy = FunctionType::get(
          F.getFunctionType()->getReturnType(), ArgTypes, /*IsVarArgs*/ false);

      // New function goes in the same place as the one being replaced
      Function *NF = Function::Create(FTy, F.getLinkage(), F.getAddressSpace());

      NF->copyAttributesFrom(&F);
      NF->setComdat(F.getComdat());
      F.getParent()->getFunctionList().insert(F.getIterator(), NF);
      NF->takeName(&F);

      // metadata too
      SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
      F.getAllMetadata(MDs);
      for (auto [KindID, Node] : MDs)
        NF->addMetadata(KindID, *Node);

      if (!F.isDeclaration()) {

        fprintf(stderr, "Extra things needed for definition %s\n",
                NF->getName().str().c_str());

        // Claim the blocks
        // Iterating over them in the new setting so that the
        // additional arguments can be referenced
        NF->splice(NF->begin(), &F);

        printf("walkies\n");
        for (BasicBlock &BB : *NF) {
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
      }

      // DAE bitcasts it, todo: check block addresses
      // This fails to update call instructions, unfortunately
      // It may therefore also fail to update globals
      F.replaceAllUsesWith(NF);

      F.eraseFromParent();
    }

    return Changed;
  }
};
} // namespace
char ExpandVAIntrinsics::ID = 0;

char &llvm::ExpandVAIntrinsicsID = ExpandVAIntrinsics::ID;

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
