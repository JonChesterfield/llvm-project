//===- AlwaysSpecializer.cpp - implementation of always_specialize --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/AlwaysSpecializer.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/IPO/FunctionSpecialization.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "llvm/Support/CommandLine.h"


#include <cstdio> // todo
#include <tuple>

using namespace llvm;

#define DEBUG_TYPE "always-specialize"

namespace {

cl::opt<bool> EnableAlwaysSpecialize(
    "enable-always-specialize",
    cl::desc("Enable the always specialize pass"),
    cl::init(true), cl::Hidden);

  
class AlwaysSpecializer : public ModulePass {
public:
  static char ID;

  AlwaysSpecializer() : ModulePass(ID) {}

  StringRef getPassName() const override { return "Always specializer"; }

  bool runOnFunctionArgument(Module &M, Function &F, unsigned ArgNo);
  bool runOnFunction(Module &M, Function &F);
  bool runOnModule(Module &M) override;

  Function *cloneCandidateFunction(Function *F, unsigned ArgNo, Constant *C);

  using KeyType = std::tuple<Function *, unsigned, Constant *>;

  SmallVector<Function *, 8> worklist;
  DenseMap<KeyType, Function *> cache;
};

static Constant *getCandidateConstant(Value *V) {
  return dyn_cast<Constant>(V);
}

Function *AlwaysSpecializer::cloneCandidateFunction(Function *F, unsigned ArgNo,
                                                    Constant *C) {

  AlwaysSpecializer::KeyType key = {F, ArgNo, C};
  auto r = cache.find(key);
  if (r != cache.end()) {
    // fprintf(stderr, "Cache hit\n");
    return r->second;
  }

  // fprintf(stderr, "Cache miss\n");

  // Test cases will be simpler if the new function is inserted near
  // to the existing one but CloneFunction doesn't take an iterator
  // to specify where to insert it - todo, look for a more precise call
  ValueToValueMapTy Mappings;
  Function *Clone = CloneFunction(F, Mappings);
  Clone->setName(F->getName() + ".spec" + Twine(ArgNo));
  Clone->setLinkage(GlobalValue::InternalLinkage);

  // Gives termination when specialising multiple arguments
  Clone->removeParamAttr(ArgNo, llvm::Attribute::AlwaysSpecialize);

  Argument *V = Clone->getArg(ArgNo);

  V->replaceAllUsesWith(C); // this can turn indirect calls into direct

  cache.insert(std::make_pair(key, Clone));

  return Clone;
}

bool AlwaysSpecializer::runOnFunctionArgument(Module &M, Function &F,
                                              unsigned ArgNo) {
  bool Changed = false;

#if 1
  printf("Function %s has alwaysspec %u/%zu\n", F.getName().str().c_str(),
         ArgNo, F.arg_size());
  F.dump();
#endif

  for (User *U : make_early_inc_range(F.users())) {
    if (CallBase *CB = dyn_cast<CallBase>(U)) {
      if (CB->getCalledOperand() != &F)
        continue;

      Constant *C = getCandidateConstant(CB->getArgOperand(ArgNo));
      if (!C) {
        continue;
      }

      C->dump();

      if (Function *CF = dyn_cast<Function>(C)) {
        // Test case misses this because the constant
        // is actually a struct containing a function pointer
        // Actual issue is creating new uses of functions during
        // specialisation
        printf("Constant happens to be a function\n");
      }

      // bug here, can introduce new uses of F
      Changed = true;
      Function *clone = cloneCandidateFunction(&F, ArgNo, C);
      worklist.push_back(clone);

#if 0
      printf("Call instruction has constant at %u, clone %p\n", ArgNo,
             (void *)clone);
#endif

      CB->setCalledFunction(clone);
    }
  }

  return Changed;
}

bool AlwaysSpecializer::runOnFunction(Module &M, Function &F) {
  bool Changed = false;
  size_t arity = F.arg_size();
  
  for (size_t i = 0; i < arity; i++) {
    // On all attributes, not just the first alwaysspecialize
    if (F.hasParamAttribute(i, llvm::Attribute::AlwaysSpecialize)) {
      Changed |= runOnFunctionArgument(M, F, i);
    }
  }
  return Changed;
}

bool AlwaysSpecializer::runOnModule(Module &M) {

  if (!EnableAlwaysSpecialize)
    {
      fprintf(stderr, "Disabled Always Specializer\n");
      return false;
    }
  
  fprintf(stderr, "Running Always Specializer\n");
  bool Changed = false;


  // Go from a function to the specialisations of that function,
  // with a counter to allow testing whether more uses have been
  // added since the current set of specialisations were chosen

  #if 0

  // Idea here is to run to fixpoint without ever leaving multiple equivalent clones
  // E.g. want foo(x, 4) and foo(3, y) to and up resolved to the same function if x=3 and y=4
  // regardless of the path taken to notice the call sites have the same arguments
  // Thus two passes - the first identifies call sites with constant arguments, including
  // those uncovered by previous specialisations - without changing any existing call sites.
  //
  // That is, the new specialisations might contain calls to the original function set but
  // no function, original or cloned, contains any calls into the specialisations.
  // That ensures that later lookups hit in the same cache since they're keyed from the original
  // functions, thus no extra copies of equivalent functions are created.
  //
  // The mechanism for resolving the fixpoint is to actually instantiate the specialisations.
  // In some cases, this will lead to creating functions that ultimately prove to be unused,
  // in which case they're deleted.
  //
  //
  // If the developer requests exponential growth of code though this annotation, that's what
  // this pass will produce. The worst case is where every call to a function is made with a
  // distinct constant argument, in which case every call site will get it's own specialisation
  // which is only used there. The inliner will then recognise that as free to inline and the
  // overall behaviour, including implied code growth, is that of always_inline.
  //
  // In other cases, this is a way to get the benefit of specialisation with less code size
  // than always_inline implies.
  
  DenseMap<Function&,
    struct {
      size_t prevCount;
      DenseMap<SmallVector<Constant*, 4>, Function*> specs;
    }> Map;

  // If the function has the attribute, start it from a count of 0
  for (Function &F : make_early_inc_range(M))
    {
      if (eligible(F))
        {
          Map[F] = {0 /*prevCount*/, {} /* table */};
        }
    }
  if (Map.size() == 0) return false;


  while (true)
    {
    size_t added = 0;
    for (Function &F : Map)
      {
        {
          // If F has no uses, don't need to specialise it
          // If it has more uses than last time it was considered, want to check them
          uint64_t state = F.uses();
          if (state <= Map[F].prevCount)
            {
              break;
            }
          Map[F].prevCount = state;
        }

        // Has at least one use we haven't looked at before
        for (Use &u: F.uses())
          {
            if (!isCallWithConstants(u))
              {
                // Only care about calls we can specialise
                continue;
              }

            
            Args = getConstantArgs(u);
            Spec & s = Map[F].specs[Args];

            // If we've already specialised wrt this argument set, reuse that one
            if (s.target) continue;

            // Create a new static function. Doesn't have any uses yet.
            // Notably don't want to add it to the map we're iterating over.
            s.target = maybeCloneWithRespectToArgs(Args);

            // clone returns null if the function wasn't worth specialising,
            // in particular because the constant arguments in question were unused
            // that's important for making this pass a no-op if it's run multiple
            // times without dead argument elimination
            if (s.target) added += 1;
          }
      }

    if (added == 0) break;
  }

  // We now have a map of all functions with this attribute to specialisations,
  // indexed by the call site constant arguments. None of the new specialisations
  // are used yet.


    for (Function &F : Map)
      {
        for (Use &u: F.uses())
          {
            if (!isCallWithConstants(u))
              {
                continue;
              }

            Args = getConstantArgs(u);
            Spec & s = Map[F].specs[Args];
            assert(s.target);

            rewiteCallToTarget(s.target);
          }
      }

    // Some of the specialisations that were created when there was a call site
    // later turned out to be unused. In particular, where the call site takes
    // two arguments, one already constant, and the second also turns out to be constant
    // partway through specialisation, the call to constant, unknown may be dead
    removeDeadSpecialisations();
  
#endif
  
  // One problem here is that specialising on an argument, followed by RAUW,
  // can convert an indirect call into a direct call, where the latter has
  // explict specialise attributes, and thus should be fed back into the pass

  // Maybe check the constant against the known functions, or just push it onto
  // the worklist

  // On each existing function. Don't need to push the existing ones into the
  // worklist
  for (Function &F : make_early_inc_range(M))
    {
      Changed |= runOnFunction(M, F);
    }

  if (!Changed)
    return false;

  // Then on any functions created by this pass (for > 1 specialized argument)
  while (!worklist.empty()) {
    Function *F = worklist.pop_back_val();
    Changed |= runOnFunction(M, *F);
  }

  // Created functions that were specialised further may now be unused
  for (auto &[_, f] : cache)
    if (f->use_empty())
      f->eraseFromParent();

  return true;
}

} // namespace

char AlwaysSpecializer::ID = 0;

INITIALIZE_PASS(AlwaysSpecializer, DEBUG_TYPE, "TODO", false, false)

ModulePass *llvm::createAlwaysSpecializerPass() {
  return new AlwaysSpecializer();
}

PreservedAnalyses AlwaysSpecializerPass::run(Module &M,
                                             ModuleAnalysisManager &) {
  return AlwaysSpecializer().runOnModule(M) ? PreservedAnalyses::none()
                                            : PreservedAnalyses::all();
}

AlwaysSpecializerPass::AlwaysSpecializerPass() {}
