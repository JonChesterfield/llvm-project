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
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Support/CommandLine.h"

#include <cstdio> // todo
#include <tuple>

using namespace llvm;

#define DEBUG_TYPE "always-specialize"

namespace {

cl::opt<bool>
    EnableAlwaysSpecialize("enable-always-specialize",
                           cl::desc("Enable the always specialize pass"),
                           cl::init(true), cl::Hidden);

// Scalar/LoopStrengthReduce.cpp and Vectorize/VPlan.h both have one of these
// Try to put a templated version under DenseMapInfo.h and remove them

template <typename T,
          unsigned N = CalculateSmallVectorDefaultInlinedElements<T>::value>
struct SmallVectorDenseMapInfo {
  static SmallVector<T *, N> getEmptyKey() {
    SmallVector<T *, N> V;
    V.push_back(reinterpret_cast<T *>(-1));
    return V;
  }

  static SmallVector<T *, N> getTombstoneKey() {
    SmallVector<T *, N> V;
    V.push_back(reinterpret_cast<T *>(-2));
    return V;
  }

  static unsigned getHashValue(const SmallVector<T *, N> &V) {
    return static_cast<unsigned>(hash_combine_range(V.begin(), V.end()));
  }

  static bool isEqual(const SmallVector<T *, N> &LHS,
                      const SmallVector<T *, N> &RHS) {
    return LHS == RHS;
  }
};

class AlwaysSpecializer : public ModulePass {
public:
  static char ID;

  AlwaysSpecializer() : ModulePass(ID) {}

  StringRef getPassName() const override { return "Always specializer"; }

  bool runOnFunctionArgument(Module &M, Function &F, unsigned ArgNo);
  bool runOnFunction(Module &M, Function &F);
  bool runOnModule(Module &M) override;

  static Constant *getCandidateConstant(Value *V) {
    return dyn_cast<Constant>(V);
  }

  static bool functionEligible(const Function &F) {
    if (F.isDeclaration()) // redundant with exact defn?
      return false;

    if (F.use_empty())
      return false;

    
    // E.g. leave linkonce_odr alone
    if (!F.hasExactDefinition())
      return false;

    // Can't sensibly clone naked functions (todo, does clone properly error on this?)
    if (F.hasFnAttribute(Attribute::Naked))
      return false;

    
    // How about F.hasFnAttribute(Attribute::Naked)
    if (F.isIntrinsic())
      return false; // redundant with above?

    size_t arity = F.arg_size();

    for (size_t i = 0; i < arity; i++) {
      if (F.hasParamAttribute(i, llvm::Attribute::AlwaysSpecialize)) {
        return true;
      }
    }

    return false;
  }

  template <unsigned N>
  static bool callEligible(const Function &F, const CallBase *CB,
                           SmallVector<Constant *, N> &out) {
    size_t arity = F.arg_size();
    if (CB->getCalledOperand() != &F)
      return false;

    if (CB->getFunctionType() != F.getFunctionType()) return false;

    // redundant with the above?
    if (CB->arg_size() != arity)
      return false; // todo

    bool eligible = false;

    out.clear();
    for (size_t i = 0; i < arity; i++) {
      Constant *Arg = getCandidateConstant(CB->getArgOperand(i));
      if (Arg && F.hasParamAttribute(i, llvm::Attribute::AlwaysSpecialize)) {
        // TODO: Need to check that the arg is used by the function as well
        eligible = true;
        out.push_back(Arg);
      } else {
        out.push_back(nullptr);
      }
    }

    return eligible;
  }

  Function *cloneCandidateFunction(Function *F, unsigned ArgNo, Constant *C);
  Function *cloneCandidateFunction(Function *F, SmallVector<Constant *, 4> C);

  using KeyType = std::tuple<Function *, unsigned, Constant *>;

  SmallVector<Function *, 8> worklist;
  DenseMap<KeyType, Function *> cache;

  // Looking to run without a worklist
  struct FunctionSpecializations {
    size_t prevCount = 0;
    DenseMap<SmallVector<Constant *, 4>, Function *,
             SmallVectorDenseMapInfo<Constant, 4>>
        specs;
  };

  // Probably needs info about smallvector, might need to be on a pointer not a
  // ref
  DenseMap<Function *, FunctionSpecializations> SpecMap;
};

static bool operator==(const AlwaysSpecializer::FunctionSpecializations &lhs,
                       const AlwaysSpecializer::FunctionSpecializations &rhs) {
  return lhs.prevCount == rhs.prevCount && lhs.specs == rhs.specs;
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
  Clone->copyAttributesFrom(F);
  
  Clone->setComdat(F->getComdat()); // if comdat, should we be refusing to clone it?

  Clone->setName(F->getName() + ".spec" + Twine(ArgNo));
  Clone->setLinkage(GlobalValue::InternalLinkage);

  // hack on F->getParent()->getFunctionList()? 
  
  // Gives termination when specialising multiple arguments
  Clone->removeParamAttr(ArgNo, llvm::Attribute::AlwaysSpecialize);

  Argument *V = Clone->getArg(ArgNo);

  V->replaceAllUsesWith(C); // this can turn indirect calls into direct

  cache.insert(std::make_pair(key, Clone));

  return Clone;
}

__attribute__((used)) Function *
AlwaysSpecializer::cloneCandidateFunction(Function *F,
                                          SmallVector<Constant *, 4> C) {

  for (size_t i = 0; i < C.size(); i++) {
    if (C[i] != 0)
      goto good;
  }

  return 0;

good:;

  // Find the existing specialisations of F
  auto r = SpecMap.find(F);
  assert(r != SpecMap.end());

  FunctionSpecializations &existing = r->second;

  // See if we've already created one for this argument vector
  auto r2 = existing.specs.find(C);
  if (r2 != existing.specs.end()) {
    fprintf(stderr, "Already spawned\n");
    return r2->second;
  }

  // create a name based on the vector
  ValueToValueMapTy Mappings;
  Function *Clone = CloneFunction(F, Mappings);
  Clone->copyAttributesFrom(F);
  Clone->setComdat(F->getComdat()); // if comdat, should we be refusing to clone it?
  
  Clone->setName(F->getName() + ".spec");
  Clone->setLinkage(GlobalValue::InternalLinkage);

  // hack on F->getParent()->getFunctionList()?
  
  // Replace uses of the argument with the constant
  // Strip the specialize from the parameters that are being replaced
  for (size_t i = 0; i < C.size(); i++) {
    Constant *c = C[i];
    if (c == 0) continue;
         
    Clone->removeParamAttr(i, llvm::Attribute::AlwaysSpecialize);

    Argument *V = Clone->getArg(i);
    if (V->use_empty()) {
      fprintf(stderr, "Um, argument has no uses - do we want to be replacing it?\n");
    }

    printf("What are the uses?\n");
    for (Use &u : make_early_inc_range(V->uses()))
      {
        // Hopefully some uses of the argument are instructions
        printf("Use\n");
        u.get()->dump();

        User * user = u.getUser();
        if (Instruction * I = dyn_cast<Instruction>(u.getUser()))
          {
            for (Use& op : I->operands())
              {
                if (op == u)
                  {
                    printf("Use is an operand of an instruction\n");

                    SimplifyQuery SQ = SimplifyQuery(Clone->getDataLayout(),
                                                       I);
                    // This bails if it isn't an instruction
                    Value * maybe = simplifyWithOpReplaced(I,
                                                           u.get(),
                                                           c, SQ,
                                                           false /*AllowRefinement*/);
                    if (maybe && false)
                      {
                        printf("better, replacing uses of\n");
                        I->dump();
                        printf("With \n");
                        maybe->dump();
                        I->replaceAllUsesWith(maybe);
                        
                        // If it's an uninteresting instruction, erase it as well
                        #if 0
                        if (!I->isEHPad() && !I->isTerminator() && !I->mayHaveSideEffects())
                          I->eraseFromParent();
                        #endif

                        printf("Continue\n");
                        break;
                      }
                      else
                        {
                          printf("No simplify\n");
                        }
                      
                  }
                }
            }

          
          else
            {
              printf("not an instruction\n");
              u.getUser()->dump();
            }
        }


     V->replaceAllUsesWith(c); // this can turn indirect calls into direct
  } // for each argument

  printf("Done\n");
  
  existing.specs.insert(std::make_pair(C, Clone));

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
        (void)CF;
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

  if (!EnableAlwaysSpecialize) {
    fprintf(stderr, "Disabled Always Specializer\n");
    return false;
  }

  fprintf(stderr, "Running Always Specializer\n");
  bool Changed = false;

  bool newpath = true;

  // Go from a function to the specialisations of that function,
  // with a counter to allow testing whether more uses have been
  // added since the current set of specialisations were chosen

  if (newpath) {

    // Idea here is to run to fixpoint without ever leaving multiple equivalent
    // clones E.g. want foo(x, 4) and foo(3, y) to and up resolved to the same
    // function if x=3 and y=4 regardless of the path taken to notice the call
    // sites have the same arguments Thus two passes - the first identifies call
    // sites with constant arguments, including those uncovered by previous
    // specialisations - without changing any existing call sites.
    //
    // That is, the new specialisations might contain calls to the original
    // function set but no function, original or cloned, contains any calls into
    // the specialisations. That ensures that later lookups hit in the same
    // cache since they're keyed from the original functions, thus no extra
    // copies of equivalent functions are created.
    //
    // The mechanism for resolving the fixpoint is to actually instantiate the
    // specialisations. In some cases, this will lead to creating functions that
    // ultimately prove to be unused, in which case they're deleted.
    //
    //
    // If the developer requests exponential growth of code though this
    // annotation, that's what this pass will produce. The worst case is where
    // every call to a function is made with a distinct constant argument, in
    // which case every call site will get it's own specialisation which is only
    // used there. The inliner will then recognise that as free to inline and
    // the overall behaviour, including implied code growth, is that of
    // always_inline.
    //
    // In other cases, this is a way to get the benefit of specialisation with
    // less code size than always_inline implies.

    // If the function has the attribute, start it from a count of 0
    for (Function &F : make_early_inc_range(M)) {
      if (functionEligible(F)) {
        // fprintf(stderr, "Function eligible %s\n", F.getName().str().c_str());
        SpecMap[&F] = FunctionSpecializations();
      }
      else {
        // fprintf(stderr, "Function rejected %s\n", F.getName().str().c_str());        
      }
    }
    if (SpecMap.size() == 0)
      return false;

    SmallVector<Constant *, 4> ArgVec;

    while (true) {
      size_t added = 0;
      for (auto &[F, spec] : SpecMap) {
        assert(spec == SpecMap[F]);

        {
          // If F has no users, don't need to specialise it
          // If it has more users than last time it was considered, want to recheck
          // them
          uint64_t state =
            (unsigned)std::distance(F->user_begin(), F->user_end());
          //F->getNumUses(); // this is linear in the uses of the function :(
          fprintf(stderr, "Function %s, prevUses %lu, current uses %lu\n",
                  F->getName().str().c_str(),
                  SpecMap[F].prevCount,
                  state);
          if (state <= SpecMap[F].prevCount) {
            break;
          }
          SpecMap[F].prevCount = state;
        }

        fprintf(stderr, "Look at the uses of %s\n", F->getName().str().c_str());
        // Has at least one use we haven't looked at before

        for (User *u : make_early_inc_range(F->users())) {
                    
          CallBase *CB = dyn_cast<CallBase>(u);

          
          if (!CB) {
            fprintf(stderr, "Not a callbase\n");
            continue;
          }
          
          if (!CB || !callEligible(*F, CB, ArgVec)) {
            continue;
          }

          auto maybe = SpecMap[F].specs.find(ArgVec);
          if (maybe != SpecMap[F].specs.end()) {
            // If we've already specialised wrt this argument set, reuse that
            // one
            continue;
          }

          fprintf(stderr, "Clone time\n");

          // Create a new static function. Doesn't have any uses yet.
          // Notably don't want to add it to the map we're iterating over.
          Function *clone = cloneCandidateFunction(F, ArgVec);
          // clone should return null if the function wasn't worth specialising,
          // or at least something should catch that.
          // in particular because the constant arguments in question were
          // unused that's important for making this pass a no-op if it's run
          // multiple times without dead argument elimination
          if (clone) {
            SpecMap[F].specs.insert(std::make_pair(ArgVec, clone));
            added += 1;
          }
        }
      }

      if (added == 0)
        break;
    }

    // We now have a map of all functions with this attribute to
    // specialisations, indexed by the call site constant arguments. None of the
    // new specialisations are used yet.

    for (auto &[F, spec] : SpecMap) {
      for (User *u : make_early_inc_range(F->users())) {
        CallBase *CB = dyn_cast<CallBase>(u);
        if (!CB || !callEligible(*F, CB, ArgVec)) {
          continue;
        }

        // this is probably specmap[f]
        assert(spec == SpecMap[F]);
        Function *target = SpecMap[F].specs[ArgVec];
        if (target)
          CB->setCalledFunction(target);
      }

      (void)spec;
    }

    // Some of the specialisations that were created when there was a call site
    // later turned out to be unused. In particular, where the call site takes
    // two arguments, one already constant, and the second also turns out to be
    // constant partway through specialisation, the call to constant, unknown
    // may be dead

#if 0
    removeDeadSpecialisations();
#endif

    return true;

  } else { // old path

    // One problem here is that specialising on an argument, followed by RAUW,
    // can convert an indirect call into a direct call, where the latter has
    // explict specialise attributes, and thus should be fed back into the pass

    // Maybe check the constant against the known functions, or just push it
    // onto the worklist

    // On each existing function. Don't need to push the existing ones into the
    // worklist
    for (Function &F : make_early_inc_range(M)) {
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
