#!/bin/bash

# These are local-machine specific
SOURCE=~/llvm-project
BUILD=~/llvm-build/llvm
OUTPUT=$SOURCE/profile

rm -rf -- $OUTPUT/opt $OUTPUT/clang $OUTPUT/llc


INVOKE="python3 $SOURCE/llvm/utils/prepare-code-coverage-artifact.py --preserve-profiles --unified-report -C $BUILD $(which llvm-profdata) $(which llvm-cov) $BUILD/profiles/ $OUTPUT"

# This should probably merge them once and then spawn separate processes for each binary. Having three copies do the
# merge at once seems to win segfaults.
$INVOKE $BUILD/bin/clang $BUILD/bin/opt $BUILD/bin/llc 

