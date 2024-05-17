#!/bin/bash

# These are local-machine specific
SOURCE=~/llvm-project
BUILD=~/llvm-build/llvm
OUTPUT=$SOURCE/docs

# Assumes the binaries under build were compiled with $(which clang)
python3 $SOURCE/llvm/utils/prepare-code-coverage-artifact.py \
        --preserve-profiles \
        --unified-report \
        -C $BUILD \
        $(which llvm-profdata) \
        $(which llvm-cov) \
        $BUILD/profiles/ \
        $OUTPUT \
        $BUILD/bin/clang $BUILD/bin/opt $BUILD/bin/llc 


exit 0

# Github won't accept any files greater than 100MB in size
# TODO, replace them with stubs saying the file was too big?
find $OUTPUT -type f -size +100M -delete
