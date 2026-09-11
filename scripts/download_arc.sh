#!/usr/bin/env bash
# Fetches ARC-AGI-1, ARC-AGI-2 and ConceptARC into data/raw (shallow clones).
set -euo pipefail
mkdir -p data/raw
cd data/raw
[ -d ARC-AGI ] || git clone --depth 1 https://github.com/fchollet/ARC-AGI.git
[ -d ARC-AGI-2 ] || git clone --depth 1 https://github.com/arcprize/ARC-AGI-2.git
[ -d ConceptARC ] || git clone --depth 1 https://github.com/victorvikram/ConceptARC.git
ls ARC-AGI/data ARC-AGI-2/data ConceptARC/corpus | head
