#!/bin/bash
# Fail on first error
set -e
echo "Running on host: $(hostname)"
echo "Current dir : $(pwd)"

# If you need a custom Python env, activate it here
# e.g. source my_env/bin/activate

python analyzer.py -i ../parquet_files/sim/bkg/merged/GGJets_MGG-80.parquet
