#!/bin/bash

for seed in {120..139}; do
    echo "============================================================"
    echo "BASELINE  seed=$seed"
    echo "============================================================"
    MODEL=baseline SEED=$seed jupyter nbconvert         --execute --to notebook experiment.ipynb         --output /tmp/experiment_baseline_seed${seed}.ipynb

    echo "============================================================"
    echo "TPL       seed=$seed"
    echo "============================================================"
    MODEL=tpl SEED=$seed jupyter nbconvert         --execute --to notebook experiment.ipynb         --output /tmp/experiment_tpl_seed${seed}.ipynb
done
