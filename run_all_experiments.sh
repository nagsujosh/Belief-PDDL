#!/bin/bash
# Belief-PDDL: The 5-Table Experiment Harness Orchestrator
# Generates the JSONL/CSV outputs required to prove Causal AI Planning claims.

mkdir -p outputs/benchmarks

echo "=========================================="
echo "Table 1: Belief vs Threshold"
echo "=========================================="
# Compare rigid mapping versus continuous marginals with identical 30% noise.
venv/bin/python scripts/run_benchmarks.py --mode threshold --noise 0.3 --episodes 10
venv/bin/python scripts/run_benchmarks.py --mode belief_no_verifier --noise 0.3 --episodes 10
venv/bin/python scripts/run_benchmarks.py --mode belief_plus_verifier --noise 0.3 --episodes 10
venv/bin/python scripts/run_benchmarks.py --mode full_top_k --noise 0.3 --k 3 --episodes 10

echo "=========================================="
echo "Figure 2: The Advantage Curve (Increasing Uncertainty/Noise)"
echo "=========================================="
# Scale VLM Hallucination from 10% to 90%. Watch the thresholding method crash immediately.
for NOISE in 0.1 0.4 0.7 0.9; do
    echo "Running with VLM Hallucination Noise = $NOISE"
    venv/bin/python scripts/run_benchmarks.py --mode threshold --noise $NOISE --episodes 10
    venv/bin/python scripts/run_benchmarks.py --mode full_top_k --noise $NOISE --k 3 --episodes 10
done

echo "=========================================="
echo "Table 3: Ablating Sensing (Information Gain)"
echo "=========================================="
venv/bin/python scripts/run_benchmarks.py --mode full_top_k_no_sense --noise 0.5 --k 3 --episodes 10
venv/bin/python scripts/run_benchmarks.py --mode full_top_k --noise 0.5 --k 3 --episodes 10

echo "Sweeps Complete. Batch traces saved to ./outputs/benchmarks/"
