# Constrained Predicate Belief Projection for Neuro-Symbolic Planning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Constrained Predicate Belief Projection** is a formal neuro-symbolic planning method that addresses a critical failure mode in robotic and general logic systems: independently estimated symbolic predicates (via visual classification or noisy percepts) are often inconsistent and physically impossible, rendering classical deterministic planning brittle. 

We propose and implement a general robust architecture: **constrained symbolic belief inference combined with world-aware Top-K planning**.

By factorizing belief distributions locally, projecting the resultant probabilities through a generalized mathematical constraint compiler explicitly solving for the Maximum A Posteriori (MAP) and Top-K feasible universes, and executing information-gain-aware deterministic planning trees, agents execute reliable long-horizon logic under massive partial observability.

## 🛠 Core Contributions

1. **Factorized Symbolic Belief**: We explicitly maintain continuous Bayes-updated marginals across specific grounded relational predicates, fundamentally severing the reliance on single-shot thresholded visual states tracking.
2. **Constraint-Based Projection**: This is a direct mathematical inference layer. The framework compiles purely generalized `YAML` constraint domains into Operations Research constraints (Google OR-Tools CP-SAT) minimizing local inconsistency via:
   $$ \hat{x} = \arg\max_{x \in \mathcal{C}} \sum_i \log p_i^{x_i}(1-p_i)^{1-x_i} $$
3. **Top-K World Planning with Sensing**: Generating the Top-$K$ constraint-feasible worlds, planning deterministically across each universe simultaneously, computing expected action weights securely, and deploying active approximate Information Gain (IG) sensorial fallback loops if topological entropy peaks.

## 🧠 Experimental Benchmarks

This framework leverages two distinct Instantiations testing algorithmic consistency against extreme observation corruption:
- **Blocksworld**: A structurally deterministic foundational logic domain allowing rigorous empirical parameter sweeps measuring *Inconsistency Rates* and active *Information Gain* thresholding.
- **ALFWorld (Instance)**: A 3D Embodied GUI Instantiation (AI2-THOR) dynamically mapping realistic textual instructions bounding pre-trained OpenAI CLIP Vision Model arrays explicitly against the verifier to solve real physical robotic sequences.

---

## 🚀 Quickstart Reproducibility & Evaluation

To seamlessly bypass GUI or dependency installation drift, we provide deterministic experimental harnesses decoupled from visual engines for pure algorithm execution. 

### Unified Benchmark Harness
The unified evaluation harness computes Inconsistency Rates and Goal Success metrics iterating across `top-1` vs `top-k` algorithms mapping identical visual corruption constraints. 
```bash
python scripts/run_benchmarks.py --domain blocksworld --k 3
```

### Docker (Recommended Full Embodiment Evaluation)
For reviewers seeking to execute the explicit `CLIP` model and Unity simulation architectures specifically tracking Embodied Action selection loops:
```bash
docker build -t belief-pddl .
docker run --gpus all -v $(pwd)/outputs:/app/outputs belief-pddl
```

### Reviewer Metric Traces
All experiments automatically inject deterministic metric traces isolating exactly where classical Symbolic states failed vs successfully Projected MAP worlds executed optimal Pyperplan sequences:
```text
outputs/eval/alfworld_metrics.json     # Instantiation traces
outputs/benchmarks/benchmark_log.json  # SOTA Ablation traces
```

---

## Repository Implementation
```text
Belief-PDDL/
├── domains/              # Physics Axioms & CP-SAT Constraints (YAML)
│   ├── alfworld/         # General household instantiated parameters
│   └── blocksworld/      # Toy domains for fast ablation algorithms
├── notebooks/            # Mathematical Getting Started logic traces
├── scripts/              # Automated Experiment Harness scripts
│   ├── run_benchmarks.py    # The baseline empirical comparison matrix
│   └── run_alfworld_eval.py # The Unity Embodied explicit demonstration
├── src/                  # Core Architecture
│   ├── belief/           # The Factorized Updater & Generic Top-K Compiler
│   ├── perception/       # Decoupled VLM output interfaces
│   ├── planning/         # Weighted Top-K Deterministic Search Trees
```
