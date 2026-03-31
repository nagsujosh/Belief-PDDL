# Constrained Predicate Belief Projection for Neuro-Symbolic Planning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OR-Tools](https://img.shields.io/badge/solver-OR--Tools%20CP--SAT-orange.svg)](https://developers.google.com/optimization)

A neuro-symbolic planning framework that maintains **factorized symbolic belief states**, projects them through a **CP-SAT constraint compiler**, and plans over the **Top-K feasible worlds** with active uncertainty-driven sensing.

---

## Research Goal

Classical AI planning assumes a fully observed, logically consistent world state. In practice, perception systems produce noisy, **physically inconsistent** predicate estimates — e.g. simultaneously predicting a block is both `held` and `on_table`, or failing to bind the correct target instance in a cluttered household scene. Feeding such states directly into a symbolic planner causes catastrophic failures.

This framework answers: **how do we plan reliably when perception is noisy and the logical state space is partially observed?**

The core contribution is a three-stage pipeline:
1. **Factorized Bayesian belief tracking** over grounded predicates
2. **Constraint-based symbolic projection** (CP-SAT MAP inference) to eliminate physically impossible world states
3. **Top-K world planning with active sensing** to act under irreducible uncertainty
4. **Environment-specific grounding** to map symbolic actions back to executable commands

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Perception Layer                           │
│   RGB → CLIP / learned heads  OR  text → symbolic parser        │
│   (src/perception/clip_vision.py, alfworld_text.py)             │
└───────────────────────────┬─────────────────────────────────────┘
                            │  raw probabilities p̂(pred_i)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Belief Update Layer                          │
│   Log-odds Bayesian filter: l_t = l_{t-1} + α·logit(p̂_obs)    │
│   (src/belief/update.py, src/belief/state.py)                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │  continuous marginals B_t = {p(pred_i)}
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                 CP-SAT Constraint Compiler                      │
│   YAML domain rules → mutex/implication constraints →           │
│   Top-K MAP feasible world assignments W = {w_1,...,w_K}        │
│   (src/belief/projection.py, domains/*/constraints.yaml)        │
└───────────────────────────┬─────────────────────────────────────┘
                            │  K consistent Boolean world states
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│               Top-K Belief-Weighted Planner                     │
│   For each w_k: run Pyperplan → get first action a_k            │
│   Score actions by softmax-weighted joint probability           │
│   If max score < 0.40: trigger Active Sensing (IG)              │
│   (src/planning/sample_belief_planner.py)                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │  selected action
                            ▼
                    Environment / Execution
```

For a deeper walkthrough see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Latest Validated Results

### Blocksworld

Corrected fixed-seed sweeps (`seeds=0..4`, `3` episodes per seed) show that the repaired projection layer makes the full planner robust across substantial observation noise.

| Setting | Success | Avg. Steps | Avg. Sensing | Source |
|:---|:---:|:---:|:---:|:---|
| `full_top_k`, `zero_shot`, `noise=0.5`, `decay=0.7` | `100%` | `4.2` | `1.0` | local seed sweep |
| `full_top_k`, `calibrated`, `noise=0.5`, `decay=0.7` | `100%` | `5.5` | `1.3` | local seed sweep |
| `full_top_k`, `learned_head`, `noise=0.5`, `decay=0.7` | `100%` | `4.13` | `0.87` | local seed sweep |
| `full_top_k`, `zero_shot`, `noise=0.7`, `decay=0.7` | `100%` | `4.53` | `0.87` | local seed sweep |
| `full_top_k`, `learned_head`, `noise=0.7`, `decay=0.7` | `100%` | `4.07` | `1.0` | local seed sweep |
| `full_top_k`, `zero_shot`, `noise=0.9`, `decay=0.7` | `100%` | `4.33` | `0.8` | local seed sweep |

### ALFWorld

The current ALFWorld smoke test `put_egg_in_microwave` now succeeds end to end in `5` steps:

1. `goto_location sinkbasin_1`
2. `take_from_surface egg sinkbasin_1`
3. `goto_location microwave_1`
4. `open_receptacle microwave_1`
5. `put_in_container egg microwave_1`

This result is currently a **smoke-test success**, not yet a full ALFWorld benchmark table.

### What Counts As a Baseline Here

Implemented Blocksworld planning baselines:

- `threshold`: hard-threshold predicates, no belief state, no projection
- `belief_no_verifier`: belief update only, no projection
- `belief_plus_verifier`: belief update plus single projected world
- `full_top_k_no_sense`: Top-K projection without sensing fallback
- `full_top_k`: full system

Implemented perception baselines:

- `zero_shot`
- `calibrated`
- `learned_head`

Current public-ready result story:

- Blocksworld baseline machinery is implemented and the corrected full-system sweeps are strong.
- ALFWorld is now working as a live environment path, but still needs a broader comparative evaluation before claiming benchmark-level superiority.

---

## Repository Structure

```
Belief-PDDL/
├── domains/
│   ├── blocksworld/
│   │   ├── constraints.yaml     # Mutex & implication rules for CP-SAT
│   │   ├── predicates.yaml      # Predicate taxonomy (unary / binary)
│   │   └── domain.pddl          # PDDL domain for Pyperplan
│   └── alfworld/
│       ├── constraints.yaml
│       ├── predicates.yaml
│       └── domain.pddl
├── src/
│   ├── belief/
│   │   ├── state.py             # PredicateBelief: dict of marginals
│   │   ├── update.py            # BeliefUpdater: log-odds Bayesian filter
│   │   └── projection.py        # BeliefProjector: CP-SAT Top-K compiler
│   ├── perception/
│   │   ├── clip_vision.py       # CLIPVisionBackbone (zero-shot scoring)
│   │   ├── backbones.py         # Generic encoder interface
│   │   ├── unary_head.py        # MLP head for unary predicates
│   │   ├── binary_head.py       # MLP head for binary predicates
│   │   └── calibrate.py         # Temperature scaling calibration
│   ├── planning/
│   │   ├── deterministic_planner.py   # Pyperplan wrapper
│   │   └── sample_belief_planner.py   # Top-K weighted planner + IG
│   ├── execution/
│   │   └── replan_loop.py       # Belief-aware replanning loop
│   └── envs/
│       ├── blocksworld_env.py   # Procedural Blocksworld environment
│       └── alfworld_env.py      # ALFWorld / AI2-THOR wrapper
├── scripts/
│   ├── run_benchmarks.py        # Ablation harness (CLIP + CP-SAT)
│   ├── run_alfworld_eval.py     # Full 3D ALFWorld evaluation
│   ├── collect_rollouts.py      # Offline episode data collection
│   └── test_belief.py           # Unit tests for belief module
├── tests/mocks/
│   └── blocksworld_env.py       # Semantic mock env with PIL renderer
├── notebooks/
│   └── blocksworld_demo.ipynb   # Getting started walkthrough
├── run_all_experiments.sh       # Full ablation orchestrator
├── Dockerfile                   # Headless CUDA + xvfb container
├── requirements.txt
└── pyproject.toml
```

---

## Installation

### Option A — Local (Recommended for RTX GPU)

```bash
# 1. Clone
git clone https://github.com/nagsujosh/Belief-PDDL.git
cd Belief-PDDL

# 2. Create environment (Conda recommended for CUDA pinning)
conda create -n belief_pddl python=3.10 -y
conda activate belief_pddl

# 3. Install PyTorch with CUDA 12.1+ (required for RTX 4090/5090)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining dependencies
pip install -r requirements.txt
pip install -e .

# 5. Download ALFWorld data (~3-5 GB, only needed for ALFWorld eval)
export ALFWORLD_DATA=~/.cache/alfworld
alfworld-download && alfworld-download --extra
```

### Option B — Docker (Headless / Reproducibility)

```bash
docker build -t belief-pddl .
docker run --gpus all -v $(pwd)/outputs:/app/outputs belief-pddl
```

---

## Running Experiments

### Blocksworld Ablation Suite

Run the full ablation matrix:

```bash
./run_all_experiments.sh
```

This sequentially runs the main ablation conditions and writes results to `outputs/benchmarks/*.jsonl`.

**Run a single condition manually:**

```bash
# Full system: Belief + CP-SAT + Top-K + Active Sensing
python scripts/run_benchmarks.py --mode full_top_k --noise 0.3 --k 3 --episodes 10

# Ablation: no verifier (raw belief → planner)
python scripts/run_benchmarks.py --mode belief_no_verifier --noise 0.3 --episodes 10

# Ablation: no sensing (Top-K but no IG fallback)
python scripts/run_benchmarks.py --mode full_top_k_no_sense --noise 0.5 --k 3 --episodes 10

# Baseline: hard threshold (VLM prob > 0.5 → True)
python scripts/run_benchmarks.py --mode threshold --noise 0.3 --episodes 10
```

**CLI Arguments for `run_benchmarks.py`:**

| Argument | Default | Description |
|:---|:---|:---|
| `--mode` | `full_top_k` | `threshold`, `belief_no_verifier`, `belief_plus_verifier`, `full_top_k_no_sense`, `full_top_k` |
| `--noise` | `0.3` | VLM hallucination rate (0.0 = perfect, 1.0 = completely adversarial) |
| `--k` | `3` | Number of Top-K feasible worlds to enumerate |
| `--episodes` | `5` | Number of independent evaluation episodes |

**Summarize results after a run:**

```bash
python summarize_results.py
```

**Recommended fixed-seed comparison run:**

```bash
python scripts/run_seed_sweep.py \
  --mode full_top_k \
  --noise 0.5 \
  --episodes 3 \
  --k 3 \
  --alpha 1.0 \
  --decay 0.7 \
  --device cuda \
  --perception-mode learned_head \
  --learned-head-path outputs/perception/perception_compare_e200_learned_head.pt \
  --seeds 0,1,2,3,4 \
  --output outputs/benchmarks/seed_sweep_noise_0.5_decay_0.7_learned_head.json
```

---

### ALFWorld Evaluation (Full 3D Embodied)

Requires an ALFWorld text-game dataset root containing generated games
(`traj_data.json` and `game.tw-pddl`) plus a display when using the visual stack.
For the current text-first benchmark, pass the dataset root explicitly:

```bash
# Text-first symbolic benchmark on the default preset
python scripts/run_alfworld_eval.py \
  --task-preset put_egg_in_microwave \
  --alfworld-data /path/to/alfworld/data

# Headless (SSH / server)
xvfb-run -a python scripts/run_alfworld_eval.py \
  --task-preset put_egg_in_microwave \
  --alfworld-data /path/to/alfworld/data
```

Results are written to `outputs/eval/alfworld_metrics.json`.

The current validated smoke task is:

```bash
python scripts/run_alfworld_eval.py \
  --task-preset put_egg_in_microwave \
  --alfworld-data ~/.cache/alfworld \
  --episodes 1 \
  --max-steps 15 \
  --alpha 1.0 \
  --decay 1.0 \
  --k 3
```

---

## Hyperparameters

### Belief Update (`src/belief/update.py` — `BeliefUpdater`)

| Parameter | Default | Meaning |
|:---|:---:|:---|
| `alpha` | `1.0` | Weight applied to incoming VLM observation evidence in log-odds space. Higher = faster belief swing per observation |
| `beta` | `2.0` | Weight for deterministic action effect updates (e.g. after executing `pickup`, block is definitely held) |

The update rule is:
```
l_t = l_{t-1} + alpha * logit(p_obs) + beta * logit(p_action)
p_t = sigmoid(l_t)
```

### CP-SAT Projector (`src/belief/projection.py` — `BeliefProjector`)

| Parameter | Default | Meaning |
|:---|:---:|:---|
| `k` | `3` | Number of Top-K MAP feasible worlds to enumerate via successive blocking |
| score precision | `×1000` | Log-odds scores are scaled to integers (CP-SAT requires integer coefficients) |

The objective being maximized:
```
maximize  Σ_i  score(p_i) · x_i
subject to  YAML mutex constraints
            YAML implication constraints
```
where `score(p) = int(log(p / (1-p)) × 1000)`.

### Top-K Planner (`src/planning/sample_belief_planner.py` — `SampleBeliefPlanner`)

| Parameter | Default | Meaning |
|:---|:---:|:---|
| Sensing threshold | `0.40` | If the top-voted action scores below 40% softmax weight across K worlds, switch to active sensing |
| IG formula | `H(obj) × (1 - P(visible(obj)))` | Information Gain of sensing an object = its total entropy × probability it is currently hidden |

### CLIP Vision Backbone (`src/perception/clip_vision.py`)

| Parameter | Default | Meaning |
|:---|:---:|:---|
| `model_name` | `openai/clip-vit-base-patch32` | HuggingFace model ID. Swap to `openai/clip-vit-large-patch14` for stronger features |
| `device` | auto (`cuda` if available, else `cpu`) | Inference device |

To switch to the larger model (recommended for GPU runs):
```python
# In scripts/run_benchmarks.py __init__:
self.vlm = CLIPVisionBackbone(model_name="openai/clip-vit-large-patch14")
```

---

## Domain Configuration

Domains are defined entirely in `domains/<domain>/`:

**`constraints.yaml`** — rules compiled into CP-SAT constraints:
```yaml
mutex:
  - ["on_table(x)", "on(x,y)"]   # block can't be on table AND on another block
  - ["holding(x)", "arm_empty()"] # can't hold something and have empty arm

implications:
  - if: "on(x,y)"
    then: "not clear(y)"          # if x is on y, y is not clear
```

**`domain.pddl`** — standard PDDL for Pyperplan:
```pddl
(:action pickup
  :parameters (?x - block)
  :precondition (and (clear ?x) (on_table ?x) (arm_empty) (visible ?x))
  :effect (and (not (on_table ?x)) (holding ?x) (not (arm_empty)))
)
```

Adding a new domain requires only: `constraints.yaml`, `predicates.yaml`, `domain.pddl`.

---

## Output Files

Benchmark outputs are written to `outputs/benchmarks/` with descriptive filenames that may include noise, alpha, decay, perception mode, and seed.

Per-episode JSONL outputs look like:
```json
{
  "episode": 0,
  "mode": "full_top_k",
  "noise_level": 0.3,
  "success": true,
  "steps": 8,
  "inconsistency_events": 1,
  "sensing_actions": 3,
  "replans": 0
}
```

Aggregate seed sweeps and ALFWorld smoke traces are also written under:

- `outputs/benchmarks/*.json`
- `outputs/eval/*.json`

Note: `outputs/` is gitignored, so if you want results visible on GitHub, summarize them in the README or another committed doc.

---

## Citation

If you use this codebase, please cite:
```bibtex
@misc{belief-pddl-2026,
  author = {Sujosh Nag},
  title  = {Constrained Predicate Belief Projection for Neuro-Symbolic Planning},
  year   = {2026},
  url    = {https://github.com/nagsujosh/Belief-PDDL}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
