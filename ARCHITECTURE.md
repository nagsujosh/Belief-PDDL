# Architecture Deep Dive

This document explains each component of the pipeline in detail — the math, the design decisions, and how the pieces connect.

---

## 1. Perception Layer — `src/perception/clip_vision.py`

### What it does
Takes a raw RGB frame from the environment and outputs a **probability score for each symbolic predicate** (e.g. `on(block_0, block_1)`, `clear(block_2)`).

### How it works
We use OpenAI CLIP (`clip-vit-base-patch32` by default) as a **zero-shot visual classifier**. For each predicate `P`, we construct two natural language queries:

- **Positive**: `"A red block exactly on top of a blue block."`
- **Negative**: `"A red block is NOT touching the blue block."`

CLIP embeds both the image and the text into a shared 512-dimensional space. The probability of the predicate being true is:

```
P(pred = True | image) = softmax([sim(img, pos_text), sim(img, neg_text)])[0]
```

where `sim` is scaled cosine similarity using CLIP's learned `logit_scale`.

### Key classes

| Class | Purpose |
|:---|:---|
| `CLIPVisionBackbone` | Loads model, caches text embeddings, exposes `encode_image()` and `zero_shot_prob()` |

### Design decision: text embedding cache
Text descriptions for a given domain are fixed across an episode. We cache them after the first call so per-step cost is just one image forward pass, not `N_predicates × 2` text forward passes.

---

## 2. Belief Update Layer — `src/belief/`

### What it does
Maintains a **running probabilistic estimate** of every predicate over time, rather than taking a single snapshot from each perception call.

### `state.py` — `PredicateBelief`
A dictionary `{predicate_string: float}` mapping each grounded predicate to its current marginal probability. Initializes all predicates at a prior of `0.5` (maximum uncertainty).

### `update.py` — `BeliefUpdater`

Uses a **log-odds (logit) Bayesian filter**:

```
logit(p) = log(p / (1 - p))

l_t  =  l_{t-1}  +  α · logit(p̂_obs)  +  β · logit(p̂_action)

p_t  =  sigmoid(l_t)
```

**Why log-odds?** It converts probabilities to an unbounded real space where Bayesian updates are additive. A VLM saying `p=0.9` for a predicate that was previously `0.1` doesn't jump immediately to `0.9` — it shifts the log-odds gradually, making the belief resistant to single noisy observations.

**Parameters:**
- `alpha=1.0`: How much each VLM observation shifts the belief. Lower = more conservative / smoother.
- `beta=2.0`: Action effects (e.g. after `pickup(block_0)`, we know `holding(block_0)=True`) are weighted more strongly than VLM observations because they are deterministic.

---

## 3. CP-SAT Constraint Compiler — `src/belief/projection.py`

### What it does
Takes the current belief dictionary `{pred: prob}` and finds the **Top-K most probable logically consistent world states**.

### The problem it solves
If you ask CLIP 15 independent questions about a 3-block scene, you get 15 independent probabilities that have no awareness of each other. The result will frequently be logically impossible — e.g. `P(on_table(block_0)) = 0.9` and `P(on(block_0, block_1)) = 0.8` simultaneously, which violates the mutex rule `on_table(x) ∧ on(x,y) → ⊥`.

### MAP Inference via CP-SAT

We formulate this as a **pseudo-Boolean optimization** problem:

```
maximize   Σ_i  score(p_i) · x_i
subject to  logical constraints from constraints.yaml
```

where:
- `x_i ∈ {0, 1}` is the Boolean truth value of predicate `i`
- `score(p_i) = int(logit(p_i) × 1000)` — positive when `p_i > 0.5`, negative otherwise
- Constraints are compiled from YAML mutex and implication rules

This is exactly MAP inference in a Markov Random Field with hard logical constraints, solved using Google OR-Tools CP-SAT (an exact combinatorial solver).

### Top-K Extraction via Successive Blocking

After finding the MAP solution `w_1`, we add a **blocking constraint** that forbids that exact assignment, then solve again to get `w_2`, and so on. This produces K distinct feasible world hypotheses ranked by joint probability.

### YAML Constraint Format

```yaml
mutex:
  - ["pred_A(x)", "pred_B(x)"]     # pred_A and pred_B can't both be true
  - ["pred_C(x,y)", "pred_C(x,z)"] # x can't relate to two different objects

implications:
  - if: "pred_A(x,y)"
    then: "not pred_B(y)"           # if A then not B
```

The compiler grounds these abstract templates against all objects in the current scene via `itertools.product`, so adding a new domain requires only editing the YAML — zero Python changes.

---

## 4. Top-K Belief-Weighted Planner — `src/planning/sample_belief_planner.py`

### What it does
Given K consistent world states, selects the single best action to execute next.

### Algorithm

**Step 1 — Weight each world** by its log joint probability under the current belief:
```
w(world_k) = Σ_i  log P(pred_i = world_k[pred_i])
```
Normalized via softmax over log weights.

**Step 2 — Plan in each world** using Pyperplan (A* with FF heuristic) to find the optimal deterministic action sequence for each `w_k`. Extract the first action `a_k`.

**Step 3 — Aggregate** via weighted voting:
```
score(action a) = Σ_{k: first_action(w_k) = a}  weight(w_k)
```
Execute the action with the highest aggregated score.

**Step 4 — Sensing fallback**: If `max_score < 0.40`, the planner doesn't trust any world enough to act. Instead, compute **Information Gain** for each object:
```
IG(obj) = H(obj) × (1 - P(visible(obj)))

H(obj) = Σ_{pred mentioning obj}  H_binary(P(pred))
```
Emit a `reveal_side(obj)` sensing action targeting the highest-IG object. This collapses uncertainty for that object on the next step (active sensing oracle).

---

## 5. Deterministic Planner Wrapper — `src/planning/deterministic_planner.py`

### What it does
Translates a Python `{predicate: bool}` state dictionary into a valid PDDL problem file, invokes Pyperplan as a subprocess, and parses back the action sequence.

### PDDL generation
The `_generate_problem_pddl()` method dynamically constructs:
```pddl
(define (problem auto_gen)
  (:domain blocksworld)
  (:objects block_0 block_1 block_2 - block)
  (:init
    (clear block_0)
    (on_table block_1)
    (arm_empty)
    ...
  )
  (:goal (and (on block_0 block_1)))
)
```
Only predicates with `True` values in the projected world state appear in `:init`.

---

## 6. Execution Loop — `src/execution/replan_loop.py`

The outer loop that ties everything together for a single episode:

```
while not done and steps < max_steps:
    obs = env.observe()
    vlm_logits = clip.score(obs.rgb, predicates)
    belief = updater.update(belief, vlm_logits)
    top_k = projector.project_top_k_map_states(belief.probs, k=K)
    action, args = planner.select_action(belief.probs, top_k, goal, objects)
    obs, reward, done = env.step(action, args)
    log_metrics(...)
```

---

## Ablation Design

Five conditions systematically isolate each component's contribution:

| Mode | Belief | Verifier | Top-K | Sensing |
|:---|:---:|:---:|:---:|:---:|
| `threshold` | ❌ | ❌ | ❌ | ❌ |
| `belief_no_verifier` | ✅ | ❌ | ❌ | ❌ |
| `belief_plus_verifier` | ✅ | ✅ (K=1) | ❌ | ❌ |
| `full_top_k_no_sense` | ✅ | ✅ | ✅ | ❌ |
| `full_top_k` | ✅ | ✅ | ✅ | ✅ |

Each column adds exactly one component, making the causal contribution of each individually identifiable.
