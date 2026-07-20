# Quantum NAS for Variational Quantum Circuits

Evolutionary framework for designing quantum circuits using a Quantum Neural Architecture Search (QNAS) approach. Two distinct problem domains are implemented:

- **QNN (Quantum Neural Networks)** — classification on classical datasets
- **Max-Cut** — combinatorial optimization on weighted graph instances

Both domains support two ansatz construction strategies (rotation and Clifford), evolved through a multi-objective genetic algorithm (NSGA-II) via DEAP.

---

## QNN — Classification

Trains quantum circuits as classifiers on sklearn datasets. Located in `algorithms/qnn/`.

### Datasets

| Dataset | Features | Qubits | Classes |
|---------|----------|--------|---------|
| Iris | 4 | 4 | 3 |
| Wine | 13 | 4 | 3 |
| Breast Cancer | 30 | 5 | 2 |

### Encoding

**Amplitude encoding** (rotation approach): features are standardized (StandardScaler), reduced via PCA to `2^n_qubits` dimensions (padded with zeros if necessary), normalized to unit norm, and loaded via `QuantumCircuit.initialize()`.

**Clifford-angle encoding** (Clifford approach): PCA reduces features to `n_qubits` dimensions. Each feature is discretized into 4 buckets via percentile thresholds ([25th, 50th, 75th] quantiles), then mapped to H/X gates on each qubit (level 0: no gate, 1: H, 2: X, 3: X+H). CNOT gates connect adjacent qubits where both have level ≥ 2.

### Ansatz strategies

**Rotation approach** (`qnn_rotation.py`): each gene is either a Clifford gate (H, S, CX) or a parametric block (`PARAM_BLOCK`) structured as `("PARAM_BLOCK", p_type, p_idx, gate, qubit)`. `p_type` is `"INPUT"` (angle from `manual_input_values`) or `"WEIGHT"` (trainable weight via COBYLA). Parametric blocks apply RX/RY/RZ rotations. Weight optimization uses COBYLA (scipy) with 4 random starts (unless inherited) and a negative log-likelihood objective. Depth objective is `depth ** 1.2` by default.

**Clifford approach** (`qnn_clifford.py`): only Clifford gates — H, S, CX. Parametric blocks are structured as `("PARAM_BLOCK", param_idx, block_gates, tree)` where `block_gates` is a list of Clifford gates and `tree` is a **GP expression tree** (DEAP `gp.PrimitiveTree`) that controls how many times the block is repeated (clamped to `[1, 2]`). The tree is compiled from a primitive set supporting `add`, `sub`, `mul`, `safe_mod`, `sin`, `cos`, and ephemeral random constants `[-3, 3]`. It is evaluated with `(in_val, w_val)` to produce the repetition count. A Hall of Fame (`BLOCK_HOF`, max 100) caches useful blocks for reuse across individuals. A cache (`_COMPILED_TREE_CACHE`) avoids recompilation.

### Readout

Expectation values ⟨Z⟩ are computed from exact circuit probabilities. During training, a logistic regression (C=0.5, lbfgs, max_iter=200) with 3-fold cross-validation maps ⟨Z⟩ to class probabilities. During validation/testing, a pre-fitted classifier is used if available; otherwise softmax with β=5.0 is applied directly.

### Configuration

Main config: `algorithms/qnn/config.json`
Dataset-specific overrides: `algorithms/qnn/configs/{iris,wine,breast_cancer}.json`

Key config sections:

| Section | Parameters |
|---------|-----------|
| `qnn` | dataset, datasets_to_run, test_split, val_split |
| `population` | mu (parent pop), lambda (offspring) |
| `variation` | crossover_prob, mutation_prob, mutation_indpb, crossover_indpb |
| `evolution` | generations, patience, patience_window, improvement_epsilon, champion_inheritance_prob, champion_check_k, guided_individual_length_min, guided_individual_length_factor |
| `evaluation` | cobyla_maxiter_factor |
| `encoding` | enable_input_params, param_block_prob, manual_input_values |
| `execution` | multiprocessing, processes, device, statevector_max_parallel_threads |
| *(top-level)* | approach, random_seed, approaches |

### Usage

```powershell
# Single run
python algorithms/qnn/main.py --dataset iris --approach rotation
python algorithms/qnn/main.py --dataset iris --approach clifford

# All datasets sequentially
python algorithms/qnn/main.py --dataset all --approach rotation

# Both approaches on same dataset
python algorithms/qnn/main.py --dataset breast_cancer --approach both

# Multi-run (uses approaches and datasets_to_run from config)
python algorithms/qnn/main.py
```

---

## Max-Cut — Combinatorial Optimization

Evolves quantum circuits to approximate maximum-cut solutions on weighted graph instances. Located in `algorithms/general_max_cut/`.

### Instance format

Graph instances are stored as text files in `max_cut_instances/`:

```
num_nodes num_edges optimal_cut
u v weight
...
```

Nodes are 1-indexed. The header line includes the known optimal cut value. If absent, it is computed via brute force (≤20 nodes) or the networkx `one_exchange` heuristic.

Available instances: 10 nodes (15 files), 14 nodes (15 files), 20 nodes (15 files).

### Instance loading and splitting

Instances are loaded from `max_cut_instances/`, filtered by `instance_qubits_filter` and `max_qubits`, capped at `max_instances_per_size`. The loaded set is shuffled and split into training and validation sets according to `validation_fraction` and `seed`.

### Initial population

20% of the initial population is created via a **heuristic initializer** (`generate_heuristic_individual`): a DFS traversal of the graph places CX gates along edges, creating a circuit skeleton. The remaining 80% are generated via the standard guided individual generator.

### Ansatz strategies

**Rotation approach** (`max_cut_rotation.py`): same gene structure as QNN rotation — Clifford gates + parametric RX/RY/RZ blocks `("PARAM_BLOCK", p_type, p_idx, gate, qubit)`. Weight optimization via COBYLA with weight caching across generations.

**Clifford approach** (`max_cut_clifford.py`): parametric blocks are `("PARAM_BLOCK", param_idx, block_gates, tree)` — a list of Clifford gates plus a GP expression tree evaluated with `(in_val, w_val)` to produce the repetition count (clamped to `[1, 8]`). Uses the same primitive set as QNN Clifford (`add`, `sub`, `mul`, `safe_mod`, `sin`, `cos`, ephemeral constants). A Hall of Fame (`BLOCK_HOF`) caches useful blocks. Block reuse probability is 0.3.

### Fitness evaluation

Circuits are executed on `AerSimulator` (Qiskit Aer) with shot annealing (`shots_start` → `shots_end` across generations). Weight optimization uses COBYLA (`max(50, num_weights * 15)` maxiter, `rhobeg=0.5`) with the objective: `-(mean_ar - 0.2 * std_ar)` across training instances, where `mean_ar` is the average CVaR-based approximation ratio. A cache (`_WEIGHT_CACHE`) seeds each generation's optimization from the previous optimum.

The `max_cut_fitness` function implements a CVaR (Conditional Value at Risk) approach:
- Sort measurement outcome bitstrings by cut value descending
- Accumulate probability mass up to γ (α parameter)
- Return the probability-weighted average cut value over that mass
- Normalize by the optimal cut to get the approximation ratio

A gamma schedule anneals from `initial` (coarse exploration) to `final` (fine exploitation) across generations.

### Validation

The champion circuit is executed on the held-out validation instances with `final_validation_shots` shots. The per-instance approximation ratio is computed with γ=1.0 (full distribution).

### Configuration

`algorithms/general_max_cut/config.json`

| Section | Parameters |
|---------|-----------|
| `population` | mu, lambda |
| `variation` | crossover_prob, mutation_prob, mutation_indpb |
| `evolution` | generations, patience, patience_window, improvement_epsilon, guided_individual_length_min, guided_individual_length_factor |
| `gamma_schedule` | initial, final gamma |
| `evaluation` | shots_start, shots_end, final_validation_shots |
| `encoding` | enable_input_params, param_block_prob, manual_input_values |
| `circuit_scale` | max_qubits, instance_qubits_filter, max_instances_per_size |
| `instance_split` | validation_fraction, seed |
| `execution` | multiprocessing, processes, device, stabilizer_max_parallel_threads, statevector_max_parallel_threads |
| *(top-level)* | approach, approaches |

### Usage

```powershell
# Single approach
python algorithms/general_max_cut/main.py --approach rotation
python algorithms/general_max_cut/main.py --approach clifford

# All approaches (from config.approaches)
python algorithms/general_max_cut/main.py
```

### Batch benchmarks

`run_max_cut_benchmarks.py` at the project root orchestrates batch runs across instances and approaches. Results can be exported to LaTeX via `results_to_latex.py`.

---

## Shared techniques across both domains

### NSGA-II multi-objective optimization

Both modules use DEAP's `selNSGA2` for Pareto-front selection. The two fitness objectives are:

1. **Soft score** (QNN) / **Approximation ratio** (Max-Cut) — minimized as negative values
2. **Circuit depth** — measured after Qiskit transpiler simplification (see table below)

### Genetic operators

| Operator | Description |
|----------|-------------|
| **Crossover** (`cx_quantum_circuit`) | Gene swap between individuals, restricted to the variable-length suffix (beyond the initial H-gate prefix). Only swaps genes where both parents share the same gate type. |
| **Mutation** (`mut_quantum_circuit`) | Per-gene INSERT/DELETE/REPLACE with `indpb` probability. PARAM_BLOCK genes in Clifford approach are mutated at the block-structure or GP-tree level rather than replaced as a whole. |
| **Adaptive mutation (QNN)** | Mutation rate increases with stagnation: `indpb = base * (1 + 5.0 * stagnant/patience)`, capped at 0.7. Triggers when `stagnant > patience/5`. |
| **Adaptive mutation (Max-Cut)** | `indpb = base * (1 + 0.5 * stagnant/patience)`, capped at 0.3. Triggers when `stagnant > patience/3`. |

### Champion inheritance (QNN only)

The best individual's weights are propagated to offspring via noisy inheritance (`N(0, 0.05)`) with probability `champion_inheritance_prob` (minimum 0.6). Individuals with their own prior weights (`stored_thetas`) use those instead. This provides a form of Lamarckian evolution.

### Diversity mechanisms (QNN only)

- **Immigrant injection**: when stagnation exceeds `patience/4`, random immigrants are added (up to `mu/3`, min 4). With 50% probability, a mutated copy of the champion is included.
- **Population reset**: when unique fitness count drops below `mu/4` or patience is nearly exhausted, the population is reset around the champion (60% mutated copies, 40% fresh individuals), up to `max_champion_resets` times.

Max-Cut does not implement these mechanisms.

### Depth optimization

Depth is optimized as a secondary objective. Simplification passes depend on approach and domain:

| Domain | Approach | Passes |
|--------|----------|--------|
| QNN | Rotation | `CommutativeCancellation` + `Optimize1qGatesDecomposition` (basis: rx, ry, rz, h, s, cx) |
| QNN | Clifford | `CommutativeCancellation` (basis: h, s, cx) |
| Max-Cut | Both | Via `build_quantum_circuit` output (no explicit simplification for depth measurement) |

### Simulation

| Approach | Simulator | Method |
|----------|-----------|--------|
| QNN Rotation | Qiskit `Statevector` (numpy) | Exact amplitudes, no shots |
| QNN Clifford | Qiskit `StabilizerState` | Exact stabilizer probabilities, no shots |
| Max-Cut Rotation | Qiskit Aer `AerSimulator` | Shot-based, method=statevector |
| Max-Cut Clifford | Qiskit Aer `AerSimulator` | Shot-based, method=stabilizer |

### Early stopping

Training halts when the best fitness does not improve for `patience` generations (with `improvement_epsilon` tolerance). QNN additionally uses a smoothed validation accuracy over a `patience_window` for more stable convergence detection.

---

## Project structure

```
TFM/
├── __init__.py
├── algorithms/
│   ├── __init__.py
│   ├── qnn/                          # Classification (QNN)
│   │   ├── __init__.py
│   │   ├── config.json               # Default configuration
│   │   ├── configs/                  # Dataset-specific configs
│   │   │   ├── iris.json
│   │   │   ├── wine.json
│   │   │   └── breast_cancer.json
│   │   ├── constants.py              # Shared constants & magic numbers
│   │   ├── config.py                 # Config loading & global state
│   │   ├── main.py                   # Entry point & evolution loop
│   │   ├── qnn_common.py             # Shared GA operators (cx, mut, apply_block, simplify)
│   │   ├── qnn_rotation.py           # Rotation approach (gates + parametric blocks)
│   │   ├── qnn_clifford.py           # Clifford approach + GP expression trees
│   │   ├── qnn_data.py               # Dataset loading & encoding
│   │   ├── qnn_baselines.py          # sklearn baselines
│   │   ├── utils.py                  # Evaluation, metrics, readout, caching
│   │   ├── plotting.py               # Evolution progress plots
│   │   └── timing.py                 # Timing aggregation (max for parallel, sum for serial)
│   │
│   └── general_max_cut/              # Max-Cut optimization (multi-instance)
│       ├── __init__.py
│       ├── config.json               # Configuration
│       ├── main.py                   # Entry point & evolution loop
│       ├── max_cut_common.py         # Shared utils, GA, instance loading, max_cut_fitness
│       ├── max_cut_rotation.py       # Rotation approach
│       ├── max_cut_clifford.py       # Clifford approach + GP trees
│       ├── utils.py                  # Config, evaluation, simulator setup, caching
│       └── timing.py                 # Timing aggregation
│
├── max_cut_instances/                # Graph instances for Max-Cut
│   ├── instance_10nodes_*.txt        # 10-node instances (15 files)
│   ├── instance_14nodes_*.txt        # 14-node instances (15 files)
│   └── instance_20nodes_*.txt        # 20-node instances (15 files)
│
├── results/                          # Output directory
│   ├── qnn/                          # QNN classification results (by dataset/approach)
│   └── general_max_cut/              # Max-Cut results (by qubits/approach)
│
├── generate_instances.py             # Max-Cut instance generator
├── run_max_cut_benchmarks.py         # Batch Max-Cut benchmark runner
├── results_to_latex.py               # Export results to LaTeX
└── README.md
```

## Output

**QNN** — timestamped files in `results/qnn/{dataset}/{approach}/`:

| File | Contents |
|------|----------|
| `*_genotype.json` | Full individual genotype (gates, params, weights) |
| `*_architecture.json` | Human-readable architecture description |
| `*.qpy` | Qiskit QPY serialized circuit |
| `*.pdf` | Circuit diagram + evolution progress plots |
| `*.json` | Complete run data (config, results, timing, history) |

**Max-Cut** — timestamped files in `results/general_max_cut/{qubits}_qubits/{approach}/`:

| File | Contents |
|------|----------|
| `*_genotype.json` | Full individual genotype (gates, params, weights) |
| `*_architecture.json` | Human-readable architecture description |
| `*.qpy` | Qiskit QPY serialized circuit |
| `*.pdf` | Circuit diagram |
| `*.png` | Evolution progress plot (AR vs Depth) |
| `*.json` | Complete run data (config, results, timing, history) |

## Dependencies

- Python ≥3.10
- Qiskit (`qiskit`, `qiskit-aer`)
- DEAP
- scikit-learn (QNN only)
- NetworkX (Max-Cut only)
- SciPy
- NumPy
- Matplotlib (+ Seaborn for QNN plots)
