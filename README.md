# Quantum NAS for Variational Quantum Circuits

Evolutionary framework for designing quantum circuits using a Quantum Neural Architecture Search (QNAS) approach. Two distinct problem domains are implemented:

- **QNN (Quantum Neural Networks)** — classification on classical datasets
- **Max-Cut** — combinatorial optimization on graph instances

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

**Amplitude encoding** (rotation approach): feature vector is normalized and loaded via `QuantumCircuit.initialize()`, producing a statevector of size `2^n_qubits`. PCA reduces dimensions to fit.

**Clifford-angle encoding** (Clifford approach): each feature is discretized into 4 buckets via percentile thresholds, then mapped to H/X gates on each qubit. CNOT gates connect adjacent qubits where both are in the upper half of buckets.

### Ansatz strategies

**Rotation approach** (`qnn_rotation.py`): each gene is either a Clifford gate (H, S, CX) or a parametric block (`PARAM_BLOCK`) specified as `(type, param_idx, gate, qubit)`. Parametric blocks apply RX/RY/RZ rotations with angles provided either by input values or trainable weights. Weight optimization uses COBYLA (scipy) with multiple random starts and a negative log-likelihood objective.

**Clifford approach** (`qnn_clifford.py`): only Clifford gates — H, S, CX. Parametric blocks are structured as `(param_idx, block_gates, reps)` where `block_gates` is a list of Clifford gates and `reps` controls how many times the block is repeated. A Hall of Fame (`BLOCK_HOF`) caches useful blocks for reuse across individuals. Simulation uses `StabilizerState` for exact probabilities without sampling.

### Readout

A logistic regression layer maps qubit expectation values ⟨Z⟩ to class probabilities. Cross-validation (3-fold) during training; direct softmax during validation/testing. The sigmoid temperature (`beta=5.0`) controls softmax sharpness.

### Configuration

Main config: `algorithms/qnn/config.json`
Dataset-specific overrides: `algorithms/qnn/configs/{iris,wine,breast_cancer}.json`

Key config sections:

| Section | Parameters |
|---------|-----------|
| `qnn` | dataset, test_split, val_split |
| `population` | mu (parent pop), lambda (offspring) |
| `variation` | crossover/mutation probabilities |
| `evolution` | generations, patience, champion settings |
| `evaluation` | cobyla_maxiter_factor |
| `encoding` | enable_input_params, param_block_prob, manual_input_values |
| `execution` | multiprocessing, device |

### Usage

```powershell
# Single run
python algorithms/qnn/main.py --dataset iris --approach rotation
python algorithms/qnn/main.py --dataset iris --approach clifford

# All datasets sequentially
python algorithms/qnn/main.py --dataset all --approach rotation

# Both approaches on same dataset
python algorithms/qnn/main.py --dataset breast_cancer --approach both

# Multi-run (uses approaches/datasets_to_run from config)
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

Available instances: 10 nodes (14 files), 14 nodes (10 files), 20 nodes (15 files).

### Encoding

Each graph edge is implicitly encoded via CX gates in the ansatz. A heuristic initializer (`generate_heuristic_individual`) creates a circuit skeleton from a DFS traversal of the graph, placing CX gates along edges.

### Ansatz strategies

**Rotation approach** (`max_cut_rotation.py`): same structure as QNN rotation — Clifford gates + parametric RX/RY/RZ blocks. `load_external_maxcut_instance` is re-exported from `max_cut_common`. Weight optimization via COBYLA.

**Clifford approach** (`max_cut_clifford.py`): parametric blocks contain a list of Clifford gates plus a **GP expression tree** that controls repetition count. The expression is compiled from a DEAP primitive set (`pset`) supporting `add`, `sub`, `mul`, `safe_mod`, `sin`, `cos`, and ephemeral random constants. At circuit construction time, the tree is evaluated with `(in_val, w_val)` to produce an integer repetition count (clamped to `[1, 8]`). A cache (`_COMPILED_TREE_CACHE`) avoids recompilation.

### Fitness evaluation

Circuits are executed on `AerSimulator` (Qiskit Aer) with configurable shot counts (shot annealing from `shots_start` to `shots_end` across generations). The `max_cut_fitness` function computes the approximate ratio by taking an α-quantile (alpha-gamma schedule) of the weighted bitstring outcomes:

- Sort measurement outcomes by cut value descending
- Accumulate probability mass up to γ (gamma schedule, annealed from 1.0 to 0.3)
- Return the probability-weighted average cut value over that mass
- Normalize by the optimal cut to get the approximation ratio

A gamma schedule anneals from coarse exploration (high γ) to fine exploitation (low γ).

### Baselines

`run_max_cut_benchmarks.py` at the project root orchestrates batch runs.

### Configuration

`algorithms/general_max_cut/config.json`

| Section | Parameters |
|---------|-----------|
| `population` | mu, lambda |
| `variation` | crossover/mutation probabilities |
| `evolution` | generations, patience |
| `gamma_schedule` | initial, final gamma |
| `evaluation` | shots_start, shots_end, final_validation_shots |
| `encoding` | enable_input_params, param_block_prob, manual_input_values |
| `circuit_scale` | max_qubits, instance_qubits_filter, max_instances_per_size |
| `instance_split` | validation_fraction, seed |
| `execution` | multiprocessing, device, thread counts |

### Usage

```powershell
# Single approach
python algorithms/general_max_cut/main.py --approach rotation
python algorithms/general_max_cut/main.py --approach clifford

# All approaches (from config.approaches)
python algorithms/general_max_cut/main.py
```

---

## Shared techniques across both domains

### NSGA-II multi-objective optimization

Both modules use DEAP's `selNSGA2` for Pareto-front selection. The two fitness objectives are:

1. **Soft score** (QNN) / **Approximation ratio** (Max-Cut) — minimized as negative values
2. **Circuit depth** — measured after Qiskit transpiler simplification (`CommutativeCancellation` + `Optimize1qGatesDecomposition`/`Optimize1qGatesSimpleCommutation`)

### Genetic operators

| Operator | Description |
|----------|-------------|
| **Crossover** (`cx_quantum_circuit`) | Uniform gene swap between individuals, restricted to the variable-length suffix (beyond the initial H-gate prefix) |
| **Mutation** (`mut_quantum_circuit`) | Per-gene INSERT/DELETE/REPLACE with `indpb` probability. INSERT/DELETE change circuit length; REPLACE swaps a gate |
| **Adaptive mutation** | Mutation rate increases with stagnation (`indpb = base * (1 + 5 * stagnant/patience)`) |

### Champion inheritance

The best individual's weights are propagated to offspring via noisy inheritance (`N(0, 0.05)`), providing a form of Lamarckian evolution.

### Diversity mechanisms

- Immigrant injection when stagnation exceeds `patience/4`
- Population reset around champion when unique fitness count drops below `mu/4` (up to `max_champion_resets` times)

### Simulation

| Approach | Simulator | Method |
|----------|-----------|--------|
| QNN Rotation | Qiskit `Statevector` | Exact amplitudes |
| QNN Clifford | Qiskit `StabilizerState` | Exact stabilizer probabilities |
| Max-Cut Rotation | Qiskit Aer `AerSimulator` | Shot-based sampling (statevector) |
| Max-Cut Clifford | Qiskit Aer `AerSimulator` | Shot-based sampling (stabilizer) |

### Early stopping

Training halts when validation accuracy / approximation ratio does not improve for `patience` generations.

---

## Project structure

```
TFM/
├── algorithms/
│   ├── qnn/                          # Classification (QNN)
│   │   ├── config.json               # Default configuration
│   │   ├── configs/                  # Dataset-specific configs
│   │   │   ├── iris.json
│   │   │   ├── wine.json
│   │   │   └── breast_cancer.json
│   │   ├── constants.py              # Shared constants & magic numbers
│   │   ├── config.py                 # Config loading & global state
│   │   ├── main.py                   # Entry point & evolution loop
│   │   ├── qnn_common.py             # Shared GA operators
│   │   ├── qnn_rotation.py           # Rotation approach
│   │   ├── qnn_clifford.py           # Clifford approach
│   │   ├── qnn_data.py               # Dataset loading & encoding
│   │   ├── qnn_baselines.py          # sklearn baselines
│   │   ├── utils.py                  # Evaluation & metrics
│   │   ├── plotting.py               # Evolution progress plots
│   │   └── timing.py                 # Timing aggregation
│   │
│   └── general_max_cut/              # Max-Cut optimization
│       ├── config.json               # Configuration
│       ├── main.py                   # Entry point & evolution loop
│       ├── max_cut_common.py         # Shared utils, GA, instance loading
│       ├── max_cut_rotation.py       # Rotation approach
│       ├── max_cut_clifford.py       # Clifford approach + GP trees
│       ├── utils.py                  # Config, evaluation, simulator setup
│       └── timing.py                 # Timing aggregation
│
├── max_cut_instances/                # Graph instances for Max-Cut
│   ├── instance_10nodes_*.txt        # 10-node instances (14 files)
│   ├── instance_14nodes_*.txt        # 14-node instances (10 files)
│   └── instance_20nodes_*.txt        # 20-node instances (15 files)
│
├── results/                          # Output directory
│   ├── qnn/
│   └── general_max_cut/
│
├── generate_instances.py             # Max-Cut instance generator
├── run_max_cut_benchmarks.py         # Batch Max-Cut benchmark runner
├── results_to_latex.py               # Export results to LaTeX
├── tablas_memoria.tex                # LaTeX tables for thesis
└── README.md
```

## Output

Each run creates timestamped files in `results/{qnn,general_max_cut}/{dataset_name}/{approach}/`:

| File | Contents |
|------|----------|
| `*_genotype.json` | Full individual genotype (gates, params, weights) |
| `*_architecture.json` | Human-readable architecture description |
| `*.qpy` | Qiskit QPY serialized circuit |
| `*.pdf` | Circuit diagram + evolution progress plots |
| `*.json` | Complete run data (config, results, timing, history) |

## Dependencies

- Python ≥3.10
- Qiskit (`qiskit`, `qiskit-aer`)
- DEAP
- scikit-learn (QNN only)
- NetworkX (Max-Cut only)
- SciPy
- NumPy
- Matplotlib + Seaborn (QNN only)
