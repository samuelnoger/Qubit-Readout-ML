# Qubit Readout Classification with Neural Networks

A self-contained project on **dispersive qubit readout**: deciding whether a superconducting qubit was prepared in $\vert{}0\rangle$ or $\vert{}1\rangle$ from a noisy microwave measurement record. Records are simulated with QuTiP, and a 1D CNN is compared against standard classical readout methods (integrated threshold, matched filter) and stronger joint linear baselines (LDA).

<img src="results/figures/sweep_t1.png" alt="Fidelity vs T1" width="400"><img src="results/figures/sweep_zeta.png" alt="Fidelity vs cross-dispersive shift" width="400"><img src="results/figures/sweep_leak.png" alt="Fidelity vs signal leakage" width="400">

---

## Background

In circuit QED, a qubit is read out through a microwave cavity. The qubit state shifts the cavity response, returning a complex amplitude signal that is integrated over time. Two physical mechanisms complicate this classification:

1. **$T_1$ decay:** A qubit prepared in $\vert{}1\rangle$ can relax during the readout window, switching the record from "excited-like" to "ground-like" partway through.
2. **Crosstalk:** In multiplexed readouts, one qubit's signal leaks into another's channel (linear leakage), and one qubit's state can shift another's resonator (cross-dispersive shift).

The metric is **assignment fidelity**, $1 - \frac{P(0\vert{}1) + P(1\vert{}0)}{2}$.

## Methods

**Simulation Model:**
- *Qubit trajectories:* QuTiP `mcsolve` runs quantum-jump trajectories with a $T_1$ collapse operator.
- *Cavity response:* Solves the cavity field piecewise-exactly over time, adding Gaussian noise. For two qubits, independent decay is coupled with linear measurement leakage ($\eta$) and nonlinear cross-dispersive shifts ($\zeta$). This is a semi-classical model.

**Classifiers:**
- *Independent Filters:* Integrated threshold and matched filter. These only see their own qubit's channel.
- *Joint Linear Discriminant (LDA):* The optimal linear classifier fitted on all available channels to subtract linear crosstalk.
- *1D CNN:* A three-layer CNN on the raw records with no global pooling, retaining the temporal position of mid-readout jumps.

## Results

### Single qubit: $T_1$ decay
- The integrated threshold is the weakest baseline. The matched filter improves upon it but remains suboptimal because the excited class contains a mixture of random decay times.
- Both the full-record LDA and the CNN outperform the matched filter, with the CNN maintaining a slight edge (a few tenths of a point) at all $T_1$ values.

### Two qubits: Crosstalk
- **Cross-dispersive shift ($\zeta$):** Because this shift is nonlinear, linear filters cannot fully invert it. The CNN degrades the least, maintaining a clear advantage over joint LDA at high shift values.
- **Linear leakage ($\eta$):** Independent methods fail rapidly. Joint LDA successfully untangles most of the leakage. The CNN remains almost flat, outperforming joint LDA slightly, likely because it tracks the temporal dependence of the leakage (e.g., when the neighboring qubit decays).

## Limitations

- The simulation is a semi-classical model with white Gaussian noise; real devices exhibit more complex noise profiles and measurement back-action.
- Realistic hardware crosstalk values are generally smaller than the extremes ($\zeta = 3$-4, $\eta = 0.3$-0.5) where the neural network shows its largest advantages.
- The CNN hyperparameters are not rigorously tuned.

## Development Methodology

The core CNN architecture, QuTiP simulation boilerplate, and classical baselines were scaffolded with the assistance of AI coding tools. Primary technical contributions focus on structuring the semi-classical readout physics, designing the simulation to isolate spatial multi-qubit crosstalk and temporal $T_1$ decay, optimizing hardware utilization for Apple Silicon (MPS), and benchmarking the joint neural architecture against standard independent filters and joint linear discriminants.

## Project structure


```

sim/engine.py, sim/engine_2q.py    QuTiP jump trajectories + cavity response + crosstalk
model/readout_model.py             Configurable ReadoutCNN and ReadoutGRU
data/generate_data.py              Dataset generation
train/arguments.py                 Command-line arguments
train/train.py                     Training, validation, and baseline evaluation
baselines_2.py                     Independent vs. joint classical classifiers
plot_records.py                    Raw records and mean response visualizations
sweep.py                           T1, cross-dispersive, and leakage sweep scripts
results/figures, results/sweeps    Generated plots and sweep data

```

## Quick start

```bash
pip install -r requirements.txt

python -m sim.engine                                   # simulator sanity checks (1 qubit)
python -m sim.engine_2q                                # simulator sanity checks (2 qubits)
python -m data.generate_data --n-qubits 1 --sigma 1.0  # simulate a 1-qubit dataset
python -m train.train --epochs 20 --arch cnn           # train the CNN and evaluate the baselines

```

Two-qubit data and sweeps:

```bash
python -m data.generate_data --n-qubits 2 --sigma 1.0 --zeta 2.0 --leak 0.2 --n-train 5000
python sweep.py --sweep t1       
python sweep.py --sweep zeta     
python sweep.py --sweep leak     
python sweep.py --sweep zeta --replot   

```

## References

* **QuTiP** for the quantum-jump simulation.
> J. R. Johansson, P. D. Nation, and F. Nori, "QuTiP 2...", Comput. Phys. Commun. **184**, 1234 (2013).
> N. Lambert et al., "QuTiP 5...", arXiv:2412.04705 (2024).
