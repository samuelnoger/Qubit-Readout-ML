# Qubit Readout Classification with Neural Networks

A small, self-contained project on **dispersive qubit readout**: deciding whether a superconducting qubit was prepared in $\vert{}0\rangle$ or $\vert{}1\rangle$ from a noisy microwave measurement record. Records are simulated with QuTiP, and a 1D CNN is compared against the standard classical readout methods and against stronger linear and nonlinear classical baselines.

This is a learning project, not a research contribution. It reproduces a known idea (neural readout classifiers can help when the qubit relaxes during measurement) on a simplified simulator, then looks at what happens when two multiplexed qubits interfere with each other.

<img src="results/figures/sweep_t1.png" alt="Fidelity vs T1" width="400"><img src="results/figures/sweep_zeta.png" alt="Fidelity vs cross-dispersive shift" width="400"><img src="results/figures/sweep_leak.png" alt="Fidelity vs signal leakage" width="400">

---

## The problem

In circuit QED, a qubit is read out through a microwave cavity. The qubit state shifts the cavity response, so a probe tone returns a different complex amplitude $\alpha = I + iQ$ for $\vert{}0\rangle$ and $\vert{}1\rangle$. The signal is small compared to amplifier noise, so the record has to be combined over time to decide. Two things make this harder:

1. **$T_1$ decay.** A qubit prepared in $\vert{}1\rangle$ can relax during the readout window, so the record switches from "excited-like" to "ground-like" partway through. A plain integrator or a fixed-weight filter cannot use *when* the switch happens.
2. **Crosstalk.** With several qubits read out through shared hardware, one qubit's signal leaks into another's channel, and one qubit's state can shift another's resonator.

The metric is **assignment fidelity**, $1 - \frac{P(0\vert{}1) + P(1\vert{}0)}{2}$, where 0.5 is guessing and 1.0 is perfect. For two qubits it is computed per qubit and averaged.

## Simulation model

Simulation happens in two stages (units: µs):

1. **Qubit trajectories.** QuTiP `mcsolve` runs quantum-jump trajectories with a $T_1$ collapse operator. Each trajectory is excited until a random jump time and ground afterwards. Ground-state preparations never jump.
2. **Cavity response.** Given the qubit trajectory $s(t) = \pm 1$, the cavity field obeys

$$\frac{d\alpha}{dt}=-i\varepsilon-\left(\frac{\kappa}{2}+i\chi s(t)\right)\alpha$$

which is solved exactly on each time step because $s(t)$ is piecewise constant. Gaussian noise of standard deviation $\sigma$ is added to each $(I, Q)$ sample.

**Two qubits.** The qubits decay independently. Each resonator $i \in \{1, 2\}$ obeys

$$\frac{d\alpha_i}{dt}=-i\varepsilon-\left(\frac{\kappa}{2}+i\left(\chi s_i(t)+\zeta s_j(t)\right)\right)\alpha_i,\quad j\neq i$$

and the measured channels mix linearly,

$$m_i=\alpha_i+\eta\alpha_j+\text{noise}$$

- $\zeta$ (`--zeta`) is a **cross-dispersive shift**: the other qubit's state changes this resonator's response. It acts nonlinearly on the records.
- $\eta$ (`--leak`) is **linear signal leakage** between channels. A classifier that sees both channels can in principle subtract it.

A static $J\sigma_{z,1}\sigma_{z,2}$ coupling is left out on purpose. It is diagonal in the computational basis, and since qubits are only prepared in basis states, it would not change the records.

This is a semi-classical readout model. It does **not** solve the full stochastic master equation for the coupled qubit-cavity system.

Defaults: $T_1 = 3$, readout window 2, dt 0.02 (101 samples), $\kappa = 10$, $\chi = 5$, $\varepsilon = 5$, $\sigma = 3$ (the sweeps use $\sigma = 1$). For two qubits: $\zeta = 1$, $\eta = 0.1$.

## Classifiers

- **Integrated threshold:** sum the record over time and apply a threshold. Sees only its own qubit's channels.
- **Matched filter:** weight each time step by the difference of the mean excited and mean ground records. Own channels only.
- **Linear discriminant (LDA):** the best linear classifier on the full record. Fitted on the own channels ("own") or on all channels ("all").
- **Gradient boosting:** a nonlinear classical model on the flattened record.
- **1D CNN:** three conv layers and a dense head on the raw record, with one output per qubit. No global pooling, so the time position of a switch is kept. A small GRU is available as an alternative (`--arch gru`).

Thresholds are chosen on training data, the best epoch is chosen on a validation set, and all reported numbers come from an independent test set (separate random seeds for each split).

## Results

All sweeps use $\sigma = 1$ and 3 seeds. The two-qubit sweeps change one crosstalk mechanism at a time: the $\zeta$ sweep has $\eta = 0$, and the $\eta$ sweep has $\zeta = 0$. Exact numbers are in `results/sweeps/`.

### Single qubit: $T_1$ decay

Fidelity rises with $T_1$ for every method, and the curves converge once decay is rare (all near 0.98 at $T_1 = 15$ µs). The integrated threshold is the weakest, falling to about 0.72 at $T_1 = 0.5$ µs. The matched filter is about 3 points behind the best methods when $T_1$ is comparable to the readout window.

The more informative comparison is with LDA. The matched filter's kernel is not the optimal *linear* filter here, because the excited class is a mixture of decay times, and an LDA on the full record does better. It lands very close to the CNN at every $T_1$. The CNN is ahead of LDA at all $T_1$ values, but only by a few tenths of a point. So most of the apparent gain of the network over the matched filter is better linear weighting, and the genuinely nonlinear part is small.

### Two qubits: cross-dispersive shift $\zeta$

Every method degrades as $\zeta$ grows, because the information in the records really shrinks: when $\zeta$ is comparable to $\chi$, a resonator cannot identify its own qubit without knowing the other. The independent methods degrade most, joint LDA less, and the CNN least. At the strongest shift tested ($\zeta = 4$, with $\chi = 5$), the CNN stays near 0.92, joint LDA falls to about 0.89, and the independent filters to 0.83-0.87. Because the shift is nonlinear, a linear filter cannot undo it fully, and this is where the network has a clear advantage.

### Two qubits: linear leakage $\eta$

The independent methods lose 4 to 6 points by $\eta = 0.5$. Joint LDA recovers most of that (about 1 point lost), and the CNN stays almost flat, ending around 1 point above joint LDA. A plausible explanation, which I have not tested, is that the other qubit's contribution depends on when it decays, which a fixed linear subtraction cannot follow.

### Takeaways

- Classifiers that see only their own channel lose fidelity under both kinds of crosstalk. A joint linear filter is the right classical baseline, not the matched filter.
- Without crosstalk the CNN is only slightly better than the best linear baseline. Its advantage grows with crosstalk, especially the nonlinear cross-dispersive shift.
- The crosstalk values where the network clearly wins ($\zeta = 3$-4, $\eta = 0.3$-0.5) are much larger than typical hardware. At small, realistic values, all joint methods are within about half a point of each other.

## Limitations

- Everything is simulated from a model I wrote. The results show what the network can learn in this setup, not how it would do on a real device.
- The cavity model is linear and semi-classical, and the noise is white and Gaussian.
- Qubits are prepared in computational basis states only, so there are no superpositions, entanglement, or measurement back-action.
- Gradient boosting is a weak nonlinear baseline here. A tuned MLP would be fairer.
- Hyperparameters are not tuned. The CNN uses fixed settings for 20-30 epochs, so its numbers are likely lower bounds.
- Sweeps use 3 seeds. Gaps below roughly half a point should not be over-interpreted.
- There is no Bayes-optimal reference yet, so it is unknown how close any classifier is to the best possible one.

## Development Methodology

The core CNN architecture, QuTiP simulation boilerplate, and classical baselines were scaffolded with the assistance of AI coding tools. Primary technical contributions focus on structuring the semi-classical readout physics, designing the simulation to isolate spatial multi-qubit crosstalk and temporal $T_1$ decay, optimizing hardware utilization for Apple Silicon (MPS), and benchmarking the joint neural architecture against standard independent filters and joint linear discriminants.

## Project structure


```

sim/engine.py              QuTiP jump trajectories + cavity response + noise (1 qubit)
sim/engine_2q.py           Two-qubit readout with cross-dispersive shift and linear leakage
model/readout_model.py     Configurable ReadoutCNN and ReadoutGRU (one logit per qubit)
data/generate_data.py      Dataset generation (train/val/test, independent seeds)
train/arguments.py         All command-line arguments (physics, data, training)
train/train.py             Training, validation, and comparison with the baselines
baselines_2.py             Independent vs. joint classical classifiers, assignment fidelity
plot_records.py            Raw records, class-mean records, matched-filter histograms
sweep.py                   T1, cross-dispersive and leakage sweeps with several seeds
generate.sh / run_train.sh Convenience scripts with the default settings
results/figures, results/sweeps   generated plots and sweep data

```

## Quick start

```bash
pip install qutip scikit-learn torch numpy matplotlib tqdm     # QuTiP >= 5

python -m sim.engine                                   # simulator sanity checks (1 qubit)
python -m sim.engine_2q                                # simulator sanity checks (2 qubits)
python -m data.generate_data --n-qubits 1 --sigma 1.0  # simulate a 1-qubit dataset
python -m train.train --epochs 20 --arch cnn           # train the CNN and evaluate the baselines

```

Two-qubit data and the sweeps:

```bash
python -m data.generate_data --n-qubits 2 --sigma 1.0 --zeta 2.0 --leak 0.2 --n-train 5000
python sweep.py --sweep t1       # single qubit, vary T1
python sweep.py --sweep zeta     # two qubits, vary the cross-dispersive shift
python sweep.py --sweep leak     # two qubits, vary the linear leakage
python sweep.py --sweep zeta --replot   # redraw from saved results without rerunning

```

Training uses Apple Silicon (MPS) when `--device mps` is set and falls back to CPU if it is unavailable.

## Sanity checks built into the simulators

`python -m sim.engine` verifies both stages independently:

1. The mean of many excited-qubit trajectories matches $\exp(-t/T_1)$ up to shot noise.
2. The noise-free cavity field converges to the analytic steady state $\frac{-i\varepsilon}{\kappa/2 + i\chi}$.

`python -m sim.engine_2q` also checks that with $\zeta = 0$ resonator 1 reproduces the single-qubit response, that the two qubits decay independently, and that the steady states with $\zeta \neq 0$ match the analytic values.

## Possible extensions

* Bayes-optimal classifier (marginalizing over the decay time) as a performance ceiling.
* An MLP baseline, a sweep with both $\zeta$ and $\eta$ nonzero, and a noise sweep.
* Non-Gaussian noise, where linear filters tuned for white noise should struggle.
* Superposition and entangled input states, which turn readout into state tomography.

## Earlier version of this repository

The project began as a parameter-fitting and pulse-control pipeline for a driven Jaynes-Cummings system using a neural surrogate. It turned out that a small, well-modeled system gives a neural network little to do beyond what a classical solver already does, so I moved to a problem where the learned model addresses something a linear method cannot. The earlier code is kept in `old_jc_control/`.

## References

* **PyTorch** and **Scikit-Learn** for the neural networks, baselines, and training loops.
* **QuTiP** for the quantum-jump simulation. If you build on the QuTiP parts of this project, please cite:
> J. R. Johansson, P. D. Nation, and F. Nori, "QuTiP 2: A Python framework for the dynamics of open quantum systems," Comput. Phys. Commun. **184**, 1234 (2013).
> N. Lambert et al., "QuTiP 5: The Quantum Toolbox in Python," arXiv:2412.04705 (2024).


* **NumPy & Matplotlib** for data handling and plotting.
