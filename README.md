# Constraints and flexibility in motoneuron control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Julia](https://img.shields.io/badge/Julia-1.8%2B-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)

**Can individual motoneurons be controlled flexibly despite receiving strongly shared synaptic input?**

This repository documents the computational modelling and experimental analyses underlying *Neuromodulation enables transient flexible control of motoneurons*. Using a biophysical motoneuron-pool model, human high-density surface EMG (HDsEMG), and longitudinal motor-unit neurofeedback data, we investigate how neuromodulation and inhibition shape the limits of independent motoneuron control.

**Main finding:** Heterogeneous neuromodulation can enable **transient decorrelation during derecruitment** without violating orderly recruitment. Improvements with training are consistent with learning to exploit existing control strategies rather than a fundamental reorganization of motoneuron population dynamics.

> **Code and data availability**
>
> The annotated notebooks are publicly available, but the supporting `utils/` functions and `Data_MN_Model/` inputs, outputs, and recordings are **withheld until publication**.
>
> A fresh clone cannot yet reproduce the complete analyses. The notebooks document the methods, parameters, and analysis workflow.
>
> For access inquiries, contact [n.torres-consul21@imperial.ac.uk](mailto:n.torres-consul21@imperial.ac.uk).

---

## 1. Biophysical model and experimental data

<p align="center">
  <img src="assets/diagram.jpg" width="600" alt="Motoneuron pool architecture and experimental force and HDsEMG data">
</p>

**Figure 1 — Model architecture and experimental data.**

- **(a) Biophysical model:** A pool of two-compartment Hodgkin–Huxley motoneurons receives shared excitatory and inhibitory inputs, with heterogeneous sensitivity to neuromodulation. Dendritic persistent inward currents (PICs) influence firing hysteresis, while somatic action potentials are converted into force through a twitch-based model.
- **(b) Experimental data:** Human tibialis anterior force-tracking contractions are recorded using HDsEMG and decomposed into motor-unit spike trains. Recruitment and derecruitment dynamics provide physiological constraints for model calibration.

The framework investigates how shared excitation, inhibition, and heterogeneous neuromodulation influence motoneuron recruitment, firing, and flexibility.

Experimental recordings from eight participants provide a physiological reference ([Avrillon et al., 2024](https://elifesciences.org/reviewed-preprints/97085v1)).

---

## 2. Physiological firing rates

<p align="center">
  <img src="assets/firings.png" width="325" alt="Simulated and experimental firing-rate distributions">
</p>

**Figure 2 — Comparison of simulated and experimental firing rates.**

Simulated mean firing-rate distributions are compared with experimental recordings across contraction intensities ranging from 10% to 80% maximum voluntary contraction (MVC).

Simulated firing rates are concentrated around 14–17 Hz, within the approximately 9–18 Hz range highlighted in the experimental recordings.

This comparison supports the physiological plausibility of the simulated firing rates, rather than an exact reproduction of every experimental condition.

---

## 3. Constraints on flexible online control

<p align="center">
  <img src="assets/online_cursor_control.jpg" width="850" alt="Motoneuron-pair flexibility and online cursor-control simulations">
</p>

**Figure 3 — Mechanisms and limits of motoneuron-pair flexibility.**

- **(a) Control hypotheses:** Three possible regimes of motoneuron-pair control are considered: sustained flexibility, transient flexibility, and rigid coupling. Different excitatory inputs, heterogeneous neuromodulation, and shared neuromodulation represent candidate underlying mechanisms.

- **(b) Online control simulation:** A proportional–derivative (PD) controller drives the biophysical model to reproduce target cursor trajectories inspired by motor-unit neurofeedback experiments ([Bräcklein et al., 2022](https://elifesciences.org/articles/72871)).

- **(c) Transient flexibility:** Heterogeneous neuromodulation and inhibition enable transient separation of motoneuron firing rates, particularly during derecruitment.

- **(d) Constrained control:** Similar neuromodulatory effects maintain strongly coupled activity, limiting independent control of motoneuron pairs.

- **(e) Force production:** Simulated force is evaluated alongside experimental force across different phases of the online control task.

- **(f) Pair-level displacement:** The degree of firing-rate separation varies with motoneuron properties and neuromodulation.

- **(g) Population-level dispersion:** Firing-rate dispersion is compared with and without neuromodulation to evaluate its contribution to control flexibility.

**Key finding:** Transient decorrelation can emerge through heterogeneous neuromodulatory effects on derecruitment, without violating orderly recruitment under the size principle.

These results suggest that apparently flexible control need not imply fully independent recruitment of individual motoneurons.

---

## 4. Learning across days: behaviour and population coupling

<p align="center">
  <img src="assets/glm.png" width="780" alt="Behavioural learning and GLM analysis of motoneuron population coupling">
</p>

**Figure 4 — Behavioural improvements and stability of population coupling across learning.**

- **(a) Behavioural performance:** Participants improve their success rates over training, with representative force trajectories illustrating changes across sessions.

- **(b) Task performance:** Force jitter and trial duration decrease between early and late training sessions.

- **(c) Experimental population analysis:** A leave-one-motoneuron-out generalized linear model (GLM) predicts each motoneuron's spike counts from the activity of the remaining population. Cross-validated pseudo-R² measures population-based predictability across training days.

- **(d) Simulated population analysis:** The same GLM approach is applied to simulated motoneuron activity to provide a model-based comparison.

**Key finding:** Behavioural performance improves while GLM-based population predictability remains relatively stable across training.

These findings are consistent with participants refining their control strategies to exploit existing physiological flexibility, rather than requiring substantial reorganization of motoneuron population dynamics.

---

## Repository contents

| Directory | Description | Availability |
|---|---|---|
| `Notebooks/` | Documented simulation and analysis pipelines | Public |
| `assets/` | Figures illustrating the methods and results | Public |
| `utils/` | Supporting Julia and Python functions | Withheld until publication |
| `Data_MN_Model/` | Experimental recordings, model inputs, and simulation outputs | Withheld until publication |

## The notebooks

Each notebook is one stage of the work and documents its own assumptions, parameters, and
outputs. Julia notebooks run the biophysical model; Python notebooks analyse experimental
recordings and model output.

### Model calibration and characterisation — Julia

| Notebook | What it does |
|---|---|
| `optimization_vthca_values.ipynb` | Fits each motoneuron's calcium activation threshold (`VthCa`) so that the model reproduces the recruitment hysteresis measured in the corresponding motor unit. Produces the parameter file that every other simulation loads. |
| `simulate_motoneuron_pool.ipynb` | The core simulation. Builds a pool of two-compartment motoneurons, drives it with a slow triangular ramp, and measures recruitment, firing rates, and hysteresis. The starting point for understanding the model. |
| `hysteresis_and_vthca_sweep.ipynb` | Sweeps the calcium threshold against inhibition for a single motoneuron, mapping how the two jointly determine hysteresis. |

### Control tasks — Julia

| Notebook | What it does |
|---|---|
| `closed_loop_force_control.ipynb` | A proportional–derivative controller drives the pool to track target force trajectories (trapezoidal, sinusoidal, chirp). |
| `online_cursor_control_task.ipynb` | Simulates the cursor-control task: a controller attempts to decorrelate a motoneuron pair under shared input. Includes an alternative parameterisation with uniform `VthCa` and randomised inhibition, to separate the contribution of heterogeneous neuromodulation. |
| `simulate_all_trials.ipynb` | Runs the pool once per recorded trial, driven by that trial's scaled force, and computes firing rates per trial. Produces the per-trial simulated rasters used by the population analyses. |

### Experimental and comparative analysis — Python

| Notebook | What it does |
|---|---|
| `experiments_data_analysis.ipynb` | Behaviour of one participant across the training sessions: spike rasters, force traces, trial duration, success rate, force smoothness, and the two-unit state space. Also performs the force scaling that turns recorded force into model input. |
| `check_firing_rates_online_control.ipynb` | Compares simulated against recorded discharge. Covers the choice of force-scaling policy, the simulation sweep, and both distribution-level and time-domain comparisons. |
| `generalized_linear_model_analysis.ipynb` | Leave-one-motoneuron-out Poisson GLMs measuring how predictable each unit is from the rest of the population, and how that predictability changes across training. |
| `hysteresis_repeatability.ipynb` | Tracks the same motor unit across 10–80% MVC contractions using the decomposition's own filter-tracking method, and quantifies how repeatable each unit's hysteresis is between contractions. |

## How the pieces fit together

The stages depend on each other in this order:

1. **Calibrate.** `optimization_vthca_values.ipynb` fits `VthCa` per motor unit against measured hysteresis. Everything downstream loads the resulting parameter file.
2. **Characterise.** `simulate_motoneuron_pool.ipynb` and `hysteresis_and_vthca_sweep.ipynb` establish how the calibrated pool behaves — recruitment order, firing rates, and the hysteresis the calibration targeted.
3. **Scale the experimental force.** `experiments_data_analysis.ipynb` converts recorded force, in arbitrary units, into a drive whose amplitude is set deliberately against the pool's recruitment thresholds.
4. **Simulate.** `simulate_all_trials.ipynb` drives the pool with that scaled force, one simulation per trial.
5. **Compare and analyse.** `check_firing_rates_online_control.ipynb` compares simulated against recorded discharge; `generalized_linear_model_analysis.ipynb` quantifies population coupling across learning.

The control-task notebooks (`closed_loop_force_control.ipynb`, `online_cursor_control_task.ipynb`) and `hysteresis_repeatability.ipynb` are independent of that chain: the first two need only the calibrated pool, and the third works directly from the experimental recordings.

A reader starting from scratch is best served by `simulate_motoneuron_pool.ipynb`, which introduces the model and its parameters, followed by `online_cursor_control_task.ipynb`, which contains the central result.

## Getting started

Clone the repository:

```bash
git clone https://github.com/natmtc/constraints-and-flexibility-motoneurons.git
cd constraints-and-flexibility-motoneurons
```

Launch JupyterLab **from the repository root**, since the notebooks resolve their paths relative to it:

```bash
jupyter lab
```

Two kernels are required: Julia 1.6 for the simulation notebooks and Python 3.11+ for the
analysis notebooks. Python dependencies are `numpy`, `pandas`, `scipy`, `matplotlib`,
`seaborn`, `statsmodels`, `h5py`, and `scikit-learn`.

**Note:** Complete execution currently requires the supporting functions and datasets that will be released alongside publication.

---

## References and citation

**Associated manuscript**

*Neuromodulation enables transient flexible control of motoneurons.*

Preprint available; manuscript under review.

**Experimental studies**

- [Bräcklein et al. (2022)](https://elifesciences.org/articles/72871) — Longitudinal motor-unit neurofeedback and learning experiments.
- [Avrillon et al. (2024)](https://elifesciences.org/reviewed-preprints/97085v1) — Human motor-unit recordings used for physiological calibration.

If you use this repository, please cite the associated manuscript and the relevant original experimental studies.

## License

This project is licensed under the MIT License.

## Contact

For questions, collaboration, or requests for early access to supporting materials:

**Natalia T. Cónsul**  
Imperial College London  
[n.torres-consul21@imperial.ac.uk](mailto:n.torres-consul21@imperial.ac.uk)
