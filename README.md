# What are the constraints of flexible motoneuron control?

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia](https://img.shields.io/badge/Julia-1.8+-blue.svg)](https://julialang.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

## Overview

This repository contains a biophysical model and analysis pipeline investigating the fundamental limits of flexible motoneuron control. Using a two-compartment Hodgkin-Huxley framework with heterogeneous neuromodulatory inputs, we demonstrate that while motoneurons cannot be independently controlled at will, limited flexibility emerges through brainstem-mediated neuromodulation of derecruitment dynamics.

**Key Finding**: Flexible motoneuron control respects Henneman's size principle but can create transient decorrelation through neuromodulatory modulation of dendritic calcium dynamics during derecruitment—establishing an upper bound on controllability in muscles.

---

## Table of Contents

- [Background](#background)
- [Model Architecture](#model-architecture)
- [Key Results](#key-results)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Availability](#data-availability)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)

---

## Background

### The Motoneuron Control Problem

Motoneurons (motor neurons) are the final common pathway from the central nervous system to muscle fibers. Traditionally, motoneurons within a single muscle pool were thought to fire in lockstep according to Henneman's **size principle**: smaller motoneurons are recruited first and fire at higher rates, creating rigid, correlated activity patterns.

Recent experiments ([Bräcklein et al., 2022](https://doi.org/10.1016/j.neuron.2022.03.011)) challenged this view by training humans to decorrelate pairs of motoneurons in cursor-control tasks, suggesting greater flexibility than previously assumed. However, the physiological mechanisms enabling this flexibility—and their limits—remained unclear.

### Research Questions

1. Can heterogeneous neuromodulation and inhibition explain observed decorrelation patterns?
2. What are the mechanistic limits of flexible motoneuron control?
3. Does learning to control motoneuron pairs reflect true independent access or strategic exploitation of existing physiological dynamics?

---

## Model Architecture

<p align="center">
  <img src="assets/diagram.jpg" width="500" alt="Two-compartment motoneuron model schematic">
  <br>
  <em>Figure 1a: Two-compartment biophysical motoneuron model with somatic and dendritic compartments</em>
</p>

### Core Components

Our model extends classical Hodgkin-Huxley dynamics with:

1. **Two-Compartment Structure**
   - Somatic compartment: spike generation
   - Dendritic compartment: integration of neuromodulatory input

2. **Heterogeneous Neuromodulatory Input**
   - Brainstem-originating monoaminergic modulation (serotonin, norepinephrine)
   - Variable gain across motoneuron pool
   - Regulates dendritic calcium channel activation

3. **Dendritic Calcium Dynamics**
   - Persistent inward currents (PICs)
   - Hysteresis in activation/deactivation thresholds
   - Black-box optimized deactivation thresholds (V<sub>th</sub><sup>Ca</sup>) per motoneuron

4. **Uneven Inhibitory Input**
   - Differential inhibition across motoneuron pool
   - Enables asymmetric derecruitment patterns

### Validation Against Human Data

The model was validated using high-density surface electromyography (HDsEMG) recordings from the Tibialis Anterior muscle of 8 participants (129 ± 44 motoneurons tracked per subject; [Avrillon et al., 2024](https://doi.org/10.7554/eLife.90239)).

<p align="center">
  <img src="assets/firings.png" width="350" alt="Firing rate distributions across motoneuron pool">
  <br>
  <em>Figure 2: Model accurately reproduces experimental firing-rate distributions</em>
</p>

The model successfully captures:
- Mean firing rate profiles across force levels
- Trial-to-trial variability in spike timing
- Collective force output dynamics
- Hysteresis in recruitment/derecruitment

---

## Key Results

### 1. Force-Tracking Performance

We evaluated model performance across three force-tracking paradigms:
- **Trapezoidal**: Ramp-and-hold contractions
- **Sinusoidal**: Rhythmic force modulation
- **Chirp**: Frequency-swept force tracking

A Proportional-Derivative (PD) controller minimized tracking error by adjusting synaptic drive. The model reproduced both mean force trajectories and physiological variability observed in human experiments.

### 2. Flexible Control Experiments

<p align="center">
  <img src="assets/flexible_exp.png" width="600" alt="Motoneuron pair flexible control results">
  <br>
  <em>Figure 3: (a) PD controller for flexible motoneuron pair control; (b) Apparent independence emerges from derecruitment modulation</em>
</p>

We simulated the cursor-control experiments from [Bräcklein et al. (2022)](https://doi.org/10.1016/j.neuron.2022.03.011), where participants learned to control motoneuron pairs along two axes.

**Key Findings**:
- True independent recruitment is **not achievable**—the size principle is never violated during recruitment
- Apparent flexibility arises from **modulating derecruitment dynamics**
- By selectively adjusting inhibition and neuromodulation, the nervous system can delay derecruitment of one motoneuron relative to another
- This creates transient decorrelation that *appears* like independent control but respects fundamental physiological constraints

### 3. Learning Dynamics Analysis

<p align="center">
  <img src="assets/glm.png" width="700" alt="Correlation analysis across 14-day learning period">
  <br>
  <em>Figure 4: (a) 14-day learning dataset; (b) Pairwise correlation evolution tracked with Poisson GLMs</em>
</p>

We analyzed the complete 14-day dataset from [Bräcklein et al. (2022)](https://doi.org/10.1016/j.neuron.2022.03.011) using spiking Poisson Generalized Linear Models (GLMs) to quantify:
- Evolution of pairwise motoneuron correlations
- Changes in coupling strength over learning
- Whether improvements reflect physiological changes vs. behavioral adaptation

**Results**: Improvements in task performance arise primarily from **behavioral optimization of existing neuromodulatory control**, not from fundamental changes in motoneuron connectivity or properties.

### Main Conclusion

**Neuromodulation defines the upper bound of controllability** in unifunctional muscles. While the nervous system cannot arbitrarily access individual motoneurons, it can exploit dendritic neuromodulation to reshape derecruitment thresholds, enabling limited, transient decorrelation. This flexibility is sufficient for fine motor control but fundamentally constrained by the size principle during recruitment.

---

## Features

### Model Capabilities
- Two-compartment biophysical motoneuron model (Hodgkin-Huxley based)
- Heterogeneous neuromodulatory input profiles across motoneuron pool
- Dendritic persistent inward currents (PICs) with hysteresis
- Black-box optimization of calcium channel deactivation thresholds (V<sub>th</sub><sup>Ca</sup>)
- Uneven inhibition distribution for asymmetric derecruitment

### Simulation & Control
- Accurate motoneuron pool simulation (spike trains, force output)
- PD controllers for:
  - Force-tracking tasks (trapezoid, sinusoid, chirp)
  - Motoneuron-pair flexible control tasks
- Simulation of derecruitment dynamics and firing-rate variability

### Analysis Tools
- Motoneuron correlation analysis over learning (Poisson GLMs)
- Pairwise coupling strength quantification
- Recruitment/derecruitment threshold extraction
- Force reconstruction from spike trains
- Statistical comparison with experimental data

### Data Processing
- Analysis pipeline for [Bräcklein et al. (2022)](https://doi.org/10.1016/j.neuron.2022.03.011) experimental dataset
- HDsEMG spike train processing
- Time-series alignment and filtering

---

## Installation

### Prerequisites

**For Model Simulation (Julia)**:
- Julia 1.8 or later ([download here](https://julialang.org/downloads/))

**For Data Analysis (Python)**:
- Python 3.8 or later

### Clone Repository

```bash
git clone https://github.com/natmtc/Motoneuron_Model.git
cd Motoneuron_Model
```

### Julia Dependencies

Launch Julia in the repository directory and run:

```julia
using Pkg
Pkg.activate(".")

Pkg.add([
    "Plots", "Distributions", "JLD", "DSP", "Parameters", 
    "Match", "SymDict", "NumericalIntegration", "Random", 
    "MAT", "CSV", "DataFrames", "XLSX", "JLD2", 
    "Colors", "ColorSchemes", "Makie", "CairoMakie", 
    "Optim", "Serialization", "JuMP", "Ipopt", 
    "Wandb", "PyCall", "Logging", "GLM", "PyPlot", "Polynomials"
])
```

### Python Dependencies

```bash
pip install numpy pandas seaborn matplotlib scipy scikit-learn umap-learn h5py
```

Or use the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Example 1: Simulate a Single Motoneuron

```julia
using Motoneuron_Model

# Initialize motoneuron with default parameters
mn = Motoneuron(
    Rin = 1.5,              # Input resistance (MΩ)
    rheobase = 10.0,        # Rheobase current (nA)
    VthCa = -45.0          # Ca²⁺ activation threshold (mV)
)

# Apply synaptic input (trapezoid profile)
time = 0:0.1:5000       # Time vector (ms)
input = create_trapezoid_input(time, ramp_duration=1000, hold_duration=2000)

# Run simulation
voltage, spikes = simulate_motoneuron(mn, input, time)

# Plot results
plot_motoneuron_response(time, voltage, spikes)
```

### Example 2: Simulate Motoneuron Pool

```julia
# Create pool of 120 motoneurons with heterogeneous properties
pool = create_motoneuron_pool(n=120, heterogeneity=0.3)

# Simulate force-tracking task
force_target = sinusoidal_target(frequency=0.5, amplitude=30, duration=10)
spikes, force_output = simulate_pool_tracking(pool, force_target)

# Analyze results
plot_raster_and_force(spikes, force_output, force_target)
```

### Example 3: Flexible Control Task

```julia
# Load motoneuron pair for cursor control
mn1, mn2 = load_motoneuron_pair("data/pair_01.jld2")

# Run 2D cursor control task
controller = PDController(Kp=0.5, Kd=0.1)
trajectory = run_flexible_control(mn1, mn2, controller, n_trials=50)

# Visualize learning
plot_cursor_trajectories(trajectory)
```

---

## Data Availability

### Experimental Datasets

1. **[Avrillon et al., 2024](https://doi.org/10.7554/eLife.90239)** (eLife)
   - HDsEMG recordings from Tibialis Anterior
   - 8 participants, 129 ± 44 motoneurons per participant
   - Force-tracking tasks (trapezoid, sinusoid, chirp)
   - Available at: [Dryad Digital Repository](https://doi.org/10.5061/dryad.xxxxx)

2. **[Bräcklein et al., 2022](https://doi.org/10.1016/j.neuron.2022.03.011)** (Neuron)
   - 14-day learning dataset
   - 2D cursor-control with motoneuron pairs
   - Available upon request from original authors

### Simulated Data

All model outputs and analysis results are available in the `results/` directory after running simulations.

---

## Citation

If you use this code or build upon this work, please cite:

```bibtex
@article{yourname2024motoneuron,
  title={Limits of Flexible Motoneuron Control in Unifunctional Muscles},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2024},
  doi={10.xxxx/xxxxx}
}
```

**Related Publications**:

```bibtex
@article{avrillon2024,
  title={Decoding the neural drive to muscles from the surface electromyogram},
  author={Avrillon, Simon and Hug, François and Farina, Dario},
  journal={eLife},
  volume={13},
  year={2024},
  doi={10.7554/eLife.90239}
}

@article{bracklein2022,
  title={Reading and modulating cortical beta bursts from motor unit spiking activity},
  author={Br{\"a}cklein, Martin and Ibáñez, Jaime and Barsakcioglu, Deren Y and Farina, Dario},
  journal={Neuron},
  volume={110},
  number={10},
  pages={1688--1701},
  year={2022},
  doi={10.1016/j.neuron.2022.03.011}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

For major changes, please open an issue first to discuss proposed modifications.

### Development Guidelines

- Follow Julia style conventions ([Blue Style](https://github.com/invenia/BlueStyle))
- Include tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

---

## Acknowledgments

This work was supported by:
- [Funding Agency 1] (Grant #XXXXX)
- [Funding Agency 2] (Grant #YYYYY)

We thank the authors of [Bräcklein et al. (2022)](https://doi.org/10.1016/j.neuron.2022.03.011) and [Avrillon et al. (2024)](https://doi.org/10.7554/eLife.90239) for making their data available.

---

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: [your.email@institution.edu]

---

## Glossary

**Motoneuron**: A neuron whose cell body is located in the spinal cord and whose axon projects to muscle fibers, controlling muscle contraction.

**Unifunctional Muscle**: A muscle with a single primary mechanical function (e.g., ankle dorsiflexion for Tibialis Anterior).

**Size Principle (Henneman's Principle)**: The orderly recruitment of motoneurons from smallest to largest based on input strength.

**Derecruitment**: The process by which motoneurons cease firing as synaptic drive decreases.

**Persistent Inward Current (PIC)**: A voltage-dependent calcium or sodium current that amplifies synaptic input in motoneuron dendrites.

**Neuromodulatory Hysteresis**: The difference between activation and deactivation thresholds for dendritic PICs, enabling history-dependent excitability.

**HDsEMG**: High-Density Surface Electromyography—a technique using electrode arrays to record muscle electrical activity and decompose individual motor unit spike trains.