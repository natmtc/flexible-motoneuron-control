# Insights on motoneuron flexible control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia](https://img.shields.io/badge/Julia-1.8+-blue.svg)](https://julialang.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

> [!IMPORTANT]
> ## Code and data are withheld until publication
>
> This repository currently publishes **the notebooks only**. They document the complete
> pipeline — every parameter, every analysis step, and the reasoning behind each choice —
> and are the intended public record of the work.
>
> Two directories are deliberately **not** released yet:
>
> | Directory | Contents | Status |
> |---|---|---|
> | `utils/` | the Julia and Python function library that the notebooks import | **withheld until publication** |
> | `Data_MN_Model/` | model inputs, simulation outputs, and the experimental recordings | **withheld until publication** |
>
> **A fresh clone therefore cannot be run end to end.** The notebooks will raise
> `ModuleNotFoundError` on `utils...` imports and `FileNotFoundError` on paths under
> `Data_MN_Model/`. This is expected, not a bug — the notebooks are published for review
> and transparency, not yet for execution.
>
> Both directories will be added to this repository once the accompanying paper is
> published. If you need earlier access — for review, replication, or collaboration —
> please get in touch at n.torres-consul21@imperial.ac.uk.

## Overview

This repository contains a biophysical model investigating the fundamental limits of flexible motoneuron control. Using a two-compartment Hodgkin-Huxley framework, we demonstrate that limited flexibility emerges through neuromodulation of derecruitment dynamics, establishing an upper bound on controllability.

**Key Finding**: Truly independent motoneuron control would require breaking Henneman's size principle by violating recruitment order. Unless a muscle has different compartments or functional subdivisions receiving distinct inputs, it may be impossible to achieve truly independent control. What happens in the most constrained scenario, when a muscle only receives one input? We show that transient decorrelation can still emerge through dendritic calcium modulation during derecruitment, creating transient flexibility without violating the size principle.

## Table of Contents

- [Background](#background)
- [Model Architecture](#model-architecture)
- [Key Results](#key-results)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Firing-rate pipeline](#firing-rate-pipeline)
- [Citation](#citation)

## Background

Motoneurons are the final pathway from the central nervous system to muscle fibers. Traditionally, they were thought to fire in lockstep according to Henneman's size principle: smaller motoneurons recruit first and fire at higher rates.

Recent experiments ([Bräcklein et al., 2022](https://elifesciences.org/articles/72871)) showed humans can decorrelate motoneuron pairs in cursor-control tasks. Participants controlled motoneurons in the Tibialis Anterior, constrained to function as a unifunctional muscle receiving a single common input. This raised a fundamental question: can motoneurons be independently controlled when they share the same input?

**Research Questions**: Can heterogeneous neuromodulation from the brainstem enable transient independent control? What are the mechanistic limits? Does learning reflect true independent volitional access or strategic exploitation of derecruitment dynamics?

## Model Architecture

<p align="center">
  <img src="assets/diagram.jpg" width="500" alt="Two-compartment motoneuron model schematic">
  <br>
  <em>Figure 1a: Two-compartment biophysical motoneuron model</em>
</p>

Our model extends Hodgkin-Huxley dynamics with:

**Two-Compartment Structure**: Somatic (spike generation) and dendritic (neuromodulatory integration) compartments

**Heterogeneous Neuromodulation**: Brainstem monoaminergic input with variable gain regulating dendritic calcium channels

**Dendritic Calcium Dynamics**: Persistent inward currents (PICs) with hysteresis and black-box optimized deactivation thresholds. The model includes facilitation (increased firing rate after repeated contractions)

**Uneven Inhibition**: Differential inhibition across the motoneuron pool

The model was validated using HDsEMG recordings from Tibialis Anterior of 8 participants (129 ± 44 motoneurons per subject; [Avrillon et al., 2024](https://elifesciences.org/reviewed-preprints/97085v1)).

<p align="center">
  <img src="assets/firings.png" width="350" alt="Firing rate distributions">
  <br>
  <em>Figure 2: Model reproduces experimental firing-rate distributions</em>
</p>

## Key Results

### Force-Tracking Performance

A PD controller minimized tracking error across trapezoidal, sinusoidal, and chirp force targets. The model reproduced both mean force trajectories and physiological variability observed in humans.

### Online-Cursor Control Experiments

<p align="center">
  <img src="assets/online_cursor_control.jpg" width="600" alt="Motoneuron pair flexible control">
  <br>
  <em>Figure 3: (a) PD controller for flexible control; (b) Flexibility emerges from derecruitment modulation</em>
</p>

We simulated cursor-control experiments from [Bräcklein et al. (2022)](https://elifesciences.org/articles/72871). True independent recruitment is not achievable; the size principle is never violated during recruitment. Apparent flexibility arises from modulating derecruitment dynamics by selectively adjusting inhibition and neuromodulation, creating transient decorrelation that appears like independent control.

### Learning Dynamics

<p align="center">
  <img src="assets/glm.png" width="700" alt="Correlation evolution over learning">
  <br>
  <em>Figure 4: (a) 14-day learning dataset; (b) Pairwise correlations tracked with Poisson GLMs</em>
</p>

Analysis of the 14-day dataset from [Bräcklein et al. (2022)](https://elifesciences.org/articles/72871) using Poisson GLMs showed that improvements arise from behavioral optimization of existing neuromodulatory control, not fundamental changes in motoneuron properties.

**Conclusion**: Neuromodulation defines the upper bound of controllability in unifunctional muscles, enabling limited transient decorrelation but not arbitrary independent control.

## Features

- Two-compartment Hodgkin-Huxley motoneuron model with heterogeneous neuromodulation
- Black-box optimization of dendritic Ca²⁺ thresholds
- PD controllers for force-tracking and flexible motoneuron-pair control
- Motoneuron pool simulation (spike trains, force output, hysteresis)
- Correlation analysis over learning (Poisson GLMs)
- Analysis pipeline for experimental datasets

## Installation

### Clone Repository

```bash
git clone https://github.com/natmtc/Motoneuron_Model.git
cd Motoneuron_Model
```

### Julia Dependencies

Launch Julia and run:

```julia
using Pkg
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

## Quick Start

> **Note** — the steps below describe how the pipeline is run with the full code and data
> in place. Until `utils/` and `Data_MN_Model/` are released (see the notice at the top),
> the notebooks can be read but not executed.

This repository is organized around a set of Jupyter notebooks that implement the full simulation and analysis pipelines.

To get started:
1. Navigate to the `notebooks/` directory.
2. Open the main simulation notebook.
3. Follow the notebooks in order to reproduce the motoneuron simulations, pool-level analyses, and control tasks.

Each notebook documents the relevant assumptions, parameters, and outputs, and states which
functions in `utils/` it depends on.

### Firing-rate pipeline

`notebooks/check_firing_rates_online_control.ipynb` is the entry point for comparing experimental
and simulated firing-rate distributions. It covers force scaling, the simulation
sweep, and the comparison as one documented pipeline, and explains the reasoning
behind each step.

Point `MN_DATA_ROOT` at the directory holding the per-subject folders of `.pkl`
recordings, then launch from the repository root:

```bash
export MN_DATA_ROOT=/path/to/DATA
jupyter lab notebooks/check_firing_rates_online_control.ipynb
```

The notebook writes scaled force traces and a manifest under
`Data_MN_Model/Data_final/`. Section 3 runs in Julia via
`utils/functions_run_simulations/run_from_manifest.jl`; the remaining sections are
Python. Amplitude decisions are recorded in the manifest, and each simulation output
stores its own sampling rate, so the two languages share one explicit contract.

## License

This project is licensed under the MIT License.

## Contact

For questions: n.torres-consul21@imperial.ac.uk

## Glossary

**Motoneuron**: Spinal neuron controlling muscle fiber contraction  
**Size Principle**: Orderly recruitment of motoneurons from smallest to largest  
**Derecruitment**: Cessation of motoneuron firing as drive decreases  
**Persistent Inward Current (PIC)**: Dendritic calcium/sodium current amplifying synaptic input  
**Neuromodulatory Hysteresis**: Difference between activation and deactivation thresholds for PICs