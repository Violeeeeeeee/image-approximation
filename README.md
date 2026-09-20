# GPU-Accelerated Image Approximation

## Overview

This project is an implementation of an evolutionary algorithm designed to approximate target images using basic geometric shapes such as triangles. It was originally developed as a coursework project for a university course on Evolutionary Algorithms.

The primary motivation was to solve an interesting, visually engaging problem that did not require searching for and preprocessing mundane datasets. The inspiration for this approach came from an article on [rogeralsing.com](https://www.rogeralsing.com/2008/12/07/genetic-programming-evolution-of-mona-lisa/), where the author approximated the Mona Lisa using randomly generated triangles. Since the original source code was not provided, I implemented my own solution from scratch.

## Approach

This project treats image generation as an optimization problem. Instead of predicting pixel values, the algorithm optimizes a set of geometric parameters (coordinates, colors, opacity).

**Rendering & Optimization:** The core bottleneck in evolutionary image approximation is rendering thousands of candidate images per generation. To solve this, the project utilizes a custom JIT-compiled renderer (`renderer_jit.py`) written in PyTorch, which allows for blazingly fast rasterization directly on the GPU.
The evolutionary pipeline (`evo_pipeline.py`) manages the population, applying random mutations and evaluating fitness using Mean Squared Error (MSE) against the target image.

**Data Assumptions:** The pipeline expects standard image formats (JPG/PNG) placed in the `data/` folder. Images are automatically resized and normalized according to the configuration file before the evolutionary process begins.

## Project Structure

```text
image-approximation/
├── Makefile                # Quick commands for running and cleaning the project
├── main.py                 # Pipeline entry point (Loads config -> Runs evolution)
├── ax_trial.py             # Hyperparameter optimization/experimentation script
├── plot.py                 # Utility for generating fitness progression plots
├── modules/
│   ├── evo_pipeline.py     # Core evolutionary algorithm (Population, Mutation, Fitness)
│   ├── renderer_jit.py     # PyTorch JIT-compiled GPU rasterizer for shapes
│   └── __init__.py
├── configs/
│   ├── config.yaml         # Main configuration file
│   └── config2.yaml, ...   # Alternative experimental setups
├── report/                 # Academic coursework report detailing methodology
├── data/
│   ├── monalisa.jpg        # Target image examples
│   └── power.jpg
├── results/                # Output directory for generated images and best genes (.pt)
└── requirements.txt        # Python dependencies

```

## Features

* **Evolutionary Algorithm:** Utilizes mutation and selection mechanics to recreate images from scratch using basic geometric primitives.
* **GPU Acceleration:** Features a custom PyTorch-based renderer to drastically decrease rendering and fitness evaluation times.
* **Hyperparameter Tuning:** Integrates the Ax platform to systematically search for and optimize the algorithm's hyperparameters.
* **Dataset-Free:** Operates directly on any single target image provided by the user without the need for large training datasets.

## Tech Stack

* **Modeling & Evolution:** Python, PyTorch (Tensors & JIT Compilation)
* **Optimization Algorithm:** Evolutionary Algorithm (Mutation, Crossover, Selection)
* **Configuration:** YAML
* **Infrastructure:** Makefile

## Setup & Installation

### Prerequisites

* Python 3.12.6
* CUDA-compatible GPU (highly recommended for PyTorch JIT rendering)

### Option 1: Virtual Environment (Local Execution)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

```

### Option 2: Conda

```bash
conda create -n image-approx python=3.12.6
conda activate image-approx
pip install -r requirements.txt

```

## Usage (via Makefile)

We use a `Makefile` to simplify execution. You can run `make help` in your terminal to see a full list of commands.

**Run the main approximation pipeline:**

```bash
make run

```

**Run with a specific configuration file:**

```bash
make run CONFIG=configs/config2.yaml

```

**Clean up output artifacts (results & caches):**

```bash
make clean

```

## Configuration Guide (`configs/config.yaml`)

The behavior of the evolutionary pipeline is entirely controlled by YAML configuration files.

**Example Structure:**

```yaml
experiment_name: "PolyEvo_Torch_v1"
seed: 42
device: "cuda"

data:
  target_path: "data/monalisa_128.jpg"
  output_path: "mona/"
  resize_to: 128

algorithm:
  num_triangles: 100
  population_size: 1024
  generations: 20000

  mutation_rate: 0.0015
  mutation_scale: 30.0
  survival_rate: 0.28
```

## Results

The implementation successfully reconstructs target images, demonstrating the effectiveness of combining evolutionary algorithms with modern GPU acceleration. The project successfully met all academic requirements and received a top grade (5) for the coursework. To see images you can check results directory or read report/report.pdf.
