# TTGSM

## Overview
This repository contains a code supporting the paper "Logarithmically complex rigorous Fourier-space solution to the 1D grating diffraction problem" [1]. The method implemented in this code is the **Tensor-Train Accelerated Generalised Source Method (TTGSM)**.

[1] E. Levdik and A. A. Shcherbakov, "Logarithmically complex rigorous Fourier space solution to the 1D grating diffraction problem," Computer Physics Communications 310, 109530 (2025).

Developed by **Evgenii Levdik** and **Alexey A. Shcherbakov**.

For more details, refer to the following article: [link](https://arxiv.org/abs/2409.07821).

## Installation
This code requires [`ttpy`](https://github.com/oseledets/ttpy), a Python implementation of the Tensor Train Toolbox. `ttpy` doesn't support installation on Windows, so it's recommended to use a Linux-based OS. To install the required dependencies, run the following in a new conda environment:

```bash
conda install numpy==1.23.5
conda install scipy
pip install git+https://github.com/oseledets/ttpy.git
```

Note that this repository cannot be installed as a library. Simply clone the repository:
```bash
git clone https://github.com/levdik/ttgsm.git
```

## Usage
An example of usage is provided in `main.py`.

## License
This project is **unlicensed** and placed in the public domain. You are free to use, modify, distribute, and incorporate this code into your own projects without any restrictions.

If you find this code useful in your work, a citation would be appreciated.
