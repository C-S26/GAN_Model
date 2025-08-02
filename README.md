# GAN Model Testing Setup

**Started on:** 02-04-2025  
**Note:** Most of the work is done in WSL; all commands and configurations are related to Linux environment. 
**(28-07-2025)Shifted for colab env**

---
## Requirements

### Software

- **Python** 3.11
- **TensorFlow** 2.15 or newer
- **NumPy**
- **tqdm**
- **matplotlib** (for plots)
- **pandas** (if using CSV data)
- Works on **Linux**, **Windows**, or **WSL** (Linux recommended)

### GPU Specifications:

- **NVIDIA GPU** (RTX 1050 or better)
- **Video RAM**: 4 GB minimum (8–12 GB preferred)
- **CUDA Toolkit**: 11.2 or newer
- **cuDNN**: 8.x
- **Driver**: Version 510+ (`nvidia-smi` to check)

> TensorFlow 2.x supports GPU automatically if drivers & CUDA are correctly set up.

### Setup Instructions

```bash
# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install packages
pip install tensorflow numpy tqdm matplotlib pandas
```

---

## Installation Notes

Use `pyenv` to install Python 3.7.17 in WSL:  
  ```bash
  pyenv install 3.11
  pyenv virtualenv 3.11 gan-env
  pyenv activate gan-env
  pip install tensorflow numpy tqdm
```
---
![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20WSL-lightgrey)
![Build Status](https://github.com/C-S26/GAN_Model/actions/workflows/build.yml/badge.svg)


