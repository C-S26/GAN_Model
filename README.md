# GAN Model Testing Setup

**Started on:** 02-04-2025  
**Note:** Most of the work is done in WSL; all commands and configurations are related to Linux environment. 
**(28-07-2025)Shifted for colab env**

---
## ✅ Requirements

### Software

- **Python** 3.11
- **TensorFlow** 2.15 or newer
- **NumPy**
- **tqdm**
- **matplotlib** (for plots)
- **pandas** (if using CSV data)
- Works on **Linux**, **Windows**, or **WSL** (Linux recommended)

### Optional: GPU Support

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
  pyenv install 3.7.17
  pyenv virtualenv 3.7.17 gan-env
  pyenv activate gan-env
  pip install tensorflow==1.15.0 numpy==1.18.5 tqdm
```
---

![Build](https://img.shields.io/badge/Project%20Going%20On)

