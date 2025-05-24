# GAN Model Testing Setup

**Started on:** 02-04-2025  
**Note:** Most of the work is done in WSL; all commands and configurations are related to Linux environment.

---

## Requirements

### Software

- Python 3.7.17 (install using `pyenv` in WSL recommended)
- TensorFlow 1.15.0 (CPU version)
- numpy 1.18.5
- tqdm (latest)
- OS: Linux, Windows, or WSL (Linux environment recommended)
- CPU-based execution by default; GPU modifications will be provided later.

### GPU PC Requirements (Optional)

- NVIDIA GPU: minimum **RTX 1050** recommended (**RTX 3050** or above preferred)
- Video RAM: minimum 4GB, ideally 8–12 GB
- GPU Driver version: >= 510.xx
- CUDA version: 10.0 (officially supported by TensorFlow 1.15)  
  > **Note:** TensorFlow 1.15 supports CUDA 10.0 and cuDNN 7.4/7.6. CUDA 12.x is *not* officially supported and may cause issues unless using custom builds.

---

## Installation Notes

- Use `pyenv` to install Python 3.7.17 in WSL:  
  ```bash
  pyenv install 3.7.17
  pyenv virtualenv 3.7.17 gan-env
  pyenv activate gan-env
  pip install tensorflow==1.15.0 numpy==1.18.5 tqdm



