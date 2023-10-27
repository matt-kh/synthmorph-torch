# synthmorph-torch

Implementation of [SynthMorph](https://arxiv.org/abs/2004.10282) in PyTorch.

This project mainly adapts code from [VoxelMorph](https://github.com/voxelmorph/voxelmorph) repo and its supporting library, [Neurite](https://github.com/adalca/neurite).s
## Purpose
This project initially serves to reproduce the SynthMorph [demo](https://colab.research.google.com/drive/1GjpjkhKGrg5W-cvZVObBo3IoIUwaPZBU?usp=sharing) created by the paper authors. The original demo was done in TensorFlow (TF), and `synthmorph_demo.ipynb` aims to reproduce it using PyTorch. 

Although the `voxelmorph` package provides the required modules in PyTorch, the registration output in PyTorch does not match the output from the TF implementation. Therefore, this repo also reimplements the required modules in PyTorch, faithfully following the TF implementation. In addition, this repo also provides weight transfer steps in `tf2torch.ipynb`.

## Dependencies
Python 3.8.x versions are recommended when using code from this repo, and there is no promise of compatibility with other Python versions. The dependencies in `requirements.txt` are based on Linux installations for GPU. For windows and/or CPU only packages, you might need to change how to install some packages, such as TensorFlow.

In your Python environment:
```bash
pip install -r requirements.txt
```

## Updates
- Currently, data generation implementation in Torch is under development in the `datagen` branch. This will eventually replace all the NumPy functions for data generation, which is noticeably slow.