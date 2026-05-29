# Image-Translation-using-LatentGANs-and-CycleVAEs
## Sketch ↔ Face Translation using Latent Cycle Consistency

A deep learning project built with PyTorch for translating face sketches into realistic face images and vice versa using a shared latent representation and cycle consistency.


---

# Overview

This project implements a generative AI model that learns bidirectional translation between:

* Face sketches → realistic face photos
* Face photos → sketches

The architecture uses:

* Shared encoder for latent representation learning
* Two decoders for domain-specific reconstruction
* Latent cycle consistency for stable translation
* Reconstruction and latent consistency losses

The model is trained on paired sketch/photo face datasets.

---

# Features

* Bidirectional image translation
* Shared latent embedding space
* Encoder-decoder architecture
* Latent cycle consistency training
* PyTorch implementation
* Training, validation, and testing pipelines
* Visualization of generated outputs
* Loss curve plotting

---

# Model Architecture

## Components

### Encoder

The encoder converts input images into a compact latent vector representation.

### Decoder A

Generates sketches from latent embeddings.

### Decoder B

Generates realistic face images from latent embeddings.

### Latent Cycle Consistency

After translation, the generated image is encoded again to ensure the latent representation remains consistent.

---

# Workflow

```text
Photo → Encoder → Latent Space → Sketch Decoder → Generated Sketch
Sketch → Encoder → Latent Space → Photo Decoder → Generated Photo
```

Cycle consistency:

```text
Input → Translation → Re-encoding → Latent Reconstruction
```

---

# Tech Stack

* Python
* PyTorch
* TorchVision
* NumPy
* Matplotlib
* PIL

---

# Dataset

The project uses a paired face sketch/photo dataset.

Dataset structure:

```text
dataset/
│
├── train/
│   ├── sketches/
│   └── photos/
│
├── val/
│   ├── sketches/
│   └── photos/
│
└── test/
    ├── sketches/
    └── photos/
```

Example dataset path used in the notebook:

```python
DATA_ROOT = "/kaggle/input/datasets/almightyj/person-face-sketches"
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Install dependencies:

```bash
pip install torch torchvision matplotlib numpy pillow
```

---

# Training

Run the notebook:

```bash
jupyter notebook gen-ai-proj.ipynb
```

Training configuration:

```python
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 50
LR = 2e-4
```

---

# Loss Function

The total loss combines:

* Reconstruction Loss (L1 Loss)
* Latent Consistency Loss (MSE Loss)

```python
Total Loss = Reconstruction Loss + Latent Cycle Loss
```

---

# Evaluation

The notebook includes:

* Validation loss evaluation
* Test loss evaluation
* Generated image visualization
* Training vs validation loss curves

---

# Results

The model learns meaningful latent representations that allow:

* Realistic sketch generation
* Face reconstruction from sketches
* Stable bidirectional translation

Generated outputs are visualized during testing using image grids.

---

# Future Improvements

Possible improvements include:

* Adding GAN discriminators
* Using perceptual loss
* Higher resolution image generation
* Attention mechanisms
* Diffusion-based refinement
* Better encoder backbones

---

# Repository Structure

```text
.
├── gen-ai-proj.ipynb
├── README.md
└── dataset/
```

---

# Learning Outcomes

This project demonstrates:

* Representation learning
* Image-to-image translation
* Encoder-decoder architectures
* Cycle consistency concepts
* Deep learning training pipelines in PyTorch

---

# License

This project is open-source and available under the MIT License.

---

# Author

Developed as a generative AI innovation by Malaika Saleem.

