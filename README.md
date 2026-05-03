# VAE Fingerprint Analysis

A Variational Autoencoder (VAE) trained on the [SOCOFing dataset](https://www.kaggle.com/datasets/ruizgara/socofing) for fingerprint reconstruction, generation, and latent space analysis.

---

## Overview

This project trains a convolutional VAE on real fingerprint images and evaluates its ability to reconstruct, generate, and meaningfully encode fingerprints in a continuous latent space. The evaluation goes beyond standard reconstruction metrics — it includes latent space interpolation, arithmetic, clustering, robustness on altered fingerprints, and generative quality via FID.

---

## Repository Structure

```
├── vae_fingerprints.ipynb            # Training notebook
├── vae_fingerprints_evaluation.ipynb # Evaluation notebook
└── README.md
```

---

## Dataset

**SOCOFing** (Sokoto Coventry Fingerprint Dataset)  
- 6,000 real fingerprint images from 600 subjects  
- Labels: subject ID, gender (M/F), hand (Left/Right), finger type (index, middle, ring, little, thumb)  
- Split: 4,800 train / 600 val / 600 test  
- Includes altered subsets: Altered-Easy, Altered-Medium, Altered-Hard  

Download from Kaggle: `kaggle datasets download -d ruizgara/socofing`  
The notebooks auto-download via Kaggle CLI or gdown fallback.

---

## Model Architecture

Convolutional VAE with symmetric encoder/decoder.

| Component | Details |
|---|---|
| Input | 128×128 grayscale images |
| Encoder | Conv2d → BN → LeakyReLU × 5 stages, channels: [32, 64, 128, 256, 512] |
| Latent space | 256-dimensional (μ and log σ² heads) |
| Decoder | ConvTranspose2d → BN → ReLU × 5 stages, channels: [512, 256, 128, 64, 32] |
| Output | 128×128 Sigmoid activation |

**Loss:** `L = BCE(reconstruction) + β · KL(N(μ,σ²) || N(0,I))`  
KL annealing over 80 epochs (β-VAE style warmup), max β = 0.1, free bits = 0.1

---

## Training

| Hyperparameter | Value |
|---|---|
| Epochs | 200 |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 5e-4 → 1e-6 (cosine schedule) |
| KL warmup | 80 epochs |
| Hardware | Google Colab T4 GPU (~3 hrs) |

Checkpoints are saved to Google Drive every 2 epochs. Best model selected by validation loss.

---

## Results

### Reconstruction Quality (validation set)

| Metric | Mean | Std |
|---|---|---|
| SSIM | 0.7019 | ±0.0727 |
| PSNR | 16.19 dB | ±1.55 |
| MSE | 0.0255 | ±0.0086 |
| KL Divergence | 3602.81 | — |

### Generative Quality

| Metric | Value |
|---|---|
| FID Score | 3.36 |

> FID computed using pixel-level features (not Inception), so it measures reconstruction distribution similarity rather than standard generative FID.

### Robustness on Altered Fingerprints

| Subset | SSIM | PSNR |
|---|---|---|
| Real | 0.7019 | 16.19 dB |
| Altered-Easy | 0.6794 | 15.75 dB |
| Altered-Medium | 0.6346 | 15.36 dB |
| Altered-Hard | 0.6182 | 15.36 dB |

Graceful degradation confirms the model learned clean fingerprint structure, not noise artifacts.

### Latent Space Clustering (by finger type)

| Metric | Value |
|---|---|
| Optimal K (K-Means) | 2 (true: 5) |
| Silhouette @ K=5 | 0.0281 |
| Adjusted Rand Index | 0.0856 |

The latent space does not naturally separate by finger type without supervised signal — an expected limitation of a standard VAE.

---

## Qualitative Results

**Latent Interpolation** — smooth, visually coherent transitions between fingerprints (same person/different fingers, different persons, across genders), demonstrating a well-regularized continuous latent space.

**Latent Arithmetic** — vector operations like `z(Left_Index) - z(Left_Middle) + z(Right_Middle)` produce plausible fingerprint outputs, suggesting partial disentanglement of hand/finger attributes.

**Latent Traversal** — varying individual latent dimensions reveals control over ridge orientation, curvature, and pattern density.

**Generated Samples** — direct sampling from N(0,I) produces blurry outputs due to the high KL divergence (prior mismatch). Reconstructions are significantly sharper.

---

## Key Observations

- The VAE is strong as a **compression and interpolation** model — latent space is smooth and continuous
- It is weaker as a **pure generative sampler** — the high KL divergence indicates posterior collapse hasn't occurred but the prior is poorly matched
- Finger type is **not disentangled** in the latent space; a supervised or β-VAE approach would be needed for structured separation
- The model is **robust to fingerprint alterations**, with SSIM degrading gracefully across Easy → Hard

---

## Setup

```bash
# Install dependencies
pip install torch torchvision torchinfo scikit-learn scipy matplotlib seaborn tqdm pillow pandas

# Run in Google Colab (recommended — Kaggle + Drive integration built in)
# Open vae_fingerprints.ipynb → Run all
# Then open vae_fingerprints_evaluation.ipynb → Run all
```

Checkpoints and results are saved to:  
`Google Drive / Colab_Notebooks / VAE / vae_fingerprints /`

---

## Possible Improvements

- Use β-VAE or FactorVAE to improve disentanglement of finger/hand attributes
- Add a supervised contrastive loss on the latent space for better clustering
- Reduce KL weight further or use free bits more aggressively to improve prior matching
- Replace pixel-level FID with Inception-based FID for standard benchmarking
- Try VQ-VAE for sharper reconstructions

---

## References

- Shehu, Y.I. et al. *SOCOFing: Sokoto Coventry Fingerprint Dataset*. arXiv:1807.10609, 2018
- Kingma, D.P. & Welling, M. *Auto-Encoding Variational Bayes*. ICLR 2014
- Higgins, I. et al. *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework*. ICLR 2017
