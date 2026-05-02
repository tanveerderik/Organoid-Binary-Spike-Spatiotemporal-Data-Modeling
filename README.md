# Organoid-Binary-Spike-Spatiotemporal-Data-Modeling
Context-conditioned hierarchical VQ-VAE for ultra-sparse binary neural spike data (x,y,t). Learns discrete spatiotemporal motifs using multi-level codebooks, with biologically constrained reconstruction and context-aware generation for downstream generative modeling (e.g., MAGVIT-style priors).

# Context-Conditioned Hierarchical VQ-VAE for Sparse Neural Spike Modeling

## Overview
This repository implements a **context-conditioned hierarchical Vector Quantized Variational Autoencoder (VQ-VAE)** for modeling **ultra-sparse binary neural spike data** represented as spatiotemporal volumes \((x, y, t)\). The data originates from electrophysiological recordings of neural cultures, where spike activity is rare, structured, and governed by biological constraints.

The goal is to learn a **discrete latent representation of neural activity motifs** that supports accurate reconstruction and controlled generation, while remaining consistent with both spatial and temporal dynamics.

---

## Motivation
Neural spike data presents several key challenges:
- **Extreme sparsity** (dominance of zero-valued regions)
- **Stochastic spike placement**
- **Strong spatial and temporal dependencies**

Naïve models tend to:
- Collapse to all-zero outputs, or  
- Overproduce spikes that violate biological realism  

This project addresses these issues through:
- Hierarchical latent discretization  
- Context-aware conditioning  
- Carefully designed loss functions and constraints  

---

## Key Features

### Hierarchical VQ-VAE Architecture
- Multi-level codebooks (VQ-VAE2-style)
- Coarse-to-fine representation learning
- Residual refinement for precise spike placement

### Context Conditioning
- **Global context**: assay or experimental identifiers  
- **Local context**: activity features (density, variability, heterogeneity, etc.)  
- Decoder cross-attention for controlled generation  

### Sparse Data Handling
- Loss balancing to prevent all-zero collapse  
- Spike-sensitive reconstruction objectives  
- Separation of active vs. blank regions  

### Biological Constraints
- Temporal adjacency / short-gap (ISI) regularization  
- Spatial support constraints (token-level and pixel-level)  
- Activity statistics alignment  

### Memory Bank Priors
- Assay-wise spatial support maps  
- Temporal adjacency distributions  
- Used for pretraining global context embeddings  

---

## Training Pipeline

The model is trained in multiple stages:

### Stage 0: Global Context (GCT) Embedding Pretraining
- Learns assay-specific embeddings  
- Uses memory banks:
  - Tokenwise spatial support  
  - Pixelwise spatial support  
  - Temporal adjacency statistics  

### Stage 1: Context-Agnostic VQ-VAE Training
- Encoder and codebooks learn spike motifs  
- No context conditioning  
- Focus on robust latent representation  

### Stage 2: Context-Conditioned Decoder Training
- Encoder and codebooks are frozen  
- Decoder learns to use context via cross-attention  
- Enables controlled and biologically consistent generation  

### (Planned) Stage 3: Token Prior Learning
- MAGVIT / MaskGIT-style prior over codebook tokens  
- Enables full generative modeling  

---

## Model Design Highlights

- **Encoder**: Sparse-aware, processes primarily active regions  
- **Codebooks**: Discrete latent representations of spike motifs  
- **Decoder**: Dense reconstruction with optional context conditioning  
- **Type Embedding**: Separates blank vs. active patches  
- **Loss Design**:
  - Weighted reconstruction loss  
  - Tolerant spike matching  
  - Spatial violation penalties  
  - Temporal adjacency constraints  
  - Codebook regularization  

---

## Applications
- Neural activity modeling and simulation  
- Electrophysiology data generation  
- Spike motif discovery  
- Organoid intelligence research  
- Context-aware neural signal synthesis  

---

## Project Status
⚠️ Active research project  
- Architecture and training strategies are under continuous development  
- Designed for extensibility toward generative priors and foundation models  

---

## Keywords
VQ-VAE, neural spikes, sparse data, electrophysiology, generative modeling, transformers, context conditioning, MAGVIT, MaskGIT, computational neuroscience
