# Denoising Diffusion Probabilistic Models (DDPM)

A PyTorch/TensorFlow implementation of the groundbreaking paper ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239) by Ho et al. (2020), trained on the CelebA dataset to generate realistic human faces.

## Table of Contents
- [Overview](#overview)
- [The Mathematics Behind DDPM](#the-mathematics-behind-ddpm)
- [U-Net Architecture](#u-net-architecture)
- [Training Process](#training-process)
- [Results](#results)
- [Installation & Usage](#installation--usage)

---

## Overview

Denoising Diffusion Probabilistic Models (DDPM) are a class of generative models that learn to generate data by reversing a gradual noising process. The key insight is beautifully simple yet powerful: if we can learn to denoise images step-by-step, we can generate new images by starting from pure noise and iteratively denoising it.

### The Core Idea

**Forward Process (Diffusion):** Gradually add Gaussian noise to real images over T timesteps until they become pure noise.

**Reverse Process (Denoising):** Learn a neural network to reverse this process removing noise step by step to generate realistic images from random noise.

---

## The Mathematics Behind DDPM

### Forward Diffusion Process

The forward process adds noise to an image x₀ over T timesteps. At each timestep t, we add a small amount of Gaussian noise:

```
q(xₜ | xₜ₋₁) = N(xₜ; √(1-βₜ)·xₜ₋₁, βₜI)
```

Where:
- **xₜ**: The noisy image at timestep t
- **βₜ**: The noise schedule (variance) at timestep t
- **N**: Gaussian distribution

**Key Insight:** We can derive a closed-form expression to jump directly to any timestep t without computing all intermediate steps!

### The Magic of αₜ and ᾱₜ (Alpha Bar)

This is where most explanations get hand-wavy, but the math is crucial:

Define:
```
αₜ = 1 - βₜ
```

**Why αₜ?** It represents the "signal retention rate" at timestep t. If βₜ is the noise we add, then αₜ = 1 - βₜ is how much of the original signal we keep.

Now, the cumulative product (this is the key):
```
ᾱₜ = ∏(i=1 to t) αᵢ = α₁ · α₂ · α₃ · ... · αₜ
```

**Why ᾱₜ (alpha bar)?** This represents the **cumulative signal retention** from the original image x₀ to timestep t. 

### The Reparameterization Trick

Using ᾱₜ, we can express any noisy image xₜ directly in terms of the original image x₀:

```
xₜ = √ᾱₜ · x₀ + √(1-ᾱₜ) · ε

where ε ~ N(0, I) is standard Gaussian noise
```

**This is huge!** Instead of applying noise T times sequentially, we can jump to any timestep in one step.

**Why these specific coefficients?**
- **√ᾱₜ**: Scales the original image
- **√(1-ᾱₜ)**: Scales the noise
- Together they ensure the variance remains normalized: `Var(xₜ) = ᾱₜ + (1-ᾱₜ) = 1`

### The Training Objective

The model learns to predict the noise ε that was added to create xₜ from x₀:

```
L_simple = 𝔼[‖ε - ε_θ(xₜ, t)‖²]
```

Where:
- **ε**: The actual noise added
- **ε_θ(xₜ, t)**: The noise predicted by our neural network
- **t**: The timestep (fed as input to help the model know how much noise to expect)

**Why predict noise instead of the clean image?** 
1. More stable training (noise has consistent scale)
2. Better gradient flow
3. Empirically works better across different timesteps

### Reverse Process (Sampling)

To generate images, we start from pure noise x_T ~ N(0, I) and iteratively denoise:

```
xₜ₋₁ = 1/√αₜ · (xₜ - (1-αₜ)/√(1-ᾱₜ) · ε_θ(xₜ, t)) + σₜ · z

where z ~ N(0, I) and σₜ = √βₜ
```

We repeat this for t = T, T-1, ..., 1 to get our final image x₀.

---

## U-Net Architecture

We use a modified U-Net architecture as our denoising network ε_θ. The U-Net was originally designed for image segmentation but works exceptionally well for diffusion models.

### Why U-Net?

1. **Preserves spatial information** through skip connections
2. **Multi-scale processing** via downsampling and upsampling
3. **Bottleneck captures global context** while maintaining local details

### Architecture Overview

```
Input: Noisy image xₜ (64×64×3) + Timestep embedding t

                        [Encoder - Downsampling Path]
                                    ↓
    64×64×32  →  [ResBlock × 2] → [Skip Connection 1] ─┐
                      ↓ MaxPool(2×2)                     │
    32×32×64  →  [ResBlock × 2] → [Attention] → [Skip 2]─┤
                      ↓ MaxPool(2×2)                      │
    16×16×128 →  [ResBlock × 2] → [Skip Connection 3] ─┤ │
                      ↓ MaxPool(2×2)                     │ │
    8×8×256   →  [ResBlock × 2] → [Skip Connection 4] ─┤ │ │
                                                         │ │ │
                      [Bottleneck]                       │ │ │
    8×8×256   →  [ResBlock] → [Attention] → [ResBlock] │ │ │
                                                         │ │ │
                        [Decoder - Upsampling Path]     │ │ │
                                    ↓                    │ │ │
    8×8×256   →  [Concatenate Skip 4] ←─────────────────┘ │ │
                      ↓ Upsample(2×2)                      │ │
    16×16×128 →  [ResBlock × 2] → [Concatenate Skip 3] ←──┘ │
                      ↓ Upsample(2×2)                        │
    32×32×64  →  [ResBlock × 2] → [Attention] → [Concat Skip 2] ←┘
                      ↓ Upsample(2×2)
    64×64×32  →  [ResBlock × 2] → [Concatenate Skip 1] ←──────────┘
                      ↓
    64×64×3   →  [Final Conv] → Output: Predicted noise ε

Timestep Embedding: Sinusoidal positional encoding (1024-dim) 
                   injected into each ResBlock via AdaGN
```

### Key Components Explained

#### 1. **Residual Blocks (ResBlock)**
```
Input → GroupNorm → SiLU → Conv3×3 → 
     → GroupNorm → Add Timestep Embedding → 
     → SiLU → Conv3×3 → Add Input (Residual) → Output
```

**Why?**
- Gradient flow through skip connections
- Timestep conditioning at each layer
- Stable training for deep networks

#### 2. **Timestep Embedding**
The timestep t is encoded using sinusoidal positional embeddings (like in Transformers):

```python
pos = t / (10000 ^ (2i/d))
emb[2i] = sin(pos)
emb[2i+1] = cos(pos)
```

This creates a unique embedding for each timestep that the model learns to interpret.

#### 3. **Self-Attention Layers**
Applied at 32×32 resolution (after first downsampling):

```
Q, K, V = Linear(x)
Attention(Q,K,V) = Softmax(QK^T / √d) · V
```

**Why only at 32×32?**
- Computational efficiency (attention is O(n²))
- Captures mid-level features (edges, textures)
- Still provides global context without being too expensive

#### 4. **Downsampling Path**
- **Purpose**: Extract hierarchical features from coarse to fine
- **Operations**: 
  - ResBlocks process features at each resolution
  - MaxPooling(2×2) reduces spatial dimensions
  - Feature channels double at each level (32→64→128→256)

#### 5. **Upsampling Path**
- **Purpose**: Reconstruct spatial resolution while refining features
- **Operations**:
  - Bilinear upsampling (2×2) increases spatial dimensions
  - Concatenate with corresponding encoder features (skip connections)
  - ResBlocks process the combined features
  - Feature channels halve at each level (256→128→64→32)

#### 6. **Skip Connections**
These are **critical** and often misunderstood:

**What they do:** Concatenate encoder features directly to decoder features at matching resolutions.

**Why they matter:**
- **Preserve fine details** that get lost in downsampling
- **Better gradient flow** during backpropagation
- **Multi-scale information:** Decoder sees both:
  - High-level features from the bottleneck (what to generate)
  - Low-level features from encoder (where to place details)

**Example at 32×32 resolution:**
```
Encoder output: [Batch, 32, 32, 64]
Decoder before concat: [Batch, 32, 32, 64]
After concatenation: [Batch, 32, 32, 128]  ← Double the channels!
```

### Downsampling vs Upsampling: The Information Flow

**Downsampling (Encoder):**
```
64×64×3  → What is this? (Raw pixels)
32×32×64 → Edges and basic shapes
16×16×128 → Object parts (eyes, nose)
8×8×256 → Global structure (face composition)
```

**Bottleneck:**
```
8×8×256 → Semantic understanding (it's a face, lighting, pose)
```

**Upsampling (Decoder):**
```
8×8×256 → Start with global structure
16×16×128 → Add object parts + encoder details
32×32×64 → Refine textures + encoder edges  
64×64×3 → Final details + encoder pixel-level info
```

**The Magic:** At each upsampling step, the network combines:
- **Top-down information:** "What should be generated" (from bottleneck)
- **Bottom-up information:** "Where details should go" (from skip connections)

This is why U-Net works so well for denoising it can maintain spatial precision while understanding global context!

---

## Training Process

### Hyperparameters

```python
BATCH_SIZE = 16  # Using P100 GPU
TIME_STEPS = 1000
IMAGE_SIZE = 64×64×3
LEARNING_RATE = 2e-4
N_EPOCHS = 10

# U-Net Architecture
N_RESNETS = 2  # ResBlocks per level
N_GROUPS = 2   # Group Normalization groups
N_HEADS = 8    # Multi-head attention
ATTN_DIM = 256 # Attention dimension

# Noise Schedule (Linear)
β_start = 1e-4
β_end = 0.02
βₜ = linspace(β_start, β_end, TIME_STEPS)
```

### Training Algorithm

```
For each epoch:
    For each batch of images x₀:
        1. Sample timestep t ~ Uniform(1, T)
        2. Sample noise ε ~ N(0, I)
        3. Create noisy image: xₜ = √ᾱₜ·x₀ + √(1-ᾱₜ)·ε
        4. Predict noise: ε_pred = model(xₜ, t)
        5. Compute loss: L = MSE(ε, ε_pred)
        6. Backprop and update weights
```

### GPU Performance Analysis

We conducted extensive testing to optimize training performance across different GPU configurations:

#### T4 vs P100 Comparison

| Metric | T4 GPU | P100 GPU | Improvement |
|--------|--------|----------|-------------|
| **Time per Epoch** | 13,672 sec (~3.8 hrs) | 6,972 sec (~1.94 hrs) | **1.95x faster** |
| **Batch Size** | 8 | 16 | 2x larger |
| **Memory Bandwidth** | 320 GB/s | 732 GB/s | 2.3x faster |
| **FP16 Performance** | 65 TFLOPS | 21 TFLOPS | Better FP32 |
| **Training Stability** | Frequent OOM errors | Stable training | ✅ |

#### Why P100 is Superior for DDPM Training

**1. Memory Bandwidth (Critical for U-Net)**
- **T4**: 320 GB/s - bottleneck for large feature maps
- **P100**: 732 GB/s - 2.3x faster data movement
- U-Net's skip connections require heavy memory I/O
- **Result**: P100 handles concatenations and upsampling much faster

**2. Batch Size Capability**
- **T4**: Limited to batch_size=8
  - Frequent OOM (Out of Memory) errors with batch_size=16
  - Required aggressive memory management
- **P100**: Stable with batch_size=16
  - Better GPU utilization
  - More stable gradient estimates
  - Faster convergence

**3. Mixed Precision Training**
```python
# Mixed precision is crucial for memory optimization
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```
- **T4**: Excels at FP16 (65 TFLOPS) but limited by memory bandwidth
- **P100**: Better balanced FP32 performance with superior memory bandwidth
- **Key insight**: For DDPM, memory bandwidth > raw FLOPS

**4. Computational Efficiency**
- T4's tensor cores are optimized for inference
- P100's architecture is optimized for training workloads
- The 1.95x speedup validates P100's training superiority

### Memory Optimization Techniques Applied

#### Issue: Initial OOM Errors on T4
```
ResourceExhaustedError: failed to allocate memory
```

#### Solutions Implemented:

**1. Enable Memory Growth**
```python
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```
**Impact**: Allocates memory as needed vs. pre-allocating all GPU memory

**2. Mixed Precision Training**
```python
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```
**Impact**: ~50% memory reduction + faster computation

**3. Batch Size Tuning**
- Started with batch_size=16 → OOM on T4
- Reduced to batch_size=8 → Stable on T4
- Switched to P100 → batch_size=16 works perfectly

**4. Single GPU Training Decision**
- Initially attempted multi-GPU (2x T4) with MirroredStrategy
- Encountered complex errors:
  ```
  InvalidArgumentError: You must feed a value for placeholder tensor
  ValueError: A KerasTensor cannot be used as input to a TensorFlow function
  ```
- **Decision**: Single GPU training is more stable and easier to debug
- **Trade-off**: Sacrificed potential 1.8-1.9x speedup for reliability

### Training Stability: The Extract Function Bug

**Critical Bug Found:**
```python
# WRONG - Causes shape mismatch in distributed training
def extract(a, t, x_shape):
    b, *_ = t.shape  # Fails in graph mode!
    out = tf.gather(a, t)
    return tf.reshape(out, (b,*((1,)*(len(x_shape)-1))))
```

**Problem**: `t.shape` returns symbolic dimensions in graph mode, causing:
```
ValueError: required broadcastable shapes [Op:Sub]
```

**Fix:**
```python
# CORRECT - Works in both eager and graph mode
def extract(a, t, x_shape):
    batch_size = tf.shape(t)[0]  # Dynamic shape
    out = tf.gather(a, t)
    return tf.reshape(out, [batch_size, 1, 1, 1])  # List, not tuple
```

**Key Lesson**: Always use `tf.shape()` for dynamic dimensions in `@tf.function` decorated code!

### Why This Works

The model learns to denoise images at **all** noise levels simultaneously:
- Early timesteps (t near 0): Tiny noise, subtle corrections
- Middle timesteps: Moderate noise, shape refinement  
- Late timesteps (t near T): Heavy noise, high-level structure

By training on random timesteps, the model becomes robust across the entire diffusion process.

### Final Training Configuration

```python
# Optimized for P100 GPU
DEVICE = "P100"
BATCH_SIZE = 16
MIXED_PRECISION = True
MEMORY_GROWTH = True
SINGLE_GPU = True  # More stable than distributed

# Expected Training Time
TOTAL_EPOCHS = 10
TIME_PER_EPOCH = 6972  # seconds
TOTAL_TIME = 10 * 6972 / 3600  # ≈ 19.4 hours
```

---

## Results

### Training Metrics
- **Dataset**: CelebA (202,599 images)
- **GPU**: NVIDIA P100 (16GB)
- **Training Time**: 
  - Per Epoch: 6,972 seconds (~1.94 hours)
  - Total (10 epochs): ~19.4 hours
- **Batch Size**: 16
- **Mixed Precision**: Enabled (FP16)
- **Final Loss**: [Your final loss here]

### Performance Comparison

#### Hardware Evolution During Training

| Stage | GPU | Batch Size | Time/Epoch | Issue |
|-------|-----|------------|------------|-------|
| **Initial** | T4 | 16 | OOM Error | Memory exhausted |
| **Attempt 1** | T4 | 8 | 13,672 sec | Stable but slow |
| **Attempt 2** | 2x T4 (MirroredStrategy) | 16 | Failed | Distributed training errors |
| **Final** | P100 | 16 | 6,972 sec | ✅ Optimal |

**Key Insight**: P100's superior memory bandwidth (732 GB/s vs 320 GB/s) makes it 1.95x faster than T4 for U-Net architectures with heavy skip connections.

### Generated Samples
[Add your generated images here after training]

**Sampling Process:**
1. Start with pure noise: x_T ~ N(0, I)
2. Iteratively denoise for t = 1000, 999, ..., 1
3. Each step removes a small amount of predicted noise
4. Final result: Realistic 64×64 face image

### Sampling Process Visualization
[Show the gradual denoising from T→0]

**Timestep Examples:**
- t=1000 (pure noise): Random static
- t=750: Vague color blobs
- t=500: Face shape emerges
- t=250: Features become clear
- t=0: High-quality face

### Training Loss Curve
[Add your loss curve plot here]

Expected behavior:
- Initial loss: ~0.5-1.0 (model learning noise patterns)
- Mid training: ~0.1-0.3 (refining details)
- Final loss: ~0.05-0.15 (generating realistic faces)

---

## Installation & Usage

### Requirements
```bash
pip install tensorflow tensorflow-gpu numpy matplotlib
```

### Training
```python
# Load dataset
python train.py --dataset celeba --epochs 10 --batch_size 8

# Resume from checkpoint
python train.py --resume unet_epoch_5.keras
```

### Generation
```python
# Generate new faces
python generate.py --num_samples 16 --model unet_epoch_10.keras
```

---

## Key Takeaways

### Mathematical Insights
1. **ᾱₜ (alpha bar) is the cumulative signal retention**, allowing us to jump to any noise level directly
2. **The model predicts noise, not images**, which provides more stable training
3. **The variance-preserving property** (√ᾱₜ and √(1-ᾱₜ)) ensures stable noise addition at all timesteps
4. **Timestep conditioning** is essential the model needs to know how much noise to remove
5. **The reparameterization trick** enables efficient one-step computation: xₜ = √ᾱₜ·x₀ + √(1-ᾱₜ)·ε

### Architectural Insights
6. **U-Net's skip connections** preserve spatial details while the bottleneck captures global context
7. **Multi-scale attention** at 32×32 balances computational cost with global understanding
8. **ResBlocks with timestep injection** allow the model to adapt its denoising strategy per timestep
9. **Symmetric encoder-decoder** ensures information can flow both ways

### Training Insights
10. **P100 > T4 for training**: Memory bandwidth matters more than raw FLOPS for U-Net
11. **Single GPU is more stable** than distributed training for complex models with dynamic shapes
12. **Mixed precision training** is essential saves 50% memory with minimal accuracy loss
13. **The `extract()` function bug** taught us: always use `tf.shape()` for dynamic dimensions
14. **Batch size impacts stability**: Larger batches = better gradient estimates = faster convergence

### Implementation Lessons Learned

**Problem**: Multi-GPU training with MirroredStrategy
```python
# Attempted but failed due to:
InvalidArgumentError: placeholder tensor errors
ValueError: KerasTensor cannot be used in TensorFlow functions
```
**Solution**: Single GPU training is simpler, more stable, and still fast enough

**Problem**: T4 OOM with batch_size=16
```python
ResourceExhaustedError: failed to allocate memory
```
**Solution**: Either reduce batch_size=8 OR switch to P100

**Problem**: Shape broadcasting errors in distributed training
```python
# Bug in extract function
b, *_ = t.shape  # Symbolic shape breaks in graph mode
```
**Solution**: Use `tf.shape(t)[0]` for dynamic batch dimensions

### Why DDPM Works So Well

The genius of DDPM lies in its simplicity:
- **Training**: Predict noise at random timesteps (parallelizable, stable)
- **Sampling**: Iteratively denoise (slow but high quality)
- **Architecture**: U-Net naturally fits the task (spatial preservation + global context)
- **Math**: Variance-preserving noise addition keeps everything normalized

---

## References

1. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - Ho et al., 2020
2. [Understanding Diffusion Models](https://arxiv.org/abs/2208.11970) - Luo, 2022
3. [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) - Ronneberger et al., 2015

---

## License

MIT License - Feel free to use this code for your own projects!

---

## Acknowledgments

- Original DDPM paper authors for the groundbreaking research
- CelebA dataset creators
- The open-source community for TensorFlow/PyTorch implementations

---

**Star this repo if you found it helpful!**
