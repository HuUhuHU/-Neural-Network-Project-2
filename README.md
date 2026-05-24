# Neural Network Project 2 — CIFAR-10 & Batch Normalization

**Course:** Neural Network and Deep Learning  

A single notebook covers both parts of Project 2:

1. **Part 1 (60%)** — Train a custom CNN (PreAct residual architecture) on CIFAR-10, with ablation studies and visualizations  
2. **Part 2 (30%)** — Compare VGG-A vs VGG-A+BN and analyze the effect of BN on the loss landscape and gradients  

**Report:** (not included here)

---

## Requirements

- Python 3.9+
- GPU recommended (Part 1 main training + 12 ablation runs are time-consuming; CPU works but is very slow)
- ≥ 4GB VRAM suggested (`BATCH_SIZE=256`; reduce batch size in the notebook if needed)

## Installation

```powershell
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# Linux / macOS 
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Project Structure

```text
PJ2_2026/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── project_2_2026.ipynb         # All code (Part 1 + Part 2)
├── project_2_2026_report.pdf    # Experiment report (Not included here)
├── data/                        # CIFAR-10 (auto-downloaded on first run)
│   └── cifar-10-batches-py/
├── figures/                     # Training curves, ablation plots, VGG figures (generated after run)
├── models/                      # Model checkpoints(in Google Drive)
│   ├── cifarcnn_best.pth        # Part 1 main model (best checkpoint)
│   ├── ablation/                # Best weights for each ablation group
│   │   ├── ablation_structure/
│   │   ├── ablation_activation/
│   │   ├── ablation_weight_decay/
│   │   └── ablation_optimizer/
│   ├── vgg_a_no_bn.pth
│   └── vgg_a_bn.pth
└── vgg_results/                 # Part 2 per-step loss logs at multiple LRs (.txt)
```

---

3. **Sections**

   | Section | Content | Approx. time (GPU) |
   |---------|---------|-------------------|
   | §0–§1.1 | Setup, data loading, compute mean/std | A few minutes |
   | §1.2–§1.3 | Model definition, training loop | Instant |
   | §1.4 | Main model training (100 epochs, early stopping) | ~1–2 h |
   | §1.5 | Ablation (4 groups, 12 runs total) | Several hours |
   | §1.6 | Curves, filters, loss landscape | A few minutes |
   | §2.x | VGG-A / VGG-A+BN, loss band, gradient analysis | ~1 h |

4. **Outputs**
   - Figures: `./figures/`
   - Weights: `./models/` and `./models/ablation/`
   - VGG loss logs: `./vgg_results/`

---

## Main Model Summary

| Item | Setting |
|------|---------|
| Architecture | CIFARCNN (PreAct residual blocks, width=48, depth=3) |
| Parameters | 2,462,426 |
| Data augmentation | RandomCrop(32, padding=4) + RandomHorizontalFlip |
| Normalization | mean/std computed on the training set |
| Optimizer | Adam, lr=1e-3 (fixed; no warmup/cosine) |
| weight_decay | 5e-4 |
| Early stopping | patience=20 |
| Best test accuracy | ~88.97%–89.2% (see actual run output) |

## Ablation Studies (Project Requirements)

| Requirement | Notebook section | Comparison |
|-------------|-------------------|------------|
| (a) filters / neurons | §1.5 structure | width = 32 / 48 / 64 |
| (b) loss + regularization | §1.5 weight_decay | wd = 0 / 1e-4 / 5e-4 / 5e-3 |
| (c) activations | §1.5 activation | ReLU / LeakyReLU / GELU |
| Optimizers (§1.1 item 5) | §1.5 optimizer | Adam (1e-3) vs SGD (0.05) |

Best checkpoints for each ablation run are saved under `./models/ablation/{group_name}/`.

---

## Loading a Checkpoint

Run in the notebook:

```python
import torch

ckpt = torch.load('./models/cifarcnn_best.pth', map_location=device)
model = CIFARCNN(width=48, depth=3, act='relu', dropout=0.4).to(device)
model.load_state_dict(ckpt['state_dict'])
print('test acc:', ckpt['test_acc'])
```

---

## Model Weights Download

**Weights link:** https://drive.google.com/drive/folders/14Vjr5zV3nAe81NKTRhvmNk3vXi6BooKF?
usp=sharing

---

