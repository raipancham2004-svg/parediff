# Setup Instructions — Personal Laptop

You will need to do this ONCE on your personal laptop (or college lab GPU machine). I'll guide you each step.

## 1. Install Conda (5 min)

```bash
# Linux/WSL:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# macOS:
brew install --cask miniconda

# Windows: download installer from conda.io
```

## 2. Create Python 3.10 environment (3 min)

```bash
conda create -n parediff python=3.10 -y
conda activate parediff
```

## 3. Install ML stack (10 min)

```bash
# PyTorch (pick the right CUDA version for your GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# PyTorch Geometric (graph NN)
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# Diffusion + utilities
pip install diffusers transformers accelerate
pip install wandb matplotlib pandas networkx
pip install scipy scikit-learn tqdm einops
```

## 4. Install OpenROAD (15 min via Docker)

OpenROAD is the open-source PnR tool we'll use to evaluate routability.

```bash
# Easiest path: Docker
docker pull openroad/orfs:latest

# Test it works:
docker run -it openroad/orfs:latest openroad --version
```

Alternative (build from source) is available but takes 1+ hours.

## 5. Clone the project

```bash
cd ~
mkdir -p projects && cd projects
# I'll give you the GitHub URL once we push the initial code
```

## 6. Sanity check — verify your GPU works

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True NVIDIA GeForce RTX <whatever>
```

If you see `False`, your CUDA install is wrong — tell me.

## 7. Sanity check — verify PyG works

```bash
python -c "from torch_geometric.nn import GCNConv; print('PyG OK')"
```

---

## When you've finished setup, tell me:

1. Output of `nvidia-smi` (so I know GPU model + VRAM)
2. Output of `python -c "import torch; print(torch.__version__)"`
3. Output of `docker run openroad/orfs:latest openroad --version`
4. Whether you have ~50 GB free disk space (we'll need it for datasets + checkpoints)

Once these are confirmed, I'll:
- Push the initial code skeleton to GitHub (you'll create the repo)
- Give you the first runnable script (download ISPD'15 dataset)
- Walk you through the first baseline run

---

## Common pitfalls

- **CUDA version mismatch:** PyTorch CUDA must match your GPU driver's CUDA. Check `nvidia-smi` top-right corner for max-supported CUDA, install PyTorch for that version or lower.
- **Out of memory on laptop:** RTX 30/40 series have 8-24 GB. We'll batch-size accordingly. If <12 GB, tell me.
- **Docker on Windows:** needs WSL2. If on Windows, install WSL2 first.
- **Disk space:** ISPD'15 ~5 GB, ASAP7 ~3 GB, NanGate45 ~2 GB, checkpoints ~10 GB, OpenROAD ~5 GB. ~25 GB minimum, 50 GB safer.
