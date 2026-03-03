# Multiverse Setup Guide

Complete installation and setup instructions for Multiverse - a production-grade reinforcement learning framework.

**Last Updated:** March 2, 2026
**Tested On:** Windows 11, Python 3.12

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Install](#quick-install)
3. [Detailed Setup](#detailed-setup)
4. [Verification](#verification)
5. [Optional Components](#optional-components)
6. [Docker Setup](#docker-setup)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required

- **Python 3.10 or higher** (tested on 3.12)
  - Check: `python --version`
  - Download: https://www.python.org/downloads/

- **Git** (for cloning the repository)
  - Check: `git --version`
  - Download: https://git-scm.com/downloads

- **10GB free disk space** (for environments, models, and run artifacts)

### Optional

- **GPU with CUDA** (for faster training)
  - PyTorch will use CPU by default
  - For GPU: Install PyTorch with CUDA from https://pytorch.org/

- **Docker & Docker Compose** (for scaling infrastructure)
  - Only needed if you want to use the optional scaling stack
  - Download: https://www.docker.com/get-started

---

## Quick Install

For experienced users who want to get started fast:

```bash
# Clone the repository
git clone https://github.com/wilker00/multiverse.git
cd multiverse

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m pytest tests/ -q

# Run your first training
python tools/train_agent.py --algo random --verse line_world --episodes 10
```

**Done!** Jump to [Quickstart Guide](QUICKSTART.md) for your first real training session.

---

## Detailed Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/wilker00/multiverse.git
cd multiverse
```

### Step 2: Create Python Virtual Environment

**Why?** Isolates Multiverse dependencies from your system Python.

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Verify activation:** Your prompt should now start with `(.venv)`

### Step 3: Install Core Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computing
- **Scikit-learn** - ML utilities
- **Pydantic** - Type validation
- **Typer** - CLI framework
- And more (see `requirements.txt` for full list)

**Installation time:** 2-5 minutes depending on internet speed

### Step 4: Configure Environment (Optional)

```bash
# Copy example configuration
cp .env.example .env

# Edit with your preferred editor
notepad .env     # Windows
nano .env        # Linux/Mac
```

**Note:** Most settings have sensible defaults. Only required if using Docker (see [Docker Setup](#docker-setup)).

### Step 5: Verify Installation

Run the test suite to ensure everything works:

```bash
python -m pytest tests/ -q
```

**Expected output shape:** A pytest pass summary for your local environment.

**Current repo-local collection:** `314 tests` via:
```bash
python -m pytest tests test_dt_memory.py --collect-only -q
```

**If tests fail:** See [Troubleshooting](#troubleshooting)

---

## Verification

### Test Core Functionality

```bash
# Test environment registry
python -c "from verses.registry import list_builtin; print(list_builtin())"

# Test agent registry
python -c "from agents.registry import list_agents; print(list_agents())"

# Run a quick training (should complete in <10 seconds)
python tools/train_agent.py --algo random --verse line_world --episodes 5 --max_steps 20
```

### Expected Results

✅ **Environment registry:** Should list 24 builtin verses
✅ **Agent registry:** Should list 25 agent types
✅ **Quick training:** Should complete without errors and save to `runs/` directory

---

## Optional Components

### GPU Support (CUDA)

**Already have PyTorch with CPU?** Uninstall and reinstall with CUDA:

```bash
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Verify GPU:**
```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Scaling Dependencies

For distributed training, streaming, and monitoring:

```bash
pip install -r requirements_scale.txt
```

**Includes:**
- Ray - Distributed computing
- Kafka - Event streaming
- Redis - Caching
- ChromaDB - Vector database
- Prometheus - Metrics
- And more...

**When to install:** Only if you need >4 workers or external services

---

## Docker Setup

For running the optional scaling stack (Kafka, Redis, ChromaDB, Prometheus, Grafana).

### Prerequisites

- Docker Desktop installed
- 8GB RAM available
- `requirements_scale.txt` dependencies installed

### Setup

1. **Configure environment:**
   ```bash
   cp .env.example .env
   nano .env  # Set GF_SECURITY_ADMIN_PASSWORD
   ```

2. **Start services:**
   ```bash
   docker compose -f docker-compose.scale.yml up -d
   ```

3. **Verify services:**
   ```bash
   docker compose -f docker-compose.scale.yml ps
   ```

### Service URLs

- **Redis:** localhost:6379
- **Kafka:** localhost:9092
- **ChromaDB:** http://localhost:8000
- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000 (login: admin / [your password])

### Stop Services

```bash
docker compose -f docker-compose.scale.yml down
```

**Clean up volumes:**
```bash
docker compose -f docker-compose.scale.yml down -v
```

---

## Troubleshooting

### Common Issues

#### 1. "ModuleNotFoundError: No module named 'X'"

**Cause:** Missing dependency

**Fix:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 2. "Manager() failed - bootstrapping phase error"

**Cause:** Windows multiprocessing issue (already fixed in codebase)

**Fix:** Ensure you have the latest code from main branch:
```bash
git pull origin main
```

#### 3. Tests fail with "CUDA out of memory"

**Cause:** GPU memory exhausted

**Fix:** Run tests on CPU only:
```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/ -q
```

#### 4. "GF_SECURITY_ADMIN_PASSWORD not set"

**Cause:** Docker Compose requires this variable

**Fix:** Set in `.env` file:
```bash
echo "GF_SECURITY_ADMIN_PASSWORD=mysecurepassword" >> .env
```

#### 5. Slow installation on Windows

**Cause:** Some packages require compilation

**Fix:** Install Visual C++ Build Tools:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

### Still Having Issues?

1. Check existing issues: https://github.com/wilker00/multiverse/issues
2. Create a new issue with:
   - Python version (`python --version`)
   - OS and version
   - Full error message
   - Steps to reproduce

---

## Next Steps

✅ **Installation complete!** Now what?

1. **[Quickstart Guide](QUICKSTART.md)** - Your first training session
2. **[Project Introduction](PROJECT_INTRO.md)** - Understand the architecture
3. **[Technical Paper](PAPER.md)** - Deep dive into algorithms
4. **[Contributing](../CONTRIBUTING.md)** - Make Multiverse better

---

## System Requirements Summary

### Minimum (for basic training)
- **CPU:** 2 cores
- **RAM:** 4GB
- **Disk:** 10GB
- **OS:** Windows 10+, Linux (Ubuntu 20.04+), macOS 11+

### Recommended (for serious development)
- **CPU:** 8+ cores
- **RAM:** 16GB
- **Disk:** 50GB SSD
- **GPU:** NVIDIA with 8GB+ VRAM
- **OS:** Linux (Ubuntu 22.04+)

### Production (for scaling infrastructure)
- **CPU:** 16+ cores
- **RAM:** 32GB+
- **Disk:** 100GB+ SSD
- **GPU:** NVIDIA with 24GB+ VRAM (multi-GPU supported)
- **Network:** 1Gbps+
- **Docker:** 8GB+ RAM allocated

---

**Questions?** See [FAQ](FAQ.md) or join the community discussions.
