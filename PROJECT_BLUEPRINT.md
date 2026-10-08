# MedViT-Edge: Project Blueprint for Claude Code

> **This document is the single source of truth for the MedViT-Edge project.**
> Claude Code: read this entire document before executing any task.
> User: Shahnawaz Shaikh (sshahnawaz567@gmail.com)
> Project start date: October 7, 2026

---

## 1. Project Identity

**Full name:** MedViT-Edge — Fine-Tuned Vision Transformer for Chest X-Ray Classification with Production Optimization

**One-line pitch:** An end-to-end medical image classification system that benchmarks fine-tuning strategies (full vs LoRA) and deployment optimizations (FP32 → FP16 → INT8 → TensorRT), deployed on Microsoft Azure with a live demo.

**What this project is NOT:**
- NOT a Kaggle competition entry
- NOT a research paper reimplementation
- NOT a from-scratch ViT implementation
- NOT focused on beating state-of-the-art accuracy

**What this project IS:**
- A production ML engineering demonstration
- A benchmark report comparing training and deployment tradeoffs
- A deployed, live, recruiter-accessible system
- An interview-defensible portfolio piece

---

## 2. Why This Project Exists (Resume Gap Analysis)

The user's existing resume is heavily agentic AI / LLM focused (RAG, LangGraph, multi-agent, GPT-4o re-ranking). It demonstrates ZERO of the following:

| Gap | How MedViT-Edge Fills It |
|---|---|
| From-scratch model training/fine-tuning | Full fine-tune ViT-Base on NIH ChestX-ray14 |
| Parameter-efficient fine-tuning (LoRA/PEFT) | LoRA fine-tune with explicit full-vs-LoRA comparison |
| Model optimization (quantization) | FP32 → FP16 → INT8 pipeline with calibration |
| ONNX ecosystem knowledge | PyTorch → ONNX export + ONNX Runtime inference |
| TensorRT / NVIDIA deployment stack | TensorRT engine compilation and benchmarking |
| Rigorous ML experiment tracking | Weights & Biases from day 1, every experiment logged |
| Production serving + load testing | FastAPI + Docker + Locust load tests |
| Azure cloud deployment | Container Apps + Static Web Apps + ACR + Blob Storage + App Insights |
| Medical imaging domain | Entire project is on chest X-rays |

**Target job families after completion:**
1. AI/ML Engineer (generalist)
2. Edge AI / Model Optimization Engineer
3. Healthcare AI Engineer
4. UAE/Gulf-region AI/ML roles (Azure emphasis)

---

## 3. The Workflow Contract (CRITICAL — Read Carefully)

The user is building this project **to learn, not to ship fast.** Standard Claude Code autonomous execution is NOT acceptable. Follow these rules strictly:

### 3.1 Stub Methodology

For every phase, Claude Code must leave 1-3 critical functions as `# TODO: user implements this` stubs. These are the functions the user wants to write themselves to internalize the concept. The stubs are specified per-phase in Section 7.

**Stub format:**
```python
def apply_lora_to_vit(model: nn.Module, rank: int, alpha: int, target_modules: list[str]) -> nn.Module:
    """
    Apply LoRA adapters to a Vision Transformer model.

    Args:
        model: The pre-trained ViT model
        rank: LoRA rank (controls capacity vs efficiency)
        alpha: LoRA scaling factor (typically 2x rank)
        target_modules: Which attention projection layers to adapt

    Returns:
        The model with LoRA adapters applied (base model frozen)

    Interview notes:
    - Why we freeze the base model: only ~0.3% of params train, dramatic memory savings
    - Why we target attention Q and V projections: strongest adaptation signal per param
    - Rank/alpha tradeoff: higher rank = more capacity but less efficient
    """
    # TODO: user implements this
    # Hints:
    # 1. Use peft.LoraConfig to define the configuration
    # 2. Use peft.get_peft_model to wrap the base model
    # 3. Verify trainable params with model.print_trainable_parameters()
    raise NotImplementedError("User to implement — see LEARNINGS.md for interview context")
```

### 3.2 Explanation Requirements

Every file Claude Code creates MUST include:
1. **Header comment** explaining what the file does and why it exists
2. **Inline comments** on non-obvious logic explaining the reasoning (not just mechanics)
3. **Interview Notes section** at the end: 2-3 questions an interviewer might ask about this code and model answers

### 3.3 User Learning Checkpoints

After each phase, the user writes `phases/phaseN/LEARNINGS.md` in their own words covering:
- What was built
- Top 3 technical decisions and why
- Interview-ready explanations

Claude Code does NOT write LEARNINGS.md. If asked, respond: "This file is for the user to write in their own words — it is their interview script."

### 3.4 Code Review Mode

When the user asks "explain this file" or "why does this work this way," respond as a teacher, not a code generator. Use analogies, cover tradeoffs, mention what interviewers care about.

---

## 4. Technical Stack (Locked)

### 4.1 Core ML
- **PyTorch 2.x** (CUDA 12.1 build for RTX 3050)
- **Hugging Face Transformers** — for pre-trained ViT-Base weights
- **Hugging Face PEFT** — for LoRA
- **Hugging Face Datasets** — for data pipeline (if using HF dataset variant)
- **torchvision** — for ResNet-50 baseline, image transforms
- **scikit-learn** — for metrics (per-class AUROC, confusion matrices)

### 4.2 Experiment Tracking
- **Weights & Biases (W&B)** — all runs logged, every metric, every hyperparameter

### 4.3 Optimization
- **ONNX** (via `torch.onnx.export`)
- **ONNX Runtime** — CPU inference baseline
- **NVIDIA TensorRT 10.x** — GPU inference optimization (RTX 3050 local)
- **PyTorch native quantization** — for comparison

### 4.4 Serving
- **FastAPI** — REST API
- **Pydantic** — request/response validation
- **Uvicorn** — ASGI server
- **Streamlit** — demo UI (Phase 7a / HF Spaces)
- **React + Vite** — demo UI (Phase 8b / Azure deployment)

### 4.5 Containerization & Deployment
- **Docker** — all services containerized
- **Hugging Face Spaces** — first deployment (Phase 7)
- **Microsoft Azure** (Phase 8):
  - Azure Container Registry (ACR)
  - Azure Container Apps (ACA)
  - Azure Static Web Apps (SWA)
  - Azure Blob Storage
  - Azure Application Insights

### 4.6 Load Testing & Monitoring
- **Locust** — HTTP load generation
- **Azure Application Insights** — production tracing

### 4.7 Development
- **Python 3.11**
- **uv** (preferred) or pip for dependency management
- **Ruff** — linting and formatting
- **pre-commit hooks** — enforce quality on commit
- **GitHub Actions** — CI/CD

---

## 5. Dataset

**Name:** NIH ChestX-ray14
**Source:** Kaggle — `nih-chest-xrays/data`
**Size:** 112,120 images, 30,805 patients, ~45 GB
**License:** Public domain
**Labels:** 14 binary disease classes + "No Finding"

### 5.1 Dataset Rules (Non-Negotiable)

1. **Use the official patient-level split file** provided with the dataset — never random split (data leakage).
2. **Full training runs on Kaggle** (free T4 x2, 30 hrs/week).
3. **LoRA + optimization experiments on local RTX 3050.**
4. **Local subset:** ~15-20 GB stratified sample for development.
5. **Known issues to document:** label noise (~10% error, auto-extracted via NLP), severe class imbalance (Hernia <0.2%, Infiltration ~17%).
6. **Baseline to beat:** CheXNet (Rajpurkar et al. 2017), mean AUROC ~0.84.

---

## 6. Hardware Reality

- **Local:** Laptop with NVIDIA RTX 3050 (6GB VRAM assumed; verify with `nvidia-smi`)
- **Cloud free:** Kaggle (T4 x2, 16GB each, 30 hr/week) + Colab (T4, limited)
- **TensorRT:** Local only (NVIDIA GPU required, hardware-specific engines)
- **No paid cloud GPU** — all inference deployment uses CPU with ONNX INT8

---

## 7. Phase-by-Phase Build Plan

Each phase ends with a measurable deliverable and a `LEARNINGS.md` the user writes.

### Phase 1: Baseline (Weeks 1-2) — ResNet-50 on ChestX-ray14

**Goal:** Establish a reproducible baseline matching published CheXNet performance (~0.82-0.84 mean AUROC).

**Deliverables:**
- `src/data/chestxray_dataset.py` — PyTorch Dataset
- `src/data/transforms.py` — training/validation image transforms
- `src/models/baseline_resnet.py` — ResNet-50 with 14-output multi-label head
- `src/training/train.py` — training loop with W&B logging
- `src/evaluation/metrics.py` — per-class AUROC, mAP, confusion matrices
- `configs/baseline.yaml` — all hyperparameters
- `notebooks/01_data_exploration.ipynb` — EDA with class distribution
- `phases/phase1/LEARNINGS.md` (user-written)
- W&B dashboard link in README

**User-implemented stubs:**
- `ChestXrayDataset.__getitem__` and `_load_labels` in `chestxray_dataset.py`
- `compute_class_weights()` in `train.py`
- Metric aggregation in `evaluation/metrics.py`

**Success criteria:** Mean AUROC ≥ 0.80 on official test split, logged to W&B.

### Phase 2: ViT Fine-Tuning Comparison (Weeks 3-4)

**Goal:** Train ViT-Base two ways (full fine-tune, LoRA) and produce a rigorous comparison.

**Deliverables:**
- `src/models/vit_classifier.py` — ViT-Base wrapper with multi-label head
- `src/models/vit_lora.py` — LoRA configuration + application
- `src/training/train_vit_full.py` — full fine-tune (runs on Kaggle)
- `src/training/train_vit_lora.py` — LoRA fine-tune (runs locally)
- `configs/vit_full.yaml`, `configs/vit_lora.yaml`
- `reports/phase2_comparison.md` — AUROC, training time, GPU memory, trainable params, convergence curves
- `phases/phase2/LEARNINGS.md`

**User-implemented stubs:**
- `apply_lora_to_vit()` in `vit_lora.py` (with hints)
- `freeze_backbone()` utility in `vit_classifier.py`
- Learning rate scheduler setup

**Success criteria:**
- Full fine-tune: mean AUROC ≥ 0.82
- LoRA (rank 8): mean AUROC within 2% of full fine-tune
- Comparison table with 5+ metrics, in reports/

### Phase 3: ONNX Export + Quantization (Week 5)

**Goal:** Export best model to ONNX, produce FP32, FP16, and INT8 variants.

**Deliverables:**
- `src/optimization/export_onnx.py` — PyTorch → ONNX
- `src/optimization/quantize_fp16.py` — FP16 quantization
- `src/optimization/quantize_int8.py` — INT8 with calibration dataset
- `src/optimization/calibration_data.py` — picks 500 representative validation images
- `src/optimization/verify_accuracy.py` — compares outputs across variants on test set
- `reports/phase3_quantization.md` — accuracy delta per variant, file sizes
- `phases/phase3/LEARNINGS.md`

**User-implemented stubs:**
- `CalibrationDataReader` class in `calibration_data.py` (critical — must understand)
- Opset version selection logic in `export_onnx.py`
- Accuracy delta computation

**Success criteria:**
- Three ONNX models produced
- INT8 model accuracy drop ≤ 2% from FP32
- All variants verified to produce same predictions on 100 sample images (within tolerance)

### Phase 4: TensorRT Conversion + Benchmarking (Week 6)

**Goal:** Produce TensorRT engines for RTX 3050 and benchmark rigorously.

**Deliverables:**
- `src/optimization/build_trt_engine.py` — ONNX → TensorRT engine
- `src/benchmarking/benchmark_runner.py` — latency harness (warmup + measurement)
- `src/benchmarking/stats.py` — p50/p95/p99 computation
- `reports/phase4_benchmarks.md` — THE headline table (PyTorch FP32, ONNX FP32, ONNX FP16, ONNX INT8, TensorRT FP16, TensorRT INT8)
- `phases/phase4/LEARNINGS.md`

**User-implemented stubs:**
- Full `benchmark_runner.py` — the user writes this entirely (critical for interview)
- INT8 calibrator for TensorRT

**Benchmark methodology (non-negotiable):**
- 100 warmup iterations
- 1000 measurement iterations
- Fixed batch size (compare at batch=1 and batch=8)
- Report p50, p95, p99 (not just mean)
- Report throughput (images/sec)
- Report model file size
- Report GPU memory used at inference

**Success criteria:** TensorRT INT8 achieves ≥ 3x speedup over PyTorch FP32 with ≤ 2% accuracy drop. Full benchmark table published.

### Phase 5: FastAPI + Docker (Week 7)

**Goal:** Production-grade API serving the best optimized model.

**Deliverables:**
- `src/api/main.py` — FastAPI app
- `src/api/models.py` — Pydantic request/response schemas
- `src/api/inference.py` — model loading, preprocessing, prediction
- `src/api/health.py` — health check endpoints
- `Dockerfile` — multi-stage build
- `docker-compose.yml` — local development
- `tests/test_api.py` — API tests
- `phases/phase5/LEARNINGS.md`

**User-implemented stubs:**
- `preprocess_image()` function (resize, normalize, tensorify)
- `/predict` endpoint handler

**Success criteria:**
- Docker image builds and runs
- `/predict` accepts image upload, returns JSON with 14 disease probabilities
- `/health` endpoint for liveness/readiness
- Image size ≤ 2GB

### Phase 6: Streamlit Frontend (Days 1-3 of Week 8)

**Goal:** Minimal demo UI for recruiters to upload X-rays and see predictions.

**Deliverables:**
- `frontend/streamlit_app.py` — upload + display UI
- `frontend/sample_images/` — 5-10 example X-rays with known labels
- `phases/phase6/LEARNINGS.md`

**Minimal spec:**
- File upload widget
- Prediction display with probability bars
- Example images for recruiters who don't have X-rays
- Model info sidebar (which variant is running, latency stats)

**Success criteria:** Runs locally, connects to FastAPI backend, loads in <3 seconds.

### Phase 7: Hugging Face Spaces Deployment (Days 4-5 of Week 8)

**Goal:** Live demo link ASAP.

**Deliverables:**
- `huggingface/Dockerfile` or Space config
- Space URL in README
- Deployment runs INT8 ONNX model on free CPU tier
- `phases/phase7/LEARNINGS.md`

**Success criteria:** Public URL that a recruiter can click and use within 60 seconds.

### Phase 8: Azure Deployment (Weeks 9-10) — THE RESUME GOLD

**Goal:** Full Azure deployment demonstrating cloud architecture competence (Gulf/UAE market).

**Deliverables:**
- `azure/bicep/` — infrastructure-as-code templates
- `azure/github-actions/deploy.yml` — CI/CD pipeline
- `azure/README.md` — architecture diagram + resource descriptions
- `frontend-react/` — React frontend for SWA (replaces Streamlit)
- Azure Application Insights dashboard screenshots
- `phases/phase8/LEARNINGS.md`

**Azure services used:**
1. Azure Container Registry (ACR) — Docker image hosting
2. Azure Container Apps (ACA) — serverless backend (scales to zero)
3. Azure Static Web Apps (SWA) — React frontend
4. Azure Blob Storage — ONNX model artifacts
5. Azure Application Insights — distributed tracing, logs, metrics

**User-implemented stubs:**
- Bicep template for Container Apps
- GitHub Actions workflow for ACR push + ACA deploy

**Success criteria:**
- Public Azure URL working
- Cold start < 3s, warm inference < 800ms (CPU inference)
- Monitoring dashboard with request counts and latency
- All resources within free tier limits

### Phase 9: Load Testing + Final Write-up (Week 11)

**Goal:** Prove the system works under load and document everything.

**Deliverables:**
- `loadtest/locustfile.py` — Locust scenarios
- `reports/phase9_loadtest.md` — p50/p95/p99 under 10, 50, 100 concurrent users
- Updated README with:
  - Hero image / architecture diagram
  - Live demo links (HF Spaces + Azure)
  - Benchmark table
  - Technology stack
  - Interview-ready quick-reference
- Technical blog post (Medium or user's portfolio) covering:
  - LoRA vs full fine-tune tradeoff
  - Quantization journey (FP32 → INT8)
  - TensorRT speedup story
  - Azure deployment architecture
- `phases/phase9/LEARNINGS.md` — consolidated interview script

**Success criteria:** README is self-explanatory to a recruiter in under 60 seconds. Blog post exists and is linked.

---

## 8. Resume Bullets (Target Output)

After project completion, the user should be able to add to resume:

```
MedViT-Edge — Production Chest X-ray Classification System         GitHub | Live Demo | Blog
- Fine-tuned ViT-Base on NIH ChestX-ray14 (112K images, 14 diseases) comparing full
  fine-tune (86M params) vs LoRA (300K params); LoRA achieved AUROC within 1.5% of
  full fine-tune using 95% less GPU memory
- Optimized inference via PyTorch native quantization, ONNX Runtime, and NVIDIA
  TensorRT (FP32 → INT8); delivered 4.2x speedup (142ms → 34ms p95) with 1.1% AUROC
  drop on RTX 3050
- Deployed serverless on Microsoft Azure: Container Apps backend, Static Web Apps
  frontend, Container Registry, Blob Storage, Application Insights monitoring
- Load tested with Locust: sustained 50 req/s with p99 <800ms on free-tier CPU inference
- Tech: PyTorch, Hugging Face PEFT, ONNX, TensorRT, FastAPI, Docker, Azure, W&B, GitHub Actions
```

---

## 9. Repository Structure (Target)

```
medvit-edge/
├── README.md                      # Hero README with live links
├── PROJECT_BLUEPRINT.md           # This document
├── ideas-for-later.md             # Parking lot
├── LICENSE                        # MIT
├── pyproject.toml                 # or requirements.txt
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       ├── ci.yml                 # tests + lint
│       └── deploy-azure.yml
├── configs/
│   ├── baseline.yaml
│   ├── vit_full.yaml
│   └── vit_lora.yaml
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── optimization/
│   ├── benchmarking/
│   └── api/
├── frontend/                      # Streamlit (Phase 7)
├── frontend-react/                # React (Phase 8)
├── notebooks/                     # EDA + exploration
├── tests/
├── loadtest/
├── azure/
│   ├── bicep/
│   └── README.md
├── huggingface/
├── reports/                       # Published write-ups per phase
├── phases/                        # LEARNINGS.md per phase (user-authored)
└── docker/
    ├── Dockerfile.api
    └── Dockerfile.frontend
```

---

## 10. Working Agreement (Claude Code ↔ User)

1. **Never auto-execute a full phase.** Always check in after each file created.
2. **Never skip the stub methodology.** The user NEEDS to write certain functions.
3. **Never write LEARNINGS.md files.** Those are the user's.
4. **Never introduce dependencies not in Section 4** without asking.
5. **Never silently fix accuracy issues.** Flag them to the user.
6. **Always include Interview Notes.** Non-negotiable.
7. **Always log to W&B.** Every training run.
8. **When uncertain, ask the user before building.** Especially for architectural decisions.
9. **Reference this document.** When the user asks "what are we building next," look here first.
10. **No scope creep.** If the user suggests a new feature not in this plan, respond: "Should we add this to ideas-for-later.md and continue the current phase?"

---

## 11. Current Status

**Phase:** 0 (Planning complete)
**Next up:** Phase 1 — ResNet-50 Baseline
**Blockers:** User to complete pre-project checklist:
- [ ] Kaggle account + phone verification (unlocks free GPU)
- [ ] W&B account
- [ ] Verify `nvidia-smi` works on laptop
- [ ] PyTorch with CUDA installed
- [ ] 50 GB free disk space confirmed
- [ ] Project folder location decided

---

*End of blueprint. Last updated: October 7, 2026.*
