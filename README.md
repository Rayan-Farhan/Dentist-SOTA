# DentaVision: Multi-Modal SOTA Dental Diagnostic Platform

[![FAST-NUCES](https://img.shields.io/badge/FAST--NUCES-FYP--2026-blue)](https://www.nu.edu.pk/)
[![Research](https://img.shields.io/badge/Based--on-STS--Tooth--2025-green)](https://doi.org/10.1038/s41597-024-04306-9)
[![Framework](https://img.shields.io/badge/Powered--by-MONAI--PyTorch-red)](https://monai.io/)

**DentaVision** is a clinical-grade implementation of the **STS-Tooth (2025)** research paper. It provides a resource-efficient, full-stack ecosystem for automated tooth segmentation and pathology detection using **R2 U-Net** on both 2D Panoramic X-rays (PXI) and 3D Cone Beam Computed Tomography (CBCT).

---

## Research Foundation & Dataset Credits

This project is a technical realization of the methodologies described in:
> **Paper:** *"A multi-modal dental dataset for semi-supervised deep learning image segmentation"* > **Authors:** Yaqi Wang, Fan Ye, et al. (Published Jan 2025, *Scientific Data*)  
> **Dataset:** **STS-Tooth** (4,000 PXIs and 148,400 CBCT scans).  

We strictly adhere to the paper's **Data Preprocessing** standards and the **ANSI/ADA Standard No. 1110-1:2025** for medical AI validation.

---

## Project Architecture

DentaVision uses a **"Heavy-Edge, Light-Cloud"** approach to ensure HIPAA/GDPR compliance and low-latency inference in clinical settings.



### Technical Stack
* **AI Engine:** PyTorch + MONAI (Recurrent Residual U-Net).
* **Backend:** FastAPI (Python) + Node.js (Orchestrator).
* **Frontend:** React (Next.js) + Cornerstone.js 3D (DICOM Viewer).
* **Database:** Supabase (PostgreSQL + pgvector).
* **Infrastructure:** Docker + Heroku + GitHub Actions.

---
## Getting Started

### 1. Prerequisites
* Docker & Docker Compose
* NVIDIA GPU (Recommended) with CUDA 12.x
* Python 3.10+ & Node.js 18+

### 2. Environment Setup
Create a .env file in the root directory:

SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
MODEL_PATH=./ai/weights/r2unet_sota.pth

### 3. Installation & Run

# Clone the repo
git clone https://github.com/your-org/DentaVision.git
cd DentaVision

# Run with Docker (Includes FastAPI, Node, and React)
docker-compose up --build

---

## Implementation Specifications

### AI Model (R2 U-Net)
* Structure: 4-level Encoder-Decoder with Recurrent Residual Blocks.
* Iterations ($t$): $t=2$ for optimal precision-latency balance.
* 3D Strategy: MONAI Sliding Window Inference (96 x 96 x 96 patches) to prevent OOM errors on CBCT volumes.

### Data Engineering Standards
* Isotropic Resampling: All 3D volumes normalized to 0.3mm voxel spacing.
* Intensity Normalization: Z-score normalization for Hounsfield Units (HU).
* Anonymization: Mandatory stripping of DICOM tags (0010,0010) through (0010,0040) before storage.

---

## The Team (FAST-NUCES)
* Bilal: AI/CV Engineer (Model Architecture & Training).
* Rayan: Database Architect & Data Engineer (Supabase & PII Stripping).
* Sheharyar: Frontend Lead (UI/UX & Cornerstone.js Integration).
* Mustafa: Cloud & DevOps (Heroku Deployment & Dockerization).
* Anas: Backend & Full-stack (FastAPI & API Gateway).
* Mujtaba: AI & Backend Integration (FastAPI & Inference Optimization).

---

## Compliance & Ethics
This project implements Privacy-by-Design. No PII (Personally Identifiable Information) is stored. All evaluations are measured against the Sørensen–Dice Coefficient ($DSC$) and Boundary Accuracy as per STS-Tooth benchmarks.

## License
* Code is licensed under the MIT License.
* The STS-Tooth Dataset is used under CC-BY-NC-ND 4.0 for academic research.
##  Directory Structure

```text
DentaVision/
├── .github/                # CI/CD Workflows & PR Templates
├── ai/                     # [Bilal/Mujtaba] Model R&D
│   ├── models/             # R2U-Net & VLM Architectures
│   ├── transforms/         # MONAI Preprocessing & Augmentation
│   └── trainers/           # Training scripts (Distributed Data Parallel)
├── backend/                # [Anas/Mujtaba] FastAPI Services
│   ├── api/                # Endpoints for Inference & Reports
│   └── core/               # PII Anonymization & DICOM Parsing
├── frontend/               # [Sheharyar/Mustafa] React Dashboard
│   ├── components/         # Cornerstone.js & VTK.js viewers
│   └── hooks/              # Supabase Auth & State Management
├── data_engineering/       # [Rayan] Dataset Management
│   ├── scripts/            # DICOM to NIfTI Conversion
│   └── dvc/                # Data Version Control meta-files
├── docker/                 # Deployment configurations
└── README.md               # You are here

