# 🏗️ DentaVision System Architecture

This document outlines the end-to-end technical flow of the DentaVision platform, based on the **STS-Tooth (2025)** research methodology.

## 1. High-Level Overview
DentaVision is a modular platform that processes 2D and 3D dental imaging through an AI-inference pipeline to produce clinical insights.



### Data Pipeline
1. **User Ingestion**: Clinicians upload DICOM files via the React frontend.
2. **PII Sanitization**: Node.js/FastAPI strips metadata (Names, IDs) to ensure HIPAA compliance.
3. **MONAI Preprocessing**: Images are resampled to 0.3mm isotropic resolution and normalized.
4. **AI Inference**: The R2 U-Net identifies tooth boundaries and detects pathologies.
5. **Persistence**: Masks and FDI-numbered labels are stored in Supabase PostgreSQL.

---

## 2. Core AI Component (R2 U-Net)
The primary segmentation engine is a **Recurrent Residual U-Net** optimized for medical boundary accuracy.



| Layer Type | Configuration | Purpose |
| :--- | :--- | :--- |
| **Encoder** | 4-Level Depth | Global feature extraction. |
| **Recurrent Block** | $t=2$ Iterations | Pixel-level edge refinement. |
| **Residual Block** | Identity Shortcut | Prevents gradient vanishing in 3D. |
| **Output** | Softmax/Sigmoid | Probability maps for tooth presence. |

---

## 3. Database Schema
We utilize **Supabase (PostgreSQL)** for its reliability and `pgvector` capabilities.

* **Patients**: `id (UUID)`, `age_group (ENUM)`, `created_at`.
* **Scans**: `id`, `patient_id`, `type (PXI/CBCT)`, `storage_url`.
* **Detections**: `id`, `scan_id`, `tooth_number (FDI)`, `confidence_score`.

---

## 4. Resource Efficiency Strategies
- **Quantization**: FP16 inference for reduced VRAM usage.
- **Sliding Window**: 3D CBCTs are processed in $96^3$ patches to avoid OOM crashes.
- **Persistent Cache**: Pre-computed transforms are cached to disk to save CPU cycles.
