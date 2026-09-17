# 🏗️ AI-Powered Concrete Structural Damage Inspection & Automated Engineering Reporting System

> An enterprise-grade, end-to-end computer vision and deep learning platform designed to automate civil structural health monitoring, crack detection, segmentation, and instant engineering compliance report generation.

---

## 🛡️ Badges
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Async-005571.svg)
![React](https://img.shields.io/badge/React-Vite-61DAFB.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🎬 3. Demo / Screenshots / Video
*(Add your interactive dashboard GIFs, architecture screenshots, or video walkthrough links here)*
- **Interactive Inspection Dashboard:** Real-time upload, multi-model inference toggle, and live severity grading.
- **Automated PDF Report Generator:** Instant download of engineering compliance certificates.

---

## 📋 4. Overview
Manual civil structural inspection is often slow, subjective, and prone to human oversight. This project introduces an automated computer vision ecosystem that bridges deep learning with civil engineering workflows. By combining classification, object detection, and precise segmentation, the system not only identifies structural anomalies in concrete surfaces but also computes critical physical parameters such as maximum crack width and severity indexing.

---

## 🎯 5. Problem Statement
Civil infrastructure degradation (spalling, scaling, and heavy cracking) threatens public safety. Traditional inspection relies on visual logging, which lacks pixel-level precision and quantitative severity indexing. Engineers require an automated, reproducible, and rapid diagnostic tool to evaluate concrete structures from standard field photographs.

---

## 💡 6. Solution
An integrated multi-stage pipeline:
1. **Filter & Classify:** ResNet50 filters non-damaged patches to optimize computational resources.
2. **Detect & Localize:** YOLOv8 bounds specific regions of damage.
3. **Segment & Quantify:** U-Net isolates exact crack pixels, feeding a mathematical engine (`severity.py`) that calculates physical width and compliance metrics.
4. **Report:** FastAPI and a React dashboard serve the results instantly with downloadable PDF certificates.

---

## ✨ 7. Key Features
- **Multi-Model Intelligence:** ResNet50 (Classification), YOLOv8 (Detection), and U-Net (Segmentation).
- **Physical Engineering Metrics:** Automated calculation of crack length, area, and severity index.
- **Asynchronous Backend:** High-performance FastAPI server supporting concurrent inspections.
- **Interactive React Dashboard:** Modern UI built with Vite, Tailwind CSS, and Three.js-ready components.
- **One-Click PDF Reporting:** Generates professional structural audit sheets on demand.

---

## 🏛️ 8. Architecture
```text
[ Client / Browser (React Dashboard) ]
                │
                ▼ (HTTP POST /inspect/full)
[ FastAPI Asynchronous Backend Engine ]
        ├──> ResNet50 (Classification: Validates Damage)
        ├──> YOLOv8   (Object Detection: Localizes Regions)
        └──> U-Net    (Pixel Segmentation & Severity Engine)
                │
                ▼
[ Automated PDF Compliance Report & JSON Response ]
🛠️ 9. Tech StackDeep Learning / CV: PyTorch, Ultralytics YOLOv8, Segmentation Models PyTorch, OpenCV, NumPy, Pandas, ONNX Runtime.Backend: Python, FastAPI, Uvicorn, ReportLab.Frontend: React, Vite, Tailwind CSS, Lucide Icons.DevOps & Tools: Docker, Docker Compose, Git, Vercel.📂 10. Project StructurePlaintextD:\nti_project\
├── src/
│   ├── backend/          # FastAPI application & endpoints
│   ├── engine/           # Severity calculation & physical metrics logic
│   └── frontend/         # React SPA dashboard components
├── models/               # Model metadata & configuration schemas
├── notebooks/            # EDA, Audit, and Training Jupyter Notebooks
├── .gitignore            # Strict exclusion rules for heavy assets
├── docker-compose.yml    # Multi-container orchestration
└── DEPLOYMENT.md         # Detailed deployment guidelines
⚙️ 11. How It Works / WorkflowUser uploads a concrete image via the React Dashboard.FastAPI receives the payload and passes the image through the Classification gate.If damaged, the image is processed by YOLOv8 for bounding box localization and U-Net for pixel-level mask generation.The Severity Engine computes physical metrics, returning JSON payloads and rendering a downloadable PDF report.📦 12. InstallationClone the repository and set up the local environment:Bashgit clone [https://github.com/yossef1907/concrete-structural-damage-inspection.git](https://github.com/yossef1907/concrete-structural-damage-inspection.git)
cd concrete-structural-damage-inspection

# Setup backend environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\Activate.ps1
pip install -r src/backend/requirements.txt
🔐 13. Configuration / Environment VariablesCreate a .env file in the root or src/backend/ directory:Code snippetPORT=8000
HOST=0.0.0.0
DEBUG_MODE=False
MODEL_CONFIDENCE_THRESHOLD=0.5
🚀 14. UsageStart the FastAPI backend server:Bashuvicorn src.backend.main:app --reload --port 8000
Start the Frontend Dashboard:Bashcd src/frontend
npm install
npm run dev
🔌 15. API DocumentationGET /health: Verifies system and model readiness.POST /inspect/full: Uploads an image and returns multi-model detection, segmentation masks, and severity metrics.🧠 16. AI/ML DetailsClassification: ResNet50 fine-tuned with optimal F1-thresholding ($0.05$).Detection: YOLOv8s optimized for structural bounding box regression.Segmentation: U-Net with ResNet34 backbone featuring small-blob morphological filtering to eliminate false positives.📊 17. DatasetSourced from benchmark structural concrete inspections, cleaned, audited, and split using unsealed protocol to prevent data leakage across Train/Val/Test cohorts.📈 18. Model Performance / EvaluationYOLOv8 Detection: mAP50 = 67.03% | Precision = 61.94% | Recall = 68.81%U-Net Segmentation: Dice Score = 93.21% | IoU = 93.13%📋 19. ResultsHigh-fidelity crack isolation with zero data leakage, providing reliable structural health indices suitable for municipal compliance audits.🖼️ 20. Example Input / OutputInput: Raw photograph of a concrete bridge pier with hairline cracking.Output: Bounding boxes, binary segmentation mask overlay, severity classification (Moderate/Severe), and a PDF audit certificate.🧪 21. TestingRun backend unit and integration tests:Bashpytest tests/
🚢 22. DeploymentDeploy using Docker Compose:Bashdocker-compose up --build -d
🔒 23. Security / PrivacyInput imagery is processed ephemerally; strict input sanitization prevents path traversal and malicious payload execution.⚠️ 24. LimitationsPerformance may degrade under extreme shadow interference or severely uncalibrated camera angles.🔮 25. Future ImprovementsUpgrade YOLOv8 to Medium/Large variants and incorporate transformer-based vision backbones (Swin Transformer) for enhanced multi-scale crack tracking.🗺️ 26. Roadmap[x] Core Vision Pipelines (Classification, Detection, Segmentation)[x] FastAPI Backend & React Dashboard[ ] Mobile App Field Integration (Flutter / React Native)[ ] Cloud Scalability via Kubernetes👥 27. ContributingContributions are welcome! Please fork the repository and submit a pull request.📝 28. LicenseDistributed under the MIT License. See LICENSE for more information.✍️ 29. AuthorsYossef Ayman Nasef - Software Engineering Student & AI Developer - GitHub Profile🙏 30. Acknowledgments / ReferencesUltralytics YOLOv8 FrameworkSegmentation Models PyTorch (SMP)NTI Project Supervisors & Open Source CV Community
