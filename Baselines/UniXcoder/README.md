# Argument Completion with UniXcoder

Predicting missing API arguments using UniXcoder. It integrates preprocessing, argument extraction, and fine-tuning of UniXcoder to provide accurate and context-aware argument recommendations.

---
Directory Structure
```
Main/
├── preprocessing/
├── extraction/
├── unixcoder/
├── main.py
├── requirements.txt
└── README.md
```
## Features

- **Preprocessing**: Supports Eclipse, NetBeans, and PY150 datasets.
- **Argument Extraction**: Extracts Argument Requests (ARs) from code files.
- **UniXcoder Fine-Tuning**: Fine-tunes UniXcoder for API argument completion.
- **Prediction**: Predicts missing arguments using the fine-tuned model.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/APICopilot.git
   cd APICopilot
   ```
