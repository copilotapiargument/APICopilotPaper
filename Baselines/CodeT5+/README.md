# Argument Completion with CodeT5+

Predicting missing API arguments using CodeT5+. It integrates preprocessing, argument extraction, and fine-tuning of CodeT5+ to provide accurate and context-aware argument recommendations.
```
APICopilot/
├── ARExtraction/
├── codet5/
├── Main.py
├── requirements.txt
└── README.md
```
---

## Features

- **Preprocessing**: Supports Eclipse, NetBeans, and PY150 datasets.
- **Argument Extraction**: Extracts Argument Requests (ARs) from code files.
- **CodeT5+ Fine-Tuning**: Fine-tunes CodeT5+ for API argument completion.
- **Prediction**: Predicts missing arguments using the fine-tuned model.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/APICopilot.git
   cd APICopilot
   ```
Install dependencies:
```
pip install -r requirements.txt
```
