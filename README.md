# Product Classification with Linear SVC

Machine-learning system for automatic product classification based on a **Linear SVC model
with 95.7% accuracy**, using a hybrid approach to handle brand-specific edge cases.

## Overview

Classifying products into the correct category from free-text product descriptions is a
classic real-world problem in e-commerce and inventory management. This project builds a
complete ML pipeline — from raw product data to a trained, evaluated classifier.

- **Model:** Linear Support Vector Classifier (scikit-learn)
- **Accuracy:** 95.7% on the test set
- **Special feature:** Hybrid approach for brand-specific cases that the base model
  handles poorly (rule-based fallback / brand-specific refinement layer)
- **Deliverable:** Interactive HTML report with results and evaluation

## Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/-pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![HTML](https://img.shields.io/badge/-HTML%20Report-E34F26?style=flat-square&logo=html5&logoColor=white)

## Pipeline

1. **Data preparation** — loading & cleaning the product dataset
2. **Feature extraction** — TF-IDF vectorization of product descriptions
3. **Model training** — Linear SVC with tuned hyperparameters
4. **Hybrid layer** — brand-specific refinement for known problem cases
5. **Evaluation** — accuracy, precision/recall per class, confusion matrix
6. **Reporting** — automated HTML report with all results

## Results

| Metric | Value |
|---|---|
| Overall accuracy | **95.7%** |
| Approach | Linear SVC + hybrid brand-specific handling |

*(Confusion matrix and per-class metrics: see the generated HTML report.)*

## Project Structure

```
product-classification/
├── notebooks/            # Jupyter notebooks: EDA, training, evaluation
├── data/                 # product dataset (not included)
├── report/               # generated HTML report
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/miroslavjovanovicsrb/product-classification-task.git
cd product-classification-task
pip install -r requirements.txt
jupyter notebook
```

## What I Learned

- Choosing and tuning a linear classifier for high-dimensional text data (TF-IDF)
- Handling class imbalance and per-class precision/recall trade-offs
- Building a hybrid system: when a pure ML model hits its limits and a targeted
  rule-based layer adds real value — without retraining the whole pipeline

## Author

**Miroslav Jovanovic** — Python Backend & AI Engineer
[LinkedIn](https://www.linkedin.com/in/miroslav-jovanovic-python) ·
[GitHub](https://github.com/miroslavjovanovicsrb)
