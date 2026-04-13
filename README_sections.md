# DS 4320 Project 2: Credit Card Fraud Detection

**Name:** [Your Name]  
**NetID:** [Your NetID]  
**DOI:** [Your DOI]  
**License:** [License Name] — [LICENSE](./LICENSE)  
**Pipeline:** [pipeline.ipynb](./pipeline.ipynb) | [pipeline.md](./pipeline.md)  
**Press Release:** [Press Release](#press-release)

---

## Executive Summary

This repository contains the full materials for DS 4320 Project 2, which develops a document-oriented database and machine learning pipeline to detect credit card fraud. A secondary dataset was constructed in MongoDB Atlas using anonymized transaction records, enriched with behavioral and merchant metadata. The analysis pipeline, implemented in a Jupyter Notebook, trains a binary classification model to distinguish fraudulent transactions from legitimate ones and visualizes model performance under severe class imbalance.

---

## Problem Definition

### Problem Statement

**General problem:** Detecting credit card fraud.

**Specific problem:** Given a dataset of anonymized credit card transactions, build a document-oriented data model in MongoDB that captures transaction-level features — including amount, time delta, merchant category, and cardholder behavioral patterns — and use it to train a binary classification model that distinguishes fraudulent transactions from legitimate ones, with particular emphasis on handling severe class imbalance (typically <0.2% fraud rate).

### Motivation

Credit card fraud costs the global economy tens of billions of dollars annually, and the harm falls disproportionately on individual cardholders who face delayed dispute resolution, temporarily frozen accounts, and eroded trust in digital payment systems. Traditional rule-based fraud detection systems — which flag transactions based on static thresholds like unusual geography or large amounts — struggle to adapt to evolving fraud tactics, creating a persistent gap between how fraudsters operate and how financial institutions respond. A machine learning pipeline grounded in a well-structured document database can capture richer contextual signals per transaction, including historical spending behavior, merchant metadata, and time-of-day patterns, and surface anomalies that flat tabular systems routinely miss. Improving fraud detection accuracy even marginally at scale translates to millions of dollars in recovered value and meaningfully better outcomes for cardholders.

### Rationale for Refinement

The general problem of "detecting credit card fraud" encompasses a wide range of technical approaches: real-time streaming detection at point of sale, graph-based network anomaly detection across cardholder relationships, and post-hoc batch classification on historical records, among others. This project narrows to **batch binary classification on anonymized transaction records** for two reasons. First, publicly available benchmark datasets — most notably the ULB Credit Card Fraud Detection dataset from Kaggle — are structured around this paradigm, making it feasible to build a fully documented, reproducible secondary dataset with traceable provenance. Second, the document model in MongoDB is particularly well-suited to transaction records, since each document can embed cardholder context, merchant attributes, and computed behavioral features in a single nested structure — something a normalized relational schema handles awkwardly and inefficiently. The refinement trades real-time operational complexity for depth of feature engineering, model interpretability, and alignment with the document model requirements of this project.

### Press Release

> **One in Every Thousand Transactions Is Fraud — Here's How Machine Learning Catches It**

[→ Full Press Release](./press_release.md)

---

## Domain Exposition

### Terminology

| Term | Definition |
|------|-----------|
| **Fraud** | A transaction initiated without the cardholder's authorization, typically after card data theft or account compromise |
| **Class imbalance** | A condition where one class (fraud) is vastly underrepresented relative to the other (legitimate), often <0.2% of records |
| **PCA features** | Principal Component Analysis-transformed features; in the ULB dataset, V1–V28 are PCA projections of original (confidential) transaction features |
| **False positive (FP)** | A legitimate transaction incorrectly flagged as fraud; results in customer friction and declined purchases |
| **False negative (FN)** | A fraudulent transaction incorrectly classified as legitimate; results in financial loss |
| **Precision** | Of all transactions flagged as fraud, the fraction that are truly fraudulent: TP / (TP + FP) |
| **Recall (Sensitivity)** | Of all actual fraud cases, the fraction correctly identified: TP / (TP + FN) |
| **F1 Score** | Harmonic mean of precision and recall; preferred metric when class imbalance is severe |
| **AUC-ROC** | Area Under the Receiver Operating Characteristic Curve; measures model discrimination ability across all classification thresholds |
| **SMOTE** | Synthetic Minority Oversampling Technique; generates synthetic fraud samples to address class imbalance during training |
| **Chargeback** | The reversal of a transaction initiated by the card issuer on behalf of the cardholder after fraud is confirmed |
| **Authorization** | The real-time approval/denial decision made by the card network at point of sale |
| **Transaction velocity** | The rate at which a cardholder makes purchases within a short time window; a common fraud signal |
| **Merchant Category Code (MCC)** | A 4-digit code classifying the merchant type (e.g., grocery, airline, ATM); relevant to fraud profiling |
| **Soft-schema document** | A MongoDB document without a rigidly enforced field structure, allowing variable fields across records |

### Domain Overview

This project lives at the intersection of **financial technology (fintech)**, **fraud analytics**, and **applied machine learning**. Credit card fraud detection is one of the most well-studied problems in industry data science, with every major card network (Visa, Mastercard, American Express) and issuing bank operating proprietary real-time scoring systems. The domain is characterized by extreme class imbalance, where fewer than 2 in every 1,000 transactions are fraudulent, meaning naive classifiers achieve high accuracy by simply predicting "not fraud" for everything. Practitioners therefore prioritize recall (catching actual fraud) and F1 score over raw accuracy, and employ resampling strategies such as SMOTE or cost-sensitive learning to train models that do not trivially collapse to the majority class. The document model is well-suited here because real-world fraud detection systems aggregate heterogeneous signals per transaction — merchant metadata, device fingerprints, behavioral baselines, geographic data — that do not fit neatly into a single flat table. MongoDB's flexible schema allows each transaction document to carry nested cardholder history and merchant context alongside the core transaction fields, mirroring the structure of production fraud scoring systems.

### Background Reading

| Title | Description | Link |
|-------|-------------|------|
| Dal Pozzolo et al. (2015), *Calibrating Probability with Undersampling for Unbalanced Classification* | Foundational paper on handling class imbalance in fraud detection; introduces probability calibration techniques | [/readings/dalpozzolo_2015.pdf](./readings/dalpozzolo_2015.pdf) |
| Bhattacharyya et al. (2011), *Data Mining for Credit Card Fraud: A Comparative Study* | Comparative analysis of decision trees, logistic regression, and SVMs for fraud detection | [/readings/bhattacharyya_2011.pdf](./readings/bhattacharyya_2011.pdf) |
| Kaggle ULB Credit Card Fraud Dataset — Description Page | Explains the provenance of the benchmark dataset used in this project, including PCA transformation rationale | [/readings/kaggle_dataset_description.pdf](./readings/kaggle_dataset_description.pdf) |
| Chawla et al. (2002), *SMOTE: Synthetic Minority Over-sampling Technique* | Original SMOTE paper describing the oversampling method used to address class imbalance in training | [/readings/chawla_smote_2002.pdf](./readings/chawla_smote_2002.pdf) |
| Federal Reserve Payments Study (2023), *Fraud in Noncash Payments* | Government report quantifying the scale of payment card fraud in the U.S.; provides real-world motivation | [/readings/fed_payments_fraud_2023.pdf](./readings/fed_payments_fraud_2023.pdf) |

---
