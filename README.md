# DS 4320 Project 2: Credit Card Fraud Detector

This project builds a MongoDB dataset with the aim of detecting credit fraud. 

**Name:** Rebecca Vanni

**NetID:** ecn2wh

**DOI:** [10.5281/zenodo.XXXXXXX](https://zenodo.org/records/19341763)

**Press Release:** → [press_release.md](/press_release/press_release.md)

**Pipeline:** → [View Pipeline](pipeline/)

**License:** [MIT License](./LICENSE)



## Problem Definition 

### General Problem:

Detecting credit card fraud.

### Specific Problem:

Detecting credit card fraud is very difficult because there is a severe class imbalance. However, if a credit card company fails to recognize fraud it is very costly and difficult to deal with. If they overpredict fraud then they risk turning valued customers away and potentially put them in risky situations. The challenge being given the amount, time, merhcant category, and behavior of cardholder is the transaction fraud and at what point does the company need to decline a transaction.   

### Rationale for Refinement:

Credit card fraud costs hundreds of thousands of people their own money. User's expierence enjoyment and trust is eroded. They could face lengthy disputes and potential asset freezing. However, the company needs to protect themselves from fraud and ensure low expenditures. As time and access to technology has expanded, there has been a greater ened for detection services. Evolving fraud patterns create a need for a machine learning pipeline which can adapt to these changes.

### Motivation for Project:

Detecting credit card fraud is a very broad statement on a larger issue. There are many different approaches (real-time screening detection, post-hoc batch classification, etc) to detecting fraud. Building a secondary dataset which has proper provenance enables a combined approach. Through using MongoDB each document has cardholder context, merhcant attributes, and computed behavioral features in one single nexted record. Relational schema is unable to due as well. The refinement trades real-time complexity for depth of feature engineering and model interpretability.

### Headline of Press Release:

**Credit card fraud deteced using machine-learning model which priortizes credit user**
→ [press_release.md](/press_release/press_release.md)

## Domain Exposition

### Terminology

| Term | Definition |
|------|-----------|
| **Fraud** | A transaction initiated without the cardholder's authorization, typically after card data theft or account compromise |
| **Class imbalance** | A condition where one class (fraud) is vastly underrepresented relative to the other (legitimate), often 2% of records |
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

### Project Domain

This project lives at the intersection of financial technology and machine learning. Credit card fraud detection is one of the most well studied and talked about machine learning topics. Credit is the back bone of the economy, you need it to buy a house, start a business, and a car. Every person has a credit card, therefore every major card network (Visa, Mastercard, American Express) needs to have a mechanism agains't fraud. The domain is characterized by extreme class imbalance, where fewer than 2 in every 1,000 transactions are fraudulent, meaning naive classifiers achieve high accuracy by simply predicting not fruad for everything. So in order to create a model which is actually effective then they employ resampling strategies such as SMOTE. MongoDB's flexible schema allows each transaction document to carry nested cardholder history and merchant context alongside the core transaction fields, mirroring the structure of production fraud scoring systems.

### Background Reading:

Link to reading folder: [link](https://myuva-my.sharepoint.com/:f:/g/personal/ecn2wh_virginia_edu/IgCPaocQjqrJSLVDzM-1H6KyAYCdGd1BeT6eWIzVjZjreQ0?e=7oUsJS)

| Title | Description | Link |
|-------|-------------|------|
| A supervised machine learning algorithm for detecting and predicting fraud in credit card transactions | Article on machine learning and detection: Random forest is the most suitable model for predicting fraudulent transactions. | [link](https://myuva-my.sharepoint.com/:u:/g/personal/ecn2wh_virginia_edu/IQCYkZBHiFlPS4coLc6EXedcAVGJoDOFORsELGzbYP2QLAc?e=p8i1eW) |
| Fraud Fact and Statistics | Information on who is a victim of fraud most often: People older then 60. Important features to look at. | [link](https://myuva-my.sharepoint.com/:u:/g/personal/ecn2wh_virginia_edu/IQCuXWD6tgg6Q5bnwqlUej1QAWg2f7ICEOd5rnNa8H7yB-U?e=AWlHhc) |
| Investment Fraud Dominates Losses – FBI IC3 and Europol IOCTA Insights | Explains how fraud has changed overtime and what fraud scams look like right now | [link](https://myuva-my.sharepoint.com/:u:/g/personal/ecn2wh_virginia_edu/IQApxFvyihVlQLRIxQQKMwh3ASLqI4kq4ijnHsBmu4j-BcE?e=aveIQw) |
| AI fraud detection in banking | Article from IBM on how AI can be used in fraud detection, using credit large datasets and machine learning model | [link](https://myuva-my.sharepoint.com/:u:/g/personal/ecn2wh_virginia_edu/IQAqelY07NPiRLF4fNZjMwSxAVJiQ8aDteCabi7aGYuy7G8?e=stFAKc) |
| Fraud in Noncash Payments | Government report quantifying the scale of payment card fraud in the U.S.; provides real-world motivation | [link](https://myuva-my.sharepoint.com/:u:/g/personal/ecn2wh_virginia_edu/IQDY776ghDMjSKwQUMdqFcu9AS2SzieclsvSxi-Rp0BVTDA?e=i8N4R5) |

## Data Creation 

### Provenance

The raw data source for this project is the ULB Credit Card Fraud Detection dataset published by researchers from the Université Libre de Bruxelles. Within the dataset there are 284,807 real transactions made on a credit card by European cardholders over two days in September 2013. Of the 200,000+ transactions, there are 492 fraud transactions (which accounts for .172%). Features V1-V28 are transformed to protect the identity of the credit cardholder whereas Time, Amount, and Class (fraud or not) are kept the same. To construct a secondary dataset from this point, each raw row was enriched and reshaped into a MongoDB document. The features were joined on cardholder attributes (account age, historical spend velocity) and merchant attributes (geographic region, merchant code). This enrichment step was performed programmatically in Python prior to ingestion. Which creates a nested structure.

### Code Table

| File | Description | Link |
|------|-------------|------|
| `ingest.py` | Downloads the raw Kaggle CSV, validates schema, and outputs a cleaned dataframe | [link](#) |
| `enrich.py` | Joins synthetic cardholder and merchant attributes to each transaction row | [link](#) |
| `load_mongo.py` | Transforms enriched rows into nested MongoDB documents and bulk-inserts into Atlas | [link](#) |

### Bias identification

The dataset contains potential bias from two sources the geographic location and time window of data collection. As the data is collected over 48 hours and in a european region, it would not be good at generalizing to other regions or time periods. For example, holiday time periods or spending habits based on card network are not represented. Additionally, the PCA tranformation is necessary but obscures the original features. You are unable to tell if attributes like age or geography were encoded in the components.

### Bias mitigation

The class imbalance bias is addressed during model training by synthetically oversampling the minority (fraud) class rather than simply upsampling it. Geographic and temporal bias are harder to mitigate because they are found during data creation. The scope of the dataset can't be changed but bias can be quanified by tracking model performance degradation whne tested on held out time windows. Preventing overfit to narrow two day distribution.

### Rationale for key decisions

Several judgment calls introduce meaningful uncertainty. First, the choice to enrich with synthetic cardholder attributes preserves privacy but means the behavioral features (transaction velocity, avg.spend) are modeled approximations. This could inflate model performance artificially. Second, the decision to use SMOTE rather than class weight adjustment trades interpretability for balance. SMOTE generated synthetic fraud samples may not reflect real fraud patterns, potentially overfitting. Third, storing documents with nested merchant and cardholder sub-documents in MongoDB introduces schema flexibility but also inconsistency risk. Some fields like merchant.mcc may be missing for certain synthetic records, which must be handled explicitly in the pipeline with default values rather than rejected outright.

## Metadata

### Implicit Schema Guidelines

Each of the documents in the transactions collection for this project must contain the following fields:

| Field | Type | Requirement |
|-------|------|-------------|
| `transaction_id` | string | Required |
| `amount` | float | Required |
| `time` | int | Required |
| `class` | int (0 or 1) | Required |
| `pca_features` | object (keys V1–V28) | Required |

Documents are recommended to contian a cardholder sub-document which has the account_age_days, avg_monthly_spend, and velocity. Documents may also contian a merchant sub-document where the MCC, region, and merchant_id are all found. Optional fields are permitted but must not conflict with required field names. All monetary values are stored in USD as floats rounded to two decimal places.



### Data table

| Property | Value |
|----------|-------|
| Collection | `transactions` |
| Total documents | 284,807 |
| Fraudulent (Class = 1) | 492 (0.172%) |
| Legitimate (Class = 0) | 284,315 (99.828%) |
| Date range | Sep 1–2, 2013 |
| PCA features per document | 28 (V1–V28) |

### Data dictionary

| Feature | Type | Description | Example |
|---------|------|-------------|---------|
| `transaction_id` | string | Unique document identifier | `"txn_00042"` |
| `time` | int | Seconds elapsed since first transaction in dataset | `3600` |
| `amount` | float | Transaction amount in USD | `149.62` |
| `class` | int | Label: 0 = legitimate, 1 = fraud | `0` |
| `V1`–`V28` | float | PCA-transformed confidential transaction features | `-1.3598` |
| `cardholder.account_age_days` | int | Days since account was opened | `730` |
| `cardholder.avg_monthly_spend` | float | Average monthly spend in USD | `842.50` |
| `cardholder.velocity_7d` | int | Number of transactions in the past 7 days | `12` |
| `merchant.mcc` | string | 4-digit Merchant Category Code | `"5411"` |
| `merchant.region` | string | Geographic region of merchant | `"Western Europe"` |
| `merchant.merchant_id` | string | Unique identifier for the merchant | `"merch_8821"` |

### Uncertainty quantification for numerical features

| Feature | Min | Max | Mean | Std Dev | Uncertainty Note |
|---------|-----|-----|------|---------|------------------|
| `amount` | 0.00 | 25,691.16 | 88.35 | 250.12 | Right-skewed; large outliers may be legitimate high-value purchases or fraud |
| `time` | 0 | 172,792 | 94,813 | 47,488 | Temporal bias: only 48 hours of data; time-of-day patterns may not generalize |
| `V1`–`V28` | varies | varies | ~0 | ~1 | PCA-normalized; original scale and units unknown, limiting domain interpretation |
| `cardholder.velocity_7d` | 0 | ~50 | synthetic | synthetic | Synthetically generated; does not reflect real cardholder behavior distributions |
| `cardholder.account_age_days` | 0 | ~3650 | synthetic | synthetic | Synthetically generated; uniformly sampled, not drawn from real account-age distributions |
