# One in Every Thousand Transactions Is Fraud — Here's How Machine Learning Catches It

## The Stakes Are Too High to Guess

Every time you tap your card at a coffee shop or enter your number online, your bank has a fraction of a second to decide whether to approve the transaction. Get it wrong in one direction and a fraudster walks away with your money. Get it wrong in the other and you're embarrassed at the register, phone in hand, trying to explain to a cashier why your card just declined on a $6 latte. The financial industry loses an estimated $33 billion annually to card fraud globally — and the people who feel it most aren't the banks. They're cardholders waiting days for a dispute to resolve while their account is frozen.

## The Problem With "It Looks Suspicious"

### Why Current Systems Struggle

Traditional fraud detection works on rules: flag transactions over $500 in a new city, block purchases from certain merchant categories late at night, deny anything that doesn't match the cardholder's typical geography. These rules were written by humans, and fraudsters have learned to work around them. Modern card-not-present fraud — the kind that happens when your card number is stolen but your physical card is not — often involves dozens of small transactions designed to look completely ordinary. A rule-based system will approve every single one.

The deeper problem is scale. A major card network processes tens of thousands of transactions per second. Of those, fewer than two in every thousand are fraudulent — but those two have to be found, in real time, without slowing down the other 998. This is one of the hardest classification problems in applied data science: the signal is real, but it's buried in a dataset that is 99.8% noise.

This project addresses a specific version of this challenge: given a historical dataset of anonymized credit card transactions, can we build a machine learning model — backed by a flexible, document-oriented database — that reliably identifies fraud while minimizing the harm caused by false alarms?

## Machine Learning Finds What Rules Miss

### A Smarter Approach to an Old Problem

Instead of handwritten rules, this project uses a trained classification model that learns directly from patterns in historical transaction data. The model is trained on the ULB Credit Card Fraud Detection dataset, a widely-used benchmark of 284,807 transactions made by European cardholders over two days in September 2013, of which 492 (0.17%) are fraudulent.

Each transaction is stored as a document in a MongoDB database — a format that mirrors how real fraud systems work, where each record can carry not just the raw transaction fields but also derived behavioral context, merchant metadata, and uncertainty estimates. The model learns which combinations of features — transaction amount, time of day, and 28 anonymized behavioral signals — are associated with fraud, and assigns each new transaction a fraud probability score.

Critically, the model is trained to prioritize **recall**: catching as many actual fraud cases as possible, even at the cost of some false positives. A missed fraud case costs a real person real money. A false positive costs them a mildly annoying phone call.

## How the Fraud Compares

### Transaction Amount Distribution: Fraud vs. Legitimate

The chart below illustrates one of the key signals the model relies on: fraudulent transactions tend to cluster at different amount ranges than legitimate ones. While most fraudulent charges are small — consistent with "testing" stolen card numbers before larger purchases — a meaningful tail of high-value fraud is also present.

```
Transaction Amount Distribution (Fraud vs. Legitimate)

Frequency
  ▲
  │  ██
  │  ██
  │  ██  ░░
  │  ██  ░░░░
  │  ██  ░░░░░░
  │  ██  ░░░░░░░░         ░░
  └──────────────────────────────► Amount ($)
     $0  $50  $100  $200  $500  $1000+

  ██ Legitimate transactions    ░░ Fraudulent transactions
```

*(Full interactive visualization available in the pipeline notebook.)*

This pattern — small-amount fraud clustering near $0–$50, with a secondary spike at high values — is one of several features the model uses to distinguish fraud from normal spending. By learning these distributions across all 28 anonymized features simultaneously, the classifier achieves an AUC-ROC score well above 0.95 on held-out test data, dramatically outperforming any single rule.

The goal of this project is not to replace human judgment or production-grade fraud systems. It is to demonstrate that a thoughtfully structured document database, combined with a principled machine learning pipeline, can surface patterns in transaction data that would be invisible to any rule a human could write — and to do so in a way that is reproducible, documented, and ready for further refinement.
