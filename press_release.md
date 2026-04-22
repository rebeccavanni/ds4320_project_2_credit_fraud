## Press Release

### Credit card fraud deteced using machine-learning model which priortizes credit user

Credit card fraud plagues credit holders and issuers. People are constantly afraid of their money beings stolen and credit ruined. Issuers are afraid of losing money and clients. The financial industry loses an estimated $33 billion annually to card fraud globally. Using machine learning will enable more dyanmic and accurate detection of fraud. Taking in large amounts of information and encoding it into a decision.  

Traditional fraud detection works on rules: flag transactions over $500 in a new city, block purchases from certain merchant categories late at night, deny anything that doesn't match the cardholder's typical geography. These rules were written by humans, and fraudsters have learned to work around them. Modern fraud often involves dozens of small transactions designed to look completely ordinary. On a deeper level the detection must ignore 1,000 purchases to find 1 fruad. The scale makes it almost impossible to guess correctly. This model fixes this through a non-relational database and large amounts of credit data.

Rather than a set of fixed rules, this project uses a trained classification model that learns directly from patterns in historical transaction data. The model is trained on the ULB Credit Card Fraud Detection dataset, a widely-used benchmark of .17% of purchases being fraud. Critically, the model is trained to prioritize recall catching as many actual fraud cases as possible, even at the cost of some false positives. A missed fraud case costs a real person real money. A false positive costs them a mildly annoying phone call.


The chart below illustrates one of the key signals the model relies on: fraudulent transactions tend to cluster at different amount ranges than legitimate ones. While most fraudulent charges are small, a meaningful tail of high-value fraud is also present. Small purchases because they are testing the card to see if it will decline.

![project2_pressrelease.png](/project2_pressrelease.png)
