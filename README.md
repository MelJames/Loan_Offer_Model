# Loan Offer Model

A production ready, scalable, data‑pipeline load offer model utilizing customer, loan, and transaction data to identify which clients a bank should approach with a loan offer.

## Problem Statement:
Banks often rely on broad, generic targeting when offering loans, which leads to low uptake and inefficient outreach. This project aims to identify which customers are most likely to accept a loan offer by using data‑driven insights derived from customer, loan, and transaction behaviour

## Pipeline architecture
YAML‑driven catalog, modular pipelines, Delta‑table layers (raw → intermediate → primary → feature).

## EDA & cleaning:
- Automated profiling (ydata‑profiling)
- Income, age, and balance imputations
- Outlier detection & schema fixes

## Feature engineering:
- Loan behavior features (offer counts, timing, outcomes)
- Monthly transaction summaries (income, expenses, overdraft, ratios)
- Encoding, correlation filtering, skewness transformations
- Bias checks (class imbalance + gender differences)

## Next Steps:
- Train ML models to predict loan‑offer uptake
- Handle class imbalance
- Add fairness checks
- Add rolling/temporal features

## Results

## Improvements & Recomendations

## Tech Stack Used:
- 
