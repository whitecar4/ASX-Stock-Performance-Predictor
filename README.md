# ASX-Stock-Performance-Predictor
A machine learning project to predict the next-day price direction (up/down) of ASX-listed stocks using historical market data and technical indicators. Built as an independent project to develop practical skills in financial data analysis, feature engineering and model validation.

> **Disclaimer:** This project was built purely for learning purposes to develop skills in financial data analysis and machine learning. It is not intended as financial advice and has not been tested in any live trading environment. Results should not be used to inform real investment decisions.

## Project Overview

- **Target stock:** MQG.AX (Macquarie Group), chosen to explore the factors that drive a major Australian financial stock. As a diversified financial services firm, MQG is influenced by macro factors, which made it a challenging target for technical indicator-based prediction.
- **Data:** 5 years of daily OHLCV data via yfinance
- **Task:** Binary classification - predict whether tomorrow's closing price will be higher or lower than today's
- **Best validated accuracy:** 54% (v10)
- **Note:** The model accepts any ASX ticker as input and can be applied to other ASX-listed companies.

## Features Engineered

- Moving Averages (MA7, MA30)
- Momentum
- RSI (Relative Strength Index)
- Bollinger Bands (BB_Width)
- HL_Spread (intraday volatility)
- ASX200 index returns (lagged)

## Key Findings

- **Volume** — the number of shares traded on a given day - was consistently the strongest predictor. High volume days may reflect significant buying or selling activity from large market participants, which could signal upcoming price movements.
- **Data leakage** was identified in v8 — OC_Spread (close minus open price) had a correlation of -0.476 with the target variable, inflating accuracy to 61%. Confirmed via correlation analysis and feature importance diagnostics, removed in v9.
- **Temporal ordering** of the train/test split was initially incorrect — data was sorted in descending order, meaning the model trained on recent data and tested on older data. Correcting this to ascending order in v10 better simulates predicting future prices from past data, and produced the best validated result of 54%.
- **Limitation:** Technical indicators alone appear to have a ceiling for predicting MQG price direction. Macro factors such as interest rates and broader market sentiment likely influence price movement in this kind of stock, beyond what price-based features can capture.

## Version History

| Version | Key Change | Accuracy |
|---------|-----------|----------|
| v1 | Baseline model | 49% |
| v2 | 10y data — older data adds noise | 47% |
| v3 | GridSearchCV introduced | 49% |
| v4 | Added RSI | 50% |
| v5 | Removed MA30 — hurt accuracy | 49% |
| v6 | Added Bollinger Bands | 51% |
| v7 | Added MACD — adds noise | 50% |
| v8 | Added HL_Spread + OC_Spread — data leakage detected | 61%* |
| v9 | Removed OC_Spread leakage, retained HL_Spread | 51% |
| v10 | Applied ascending sort — best clean result | 54% |
| v11 | XGBoost — underperformed Random Forest | 51% |
| v12 | Added ASX200 return (lagged) — no improvement | 53% |

*v8 accuracy inflated due to data leakage — not a valid result

## Tech Stack

- Python (pandas, numpy, scikit-learn, matplotlib, seaborn, yfinance)
- SQLite
- Jupyter Notebook

## Potential Improvements

- Incorporate RBA interest rate decisions as a binary feature, to identify any possible correlation with MQG's price movements
- Explore other sequence models for time series data, assessing patterns of sequences of past observations could improve accuracy. 
- Revisit OC_Spread and HL_Spread as lagged features. Same-day intraday data was removed due to leakage but may improve predictive accuracy if shifted by one day
- Apply hyperparameter tuning to XGBoost. The initial comparison used default parameters, so results may not be directly equivalent to the tuned Random Forest model
