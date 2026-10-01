# 🏠 House Price Predictor

A machine learning model that predicts house sale prices based on property features. Trained on the [Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) dataset with custom feature engineering.

---

## 📋 Project Overview

This project predicts residential home prices using a regression model trained on the Kaggle House Prices competition. The model uses **log-transformed features** and **engineered features** (e.g., `TotalSF`, `Qual_TotalSF`, `HouseAge`, `RemodAge`) to improve prediction accuracy. Predictions are returned in original dollar scale via `np.expm1()`.

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `main.py` | Main script — loads the model and runs 8 test scenarios |
| `house_price_model.pkl` | Trained regression model (serialized with `joblib`) |
| `feature_names.pkl` | Ordered list of feature names expected by the model |
| `sample_house.csv` | Baseline house row used as the template for tests |
| `submission.csv` | Example prediction output (Kaggle submission format) |
| `.gitignore` | Files excluded from version control (e.g., `.venv/`) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/bilal-157/house-price-predictor.git
   cd house-price-predictor
   ```

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install joblib pandas numpy scikit-learn
   ```

   > Add `xgboost` or `lightgbm` here if your `.pkl` model was trained with those libraries.

### Run

```bash
python main.py
```

This will execute 8 built-in tests, including minimum/maximum houses, quality-vs-price sweeps, area-vs-price sweeps, and age-vs-price sweeps.

---

## 🧠 Usage

### Make a Custom Prediction

```python
import joblib
import pandas as pd
import numpy as np

# Load model + expected feature order
model    = joblib.load('house_price_model.pkl')
features = joblib.load('feature_names.pkl')

# Start from a known baseline
base_house = pd.read_csv('sample_house.csv').iloc[0].to_dict()

def predict(house):
    df = pd.DataFrame([house])[features]
    pred = model.predict(df.values)
    # Model was trained on log(SalePrice), so invert the transform
    return np.expm1(pred[0]) if hasattr(pred, '__len__') else np.expm1(pred)

# Example: 2000 sqft, quality 7, built 2010
h = dict(base_house)
h['OverallQual']  = 7
h['GrLivArea']    = np.log1p(2000)
h['TotalSF']      = np.log1p(2000)
h['Qual_TotalSF'] = 7 * 2000

print(f"Predicted price: ${predict(h):,.0f}")
```

> ⚠️ **Important:** The model expects **log-transformed** versions of several features (`GrLivArea`, `TotalBsmtSF`, `1stFlrSF`, `2ndFlrSF`, `TotalSF`). Always apply `np.log1p()` to raw square-footage values before predicting.

---

## 🔧 Engineered Features

Beyond the raw Kaggle columns, the model relies on these derived features:

| Feature | Formula | Purpose |
|---------|---------|---------|
| `GrLivArea` (transformed) | `log1p(GrLivArea)` | Compress large area values |
| `TotalBsmtSF` (transformed) | `log1p(TotalBsmtSF)` | Compress basement size |
| `1stFlrSF` (transformed) | `log1p(1stFlrSF)` | Compress first-floor size |
| `2ndFlrSF` (transformed) | `log1p(2ndFlrSF)` | Compress second-floor size |
| `TotalSF` | `log1p(TotalBsmtSF + 1stFlrSF + 2ndFlrSF)` | Overall house size |
| `Qual_TotalSF` | `OverallQual × (TotalBsmtSF + 1stFlrSF + 2ndFlrSF)` | Interaction: quality × size |
| `HouseAge` | `2026 − YearBuilt` | Age of the property |
| `RemodAge` | `2026 − YearRemodAdd` | Years since remodel |

> 💡 The `2026` reference year is hardcoded in `main.py` — update it if you retrain in a later year.

---

## 📊 Built-in Tests (`main.py`)

| # | Test | What it does |
|---|------|--------------|
| 1 | 🏚️ Minimum House | 300 sqft, quality 1, built 1900 |
| 2 | 🏰 Maximum Mansion | 6000+ sqft, quality 10, built 2024 |
| 3 | ❌ Invalid Values | Negative quality / area — checks error handling |
| 4 | 🕳️ Zero Area | All areas = 0 |
| 5 | 📊 Quality vs Price | Sweeps `OverallQual` from 1 → 10 |
| 6 | 📊 Area vs Price | Sweeps living area from 500 → 5000 sqft |
| 7 | 📊 Age vs Price | Sweeps `YearBuilt` from 1900 → 2024 |
| 8 | 🎲 Random Tests | 5 randomized feature combinations |

Example output:

```text
============================================================
🏚️ TEST 1: MINIMUM HOUSE (300 sqft, 1900, Qual 1)
============================================================
Price: $XX,XXX
...
✅ ALL TESTS COMPLETE
```

---

## 🏆 Model Details

| Property | Value |
|----------|-------|
| **Task** | Regression |
| **Target** | `SalePrice` (log-transformed during training) |
| **Prediction Inverse** | `np.expm1(pred)` |
| **Dataset** | [Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) |
| **Training Environment** | Kaggle Notebook (GPU T4 ×2) |
| **Serialization** | `joblib` |
| **Feature Order** | Enforced via `feature_names.pkl` |

---

## ⚠️ Known Limitations

- **Input validation:** The model does not reject physically impossible inputs (e.g., negative quality, houses from the year 2050). It will still return a number. Add validation before production use.
- **Feature order matters:** Always reindex the DataFrame with `features` from `feature_names.pkl`.
- **Hardcoded reference year (2026):** `HouseAge` / `RemodAge` will drift as time passes — update and retrain periodically.
- **Assumes log-transformed inputs:** Feeding raw square-footage values will produce incorrect predictions.

---

## 🔄 Reproducing / Retraining

1. Open the [Kaggle competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).
2. Recreate the training notebook with the same feature engineering.
3. Export the model:

   ```python
   import joblib
   joblib.dump(model, 'house_price_model.pkl')
   joblib.dump(list(X.columns), 'feature_names.pkl')
   ```

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a branch: `git checkout -b feature/improvement`
3. Commit your changes: `git commit -m 'Add improvement'`
4. Push: `git push origin feature/improvement`
5. Open a Pull Request

---

## 📄 License

Licensed under the **Apache 2.0 License** — see [LICENSE](https://www.apache.org/licenses/LICENSE-2.0) for details.

---

## 👤 Author

**Bilal** — [@bilal-157](https://github.com/bilal-157)

---

## ⭐ Show Your Support

If this project was helpful, please give it a ⭐ on GitHub!

---

## 📌 Notes

- `.venv/` and other local environment files are ignored via `.gitignore`
- `submission.csv` is a sample Kaggle-format submission file
- Model file sizes are within GitHub's 100 MB limit. If you retrain a larger model, consider [Git LFS](https://git-lfs.github.com/)

---
```
