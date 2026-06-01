# Traffic Demand Prediction — Solution Explanation

## Problem
Predict traffic `demand` (a float) for each road location + time combination in the test set.
Metric: R² score (max 100 points).

---

## Approach

### 1. Feature Engineering
| Raw Feature | Transformation |
|---|---|
| `geohash` | Decoded to `lat` + `lng` coordinates |
| `timestamp` (e.g. `"2:15"`) | Split into `hour` + `minute` |
| `hour`, `minute`, `day` | Added `sin/cos` cyclical encoding (so 23:00 and 0:00 are close) |
| `LargeVehicles` | Binary: Allowed=1, Not Allowed=0 |
| `Landmarks` | Binary: Yes=1, No=0 |
| `RoadType`, `Weather` | Label encoded (each category → integer) |
| Missing values | Numeric → median fill; Categorical → "Unknown" |

### 2. Model — LightGBM
- Gradient boosting on decision trees
- Handles mixed data types and missing values natively
- Fast and accurate on tabular data

Key hyperparameters:
- `num_leaves=127` — controls tree complexity
- `learning_rate=0.05` — slow learning for better generalization
- `n_estimators=1000` with early stopping at 50 rounds

### 3. Cross-Validation
- 5-Fold KFold cross-validation
- Test predictions = average of all 5 fold models (reduces variance)
- OOF (out-of-fold) R² used to estimate real score before submission

---

## Results
| Fold | R² |
|---|---|
| 1 | 0.9324 |
| 2 | 0.9320 |
| 3 | 0.9323 |
| 4 | 0.9268 |
| 5 | 0.9345 |
| **Overall OOF** | **0.9317** |

**Estimated Score: ~93.17 / 100**

---

## Output
`submission.csv` — two columns: `Index` and predicted `demand` for all 41,778 test rows.
