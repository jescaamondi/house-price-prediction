# House Price Prediction

Predicting house sale prices from 79 features of each home (size, quality, location, garage, basement and more) using XGBoost. The notebook walks through the full process: exploring the data, filling in missing values, encoding categories, comparing two encoding methods, and producing a final set of predictions.

**Result:** the final model explains about **91% of the variation in sale prices** (R² ≈ 0.91 on a held-out validation set).

## The data

The [Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) from Kaggle's *House Prices: Advanced Regression Techniques* competition.

| File | What it contains |
| --- | --- |
| `train.csv` | 1,460 houses with their features and sale price |
| `test.csv` | 1,459 houses with features only (the prices to predict) |
| `data_description.txt` | What every column means |
| `submission.csv` | This model's predicted prices for the test set |

## Approach

1. **Look at the target.** Sale prices are right-skewed (many affordable houses, a few very expensive ones). A log transform brings the skewness from 1.88 down to 0.12, which helps the model.

   ![Log-transformed sale price distribution](images/saleprice_log_distribution.png)

2. **Fill in missing values using what the data means.** For features like `PoolQC` or `Alley`, a blank means the house simply doesn't have one, so it becomes `"None"`. Area and count features become `0`. `LotFrontage` is filled with the median for that neighbourhood, since nearby houses tend to have similar lots.
3. **Encode ordered categories as scores.** Quality ratings like `ExterQual` and `KitchenQual` (Poor → Excellent) are mapped to 0–5 so the model can use their natural order.
4. **Compare two ways of encoding the remaining categories.** The same XGBoost model was trained on label-encoded and one-hot-encoded data, with the same train/validation split, so the only difference is the encoding.
5. **Train the final model** on all training data using the winning encoding, and predict prices for the test set.

## Results

Scores on the 20% validation set (lower RMSE and MAE are better; both are on the log-price scale):

| Encoding | RMSE | MAE | R² |
| --- | --- | --- | --- |
| **Label encoding** | **0.1292** | **0.0846** | **0.9105** |
| One-hot encoding | 0.1314 | 0.0851 | 0.9075 |

Label encoding won narrowly. Tree-based models like XGBoost split on thresholds, so they handle label-encoded categories well without needing one column per category.

### What drives the price

![Top 15 feature importances](images/feature_importance.png)

Quality matters most: exterior quality (`ExterQual`) and overall quality (`OverallQual`) are the strongest predictors, followed by garage size, kitchen quality and basement quality. Living area (`GrLivArea`) also ranks highly.

## Run it yourself

```bash
git clone https://github.com/jescaamondi/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
jupyter notebook housePrediction.ipynb
```

Run all cells from top to bottom. The notebook writes a fresh `submission.csv` and saves the charts in `images/`.

## Tools

Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Jupyter
