from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "global_fuel_prices_2020_2026.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values(["country", "date"]).copy()

# Time-series style lag features
for fuel in ["petrol_usd_liter", "diesel_usd_liter", "lpg_usd_liter", "brent_crude_usd", "tax_percentage"]:
    df[f"{fuel}_lag1"] = df.groupby("country")[fuel].shift(1)

df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year

model_df = df.dropna().copy()

target = "petrol_usd_liter"
features = [
    "country",
    "region",
    "income_level",
    "subsidy_level",
    "diesel_usd_liter",
    "lpg_usd_liter",
    "brent_crude_usd",
    "tax_percentage",
    "petrol_usd_liter_lag1",
    "diesel_usd_liter_lag1",
    "lpg_usd_liter_lag1",
    "brent_crude_usd_lag1",
    "tax_percentage_lag1",
    "month",
    "year",
]

X = model_df[features]
y = model_df[target]

categorical_features = ["country", "region", "income_level", "subsidy_level"]
numeric_features = [col for col in features if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), numeric_features),
    ]
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_depth=12,
    min_samples_leaf=2
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("Petrol Price Prediction Model")
print("-----------------------------")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

results = pd.DataFrame({
    "actual": y_test.values,
    "predicted": preds
}).head(10)

print("\nSample predictions:")
print(results.round(4).to_string(index=False))
