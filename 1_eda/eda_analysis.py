from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "global_fuel_prices_2020_2026.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum())

summary = df[["petrol_usd_liter", "diesel_usd_liter", "lpg_usd_liter", "brent_crude_usd", "tax_percentage"]].describe()
print("\nSummary statistics:")
print(summary.round(3))

# 1. Average petrol price by region
region_avg = df.groupby("region", as_index=False)["petrol_usd_liter"].mean().sort_values("petrol_usd_liter", ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(region_avg["region"], region_avg["petrol_usd_liter"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Average petrol price (USD/liter)")
plt.title("Average Petrol Price by Region")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "avg_petrol_by_region.png", dpi=200)
plt.close()

# 2. Average petrol price by income level
income_avg = df.groupby("income_level", as_index=False)["petrol_usd_liter"].mean().sort_values("petrol_usd_liter", ascending=False)
plt.figure(figsize=(8, 5))
plt.bar(income_avg["income_level"], income_avg["petrol_usd_liter"])
plt.ylabel("Average petrol price (USD/liter)")
plt.title("Average Petrol Price by Income Level")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "avg_petrol_by_income.png", dpi=200)
plt.close()

# 3. Brent vs petrol scatter
sample = df.sample(min(len(df), 2500), random_state=42)
plt.figure(figsize=(8, 5))
plt.scatter(sample["brent_crude_usd"], sample["petrol_usd_liter"], alpha=0.35)
plt.xlabel("Brent crude (USD)")
plt.ylabel("Petrol price (USD/liter)")
plt.title("Brent Crude vs Petrol Price")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "brent_vs_petrol.png", dpi=200)
plt.close()

# 4. Global petrol trend over time
trend = df.groupby("date", as_index=False)["petrol_usd_liter"].mean()
plt.figure(figsize=(10, 5))
plt.plot(trend["date"], trend["petrol_usd_liter"])
plt.ylabel("Average petrol price (USD/liter)")
plt.title("Global Average Petrol Price Over Time")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "global_petrol_trend.png", dpi=200)
plt.close()

print("\nSaved charts to:", OUTPUT_DIR)
print("- avg_petrol_by_region.png")
print("- avg_petrol_by_income.png")
print("- brent_vs_petrol.png")
print("- global_petrol_trend.png")

print("\nKey insights:")
print("1. Petrol prices vary meaningfully across regions and income levels.")
print("2. Brent crude prices show a positive relationship with petrol prices.")
print("3. Weekly trends make the dataset useful for forecasting and dashboarding.")
