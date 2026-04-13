# Global Fuel Analytics Showcase

A ready data science portfolio project built from a real global fuel price dataset covering weekly observations from 2020 to 2026.

## What this project shows
- Exploratory data analysis with clear business-style insights
- Feature engineering and machine learning for petrol price prediction
- A polished Streamlit dashboard for interactive country and region analysis

## Dataset
The dataset includes:
- Country
- Region
- Income level
- Subsidy level
- Petrol, diesel, and LPG prices
- Brent crude price
- Fuel tax percentage
- Weekly dates

## Project structure
```text
global-fuel-analytics-showcase/
│
├── data/
│   └── global_fuel_prices_2020_2026.csv
├── 1_eda/
│   ├── eda_analysis.py
│   └── outputs/
├── 2_model/
│   └── price_forecast.py
├── 3_dashboard/
│   └── app.py
├── requirements.txt
└── README.md
```

## How to run
```bash
pip install -r requirements.txt
python 1_eda/eda_analysis.py
python 2_model/price_forecast.py
streamlit run 3_dashboard/app.py
```

## Resume-ready description
Built a data science project using weekly global fuel price data across multiple countries from 2020 to 2026. Performed exploratory data analysis, engineered time-series features, trained a machine learning model to predict petrol prices, and created an interactive Streamlit dashboard for stakeholder-style analysis.
