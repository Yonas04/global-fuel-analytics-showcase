from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Global Fuel Analytics Dashboard", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "global_fuel_prices_2020_2026.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date")
    return df

df = load_data()

st.title("Global Fuel Analytics Dashboard")
st.caption("Interactive analysis of weekly petrol, diesel, and LPG prices across countries from 2020 to 2026.")

with st.sidebar:
    st.header("Filters")
    regions = sorted(df["region"].dropna().unique())
    countries = sorted(df["country"].dropna().unique())
    income_levels = sorted(df["income_level"].dropna().unique())
    fuel_type = st.selectbox("Fuel type", ["petrol_usd_liter", "diesel_usd_liter", "lpg_usd_liter"], index=0)
    selected_regions = st.multiselect("Region", regions, default=regions)
    filtered_countries = sorted(df[df["region"].isin(selected_regions)]["country"].unique()) if selected_regions else countries
    selected_countries = st.multiselect("Country", filtered_countries, default=filtered_countries[:8] if len(filtered_countries) > 8 else filtered_countries)
    selected_income = st.multiselect("Income level", income_levels, default=income_levels)
    date_range = st.date_input(
        "Date range",
        value=(df["date"].min().date(), df["date"].max().date())
    )

start_date, end_date = date_range

filtered = df[
    (df["region"].isin(selected_regions)) &
    (df["income_level"].isin(selected_income)) &
    (df["date"].dt.date >= start_date) &
    (df["date"].dt.date <= end_date)
].copy()

if selected_countries:
    filtered = filtered[filtered["country"].isin(selected_countries)]

friendly_name = {
    "petrol_usd_liter": "Petrol",
    "diesel_usd_liter": "Diesel",
    "lpg_usd_liter": "LPG"
}[fuel_type]

col1, col2, col3, col4 = st.columns(4)
col1.metric(f"Average {friendly_name} price", f"${filtered[fuel_type].mean():.2f}/L")
col2.metric("Countries", f"{filtered['country'].nunique()}")
col3.metric("Average Brent crude", f"${filtered['brent_crude_usd'].mean():.2f}")
col4.metric("Average tax rate", f"{filtered['tax_percentage'].mean():.1f}%")

trend = filtered.groupby("date", as_index=False)[fuel_type].mean()
fig_trend = px.line(
    trend,
    x="date",
    y=fuel_type,
    title=f"Average {friendly_name} Price Over Time"
)
st.plotly_chart(fig_trend, use_container_width=True)

left, right = st.columns(2)

country_avg = filtered.groupby("country", as_index=False)[fuel_type].mean().sort_values(fuel_type, ascending=False).head(15)
fig_country = px.bar(
    country_avg,
    x="country",
    y=fuel_type,
    title=f"Top 15 Countries by Average {friendly_name} Price"
)
left.plotly_chart(fig_country, use_container_width=True)

scatter_sample = filtered.sample(min(len(filtered), 2500), random_state=42)
fig_scatter = px.scatter(
    scatter_sample,
    x="brent_crude_usd",
    y=fuel_type,
    color="region",
    hover_data=["country", "income_level", "subsidy_level"],
    title=f"Brent Crude vs {friendly_name} Price"
)
right.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("Filtered data preview")
st.dataframe(
    filtered[["date", "country", "region", "income_level", "subsidy_level", fuel_type, "brent_crude_usd", "tax_percentage"]]
    .rename(columns={fuel_type: f"{friendly_name.lower()}_usd_liter"}),
    use_container_width=True,
    hide_index=True
)
