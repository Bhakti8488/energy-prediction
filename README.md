# Energy Consumption Prediction

Predicting daily household electricity consumption from weather data using machine learning — with honest model comparison, exploratory analysis, and an interactive prediction tool.

## Problem

Can a day's electricity consumption be predicted from the weather? This project explores how well weather explains energy use for a single household — and, just as importantly, where the limits of that prediction lie.

## About the data

The dataset is the **daily electricity consumption of a single household in Sceaux, France** (about 7 km from Paris), recorded from **December 2006 to November 2010**, combined with local weather data from NOAA's GHCN-Daily archive.

- **Source:** aggregated from the UCI "Individual Household Electric Power Consumption" dataset (originally minute-by-minute power readings in kilowatts), merged with NOAA weather.
- **Rows:** ~1,400 days after cleaning.
- **Columns:** `date`, `AWND` (avg wind speed, m/s), `PRCP` (precipitation, mm), `TMAX` / `TMIN` (max/min temperature, °C), `daily_consumption` (aggregated daily electricity use).

**A note on units:** the original source measured power per minute in kilowatts; the `daily_consumption` values here are a daily aggregation of those readings. The numbers are a consistent measure of daily energy use (higher = more energy), but they are not a clean "kWh per day" figure, so predictions are reported in the dataset's own units.

## Approach

1. **Explored and cleaned** the data — removed impossible near-zero consumption days (likely meter errors).
2. **Exploratory Data Analysis** — visualized the consumption distribution, the temperature relationship, monthly seasonality, and feature correlations (see `eda_overview.png`).
3. **Feature engineering** — created lag features (yesterday's usage, same day last week) and 7-day rolling averages, using only past data to avoid leakage.
4. **Compared models** — linear regression vs. Random Forest, evaluated with a time-aware (chronological) train/test split.
5. **Built an interactive predictor** — `app.py` takes weather inputs and returns a predicted consumption.

## Key results

| Approach | MAE | R² |
|---|---|---|
| Linear regression (weather only) | 314.55 | 0.225 |
| Linear regression (+ month, day-of-week) | 329.93 | 0.168 |
| Random Forest | 340.58 | 0.079 |
| **Linear regression (+ lag & rolling features)** | **262.22** | **0.398** |

## What I learned

- **Good features beat fancy models.** The biggest improvement came from adding lag features (yesterday's consumption), not from a more complex model.
- **Simpler can win.** Random Forest overfit the training years and lost to linear regression, because the underlying relationship is roughly linear.
- **Feature encoding matters.** Adding month and day-of-week as plain numbers *hurt* the linear model, since it wrongly assumes a straight-line relationship across those values.
- **EDA confirmed the model.** Temperature is the dominant driver and the relationship is heating-based (colder → more energy); precipitation has almost no effect (correlation ≈ 0); TMAX and TMIN are highly correlated (0.96), so they carry nearly the same information.
- **Data has a ceiling.** Around 60% of day-to-day variation stays unexplained because it comes from household behavior (occupancy, holidays, appliance use) that isn't in the weather data. No model can extract information that isn't present.

## How to run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the analysis and model comparison:
   ```bash
   python predict.py
   ```
3. Generate the EDA charts:
   ```bash
   python eda.py
   ```
4. Try the interactive predictor:
   ```bash
   python app.py
   ```

## Files

- `predict.py` — data cleaning, feature engineering, model training and evaluation
- `eda.py` — exploratory data analysis with four visualizations
- `diagnose.py` — plots consumption over the full time period
- `app.py` — interactive command-line prediction tool
- `energy.csv` — the dataset
- `eda_overview.png` — saved EDA visualizations
- `requirements.txt` — project dependencies

## Tech used

Python 3.9 · pandas · scikit-learn · matplotlib