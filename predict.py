import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


# --- 1. Load and prepare ---
df = pd.read_csv("energy.csv")
df["AWND"] = df["AWND"].fillna(df["AWND"].mean())
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Remove near-zero bad readings
df = df[df["daily_consumption"] > 200].reset_index(drop=True)

# --- 2. Lag features (past values) ---
df["consumption_lag_1"] = df["daily_consumption"].shift(1)   # yesterday
df["consumption_lag_7"] = df["daily_consumption"].shift(7)   # same day last week

# --- 3. Rolling window features (last 7 days) ---
# .shift(1) first so we only use PAST days, never today itself
df["roll_mean_7"] = df["daily_consumption"].shift(1).rolling(7).mean()
df["roll_std_7"]  = df["daily_consumption"].shift(1).rolling(7).std()
df["temp_mean_7"] = df["TMAX"].shift(1).rolling(7).mean()

# --- 4. Drop rows with missing values (first several days lack history) ---
df = df.dropna().reset_index(drop=True)

# --- 5. Features and target ---
features = [
    "AWND", "PRCP", "TMAX", "TMIN",
    "consumption_lag_1", "consumption_lag_7",
    "roll_mean_7", "roll_std_7", "temp_mean_7"
]
X = df[features]
y = df["daily_consumption"]
# Standardize features so coefficients are comparable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=features)
# --- Remove bad/impossible readings ---
# Days with near-zero consumption are almost certainly meter errors.
before = len(df)
df = df[df["daily_consumption"] > 200]   # keep only realistic days
after = len(df)
print("Removed", before - after, "bad rows (near-zero consumption)")
print()
# --- 4. Split into training and testing sets ---
# We train on the first 80% of days, test on the last 20%.
# Because this is time-series, we split by TIME, not randomly.
split_point = int(len(df) * 0.8)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

print("Training on", len(X_train), "days")
print("Testing on", len(X_test), "days")
print()

# --- 5. Create and train the model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- 6. Make predictions on the test set ---
predictions = model.predict(X_test)
print(predictions)
# --- 7. Measure how good the predictions are ---
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", round(mae, 2))
print("R-squared score:", round(r2, 3))

# --- Which features matter most? ---
import numpy as np

print()
print("Feature influence (coefficients):")
for name, coef in zip(features, model.coef_):
    print(f"  {name}: {round(coef, 2)}")