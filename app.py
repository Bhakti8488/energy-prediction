import pandas as pd
from sklearn.linear_model import LinearRegression

# --- Train a simple weather-only model for the interface ---
df = pd.read_csv("energy.csv")
df["AWND"] = df["AWND"].fillna(df["AWND"].mean())
df = df[df["daily_consumption"] > 200]

features = ["AWND", "PRCP", "TMAX", "TMIN"]
X = df[features]
y = df["daily_consumption"]

model = LinearRegression()
model.fit(X, y)

# --- Ask the user for weather values ---
print("=" * 45)
print("  ENERGY CONSUMPTION PREDICTOR")
print("=" * 45)
print("Enter realistic weather values for a day in northern France:\n")

try:
    awnd = float(input("Wind speed (m/s), range 0-10, typical ~2.5: "))
    prcp = float(input("Precipitation (mm), range 0-50, typical 0: "))
    tmax = float(input("Max temperature (°C), range -9 to 39: "))
    tmin = float(input("Min temperature (°C), range -14 to 27: "))

    # --- Make the prediction ---
    input_data = pd.DataFrame([[awnd, prcp, tmax, tmin]], columns=features)
    prediction = model.predict(input_data)[0]

    print("\n" + "=" * 45)
    print(f"  Predicted consumption: {round(prediction, 1)} units")
    print("=" * 45)

except ValueError:
    print("\nPlease enter numbers only. Try again.")