import pandas as pd
import matplotlib.pyplot as plt

# Load and prepare
df = pd.read_csv("energy.csv")
df["date"] = pd.to_datetime(df["date"])
df["AWND"] = df["AWND"].fillna(df["AWND"].mean())
df = df[df["daily_consumption"] > 200]

# Create a figure with 4 subplots (2 rows, 2 columns)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- Plot 1: Consumption distribution ---
axes[0, 0].hist(df["daily_consumption"], bins=40, color="steelblue", edgecolor="black")
axes[0, 0].set_title("Distribution of Daily Consumption")
axes[0, 0].set_xlabel("Consumption")
axes[0, 0].set_ylabel("Number of days")

# --- Plot 2: Consumption vs Max Temperature ---
axes[0, 1].scatter(df["TMAX"], df["daily_consumption"], alpha=0.3, color="tomato")
axes[0, 1].set_title("Consumption vs Max Temperature")
axes[0, 1].set_xlabel("TMAX (°C)")
axes[0, 1].set_ylabel("Consumption")

# --- Plot 3: Monthly average consumption ---
df["month"] = df["date"].dt.month
monthly = df.groupby("month")["daily_consumption"].mean()
axes[1, 0].bar(monthly.index, monthly.values, color="seagreen")
axes[1, 0].set_title("Average Consumption by Month")
axes[1, 0].set_xlabel("Month")
axes[1, 0].set_ylabel("Average consumption")

# --- Plot 4: Correlation heatmap (simple version) ---
corr = df[["AWND", "PRCP", "TMAX", "TMIN", "daily_consumption"]].corr()
im = axes[1, 1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
axes[1, 1].set_xticks(range(len(corr.columns)))
axes[1, 1].set_yticks(range(len(corr.columns)))
axes[1, 1].set_xticklabels(corr.columns, rotation=45, ha="right")
axes[1, 1].set_yticklabels(corr.columns)
axes[1, 1].set_title("Correlation Between Variables")
# Add the numbers on each cell
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        axes[1, 1].text(j, i, round(corr.iloc[i, j], 2), ha="center", va="center")
fig.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=100)
plt.show()
print("EDA charts saved as eda_overview.png")