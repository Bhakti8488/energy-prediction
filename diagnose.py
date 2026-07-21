import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("energy.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Plot consumption over the whole time period
plt.figure(figsize=(12, 5))
plt.plot(df["date"], df["daily_consumption"], linewidth=0.7)
plt.title("Daily Energy Consumption Over Time")
plt.xlabel("Date")
plt.ylabel("Consumption")
plt.grid(True)
plt.savefig("consumption_over_time.png")
plt.show()

# Also check: is the test period's average very different from training?
split_point = int(len(df) * 0.8)
train_avg = df["daily_consumption"][:split_point].mean()
test_avg = df["daily_consumption"][split_point:].mean()
print("Training period average consumption:", round(train_avg, 1))
print("Test period average consumption:", round(test_avg, 1))