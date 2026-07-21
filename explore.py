import pandas as pd

# Load the dataset
df = pd.read_csv("energy.csv")

# Look at the first few rows
print(df.head(8))

# Check the shape and columns
print("Rows and columns:", df.shape)
print()

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())
print()
# Basic statistics
print("Summary statistics:")
print(df.describe())