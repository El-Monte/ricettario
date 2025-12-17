import pandas as pd

# Load the dataset
# sep=';' tells pandas we used semicolons, not commas
df = pd.read_csv('recipes.csv', sep=';')

# Show the first few rows
print("✅ Recipe Book Loaded Successfully!")
print(f"Total Recipes: {len(df)}")
print("\nFirst recipe example:")
print(df.iloc[0])