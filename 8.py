#8
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset (from Seaborn's built-in dataset)
titanic_data = sns.load_dataset("titanic")

# Step 2: Data Exploration
print("First few rows of the dataset:")
print(titanic_data.head())

print("\nSummary statistics:")
print(titanic_data.describe())

print("\nData types and missing values:")
print(titanic_data.info())

# Step 3: Data Cleaning
# Handling missing values
titanic_data.dropna(subset=["age"], inplace=True)
titanic_data["embarked"].fillna(titanic_data["embarked"].mode()[0], inplace=True)

# Step 4: Data Visualization
# Example visualizations:

# - Countplot to visualize the count of passengers by class and gender
plt.figure(figsize=(10, 5))
sns.countplot(x="class", hue="sex", data=titanic_data)
plt.title("Passenger Count by Class and Gender")
plt.show()

# - Histogram of passenger ages
plt.figure(figsize=(10, 5))
sns.histplot(titanic_data["age"], bins=20, kde=True)
plt.title("Distribution of Passenger Ages")
plt.show()

# - Survival rate by class
survival_rate_by_class = titanic_data.groupby("class")["survived"].mean()
plt.figure(figsize=(8, 4))
sns.barplot(x = survival_rate_by_class.index, y = survival_rate_by_class.values)
plt.title("Survival Rate by Class")
plt.show()
