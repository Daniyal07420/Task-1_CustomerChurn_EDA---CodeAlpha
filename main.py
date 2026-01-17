import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("customer_churn_business_dataset.csv")
df = pd.DataFrame(data)
print(df.columns)
print(df.head())
# Data Cleaning
df = df.dropna()
df = df.dropna(axis=1, how='all')
df = df.drop_duplicates()
df = df.reset_index(drop=True)
print(df.info())
# Descriptive Analysis
customer_id = df["customer_id"].nunique()
churned_customers = df[df["churn"] == 1].shape[0]
non_churned_customers = df[df["churn"] == 0].shape[0]
data_types = df.dtypes
missing_values = df.isnull().sum()
print(f"Total Customers: {customer_id}")
print(f'Churned Customers: {churned_customers}')
print(f'Non-Churned Customers: {non_churned_customers}')
print("Data Types:\n", data_types)
print("Missing Values:\n", missing_values)
# KPI Calculations
total_customers = df.shape[0]
churned_customers = df[df["churn"] == 1].shape[0]
churn_rate = (churned_customers / total_customers * 100) if total_customers > 0 else 0
average_monthly_charges = df["monthly_fee"].mean()
# Univariate Analysis (Single Variable)
plt.figure(figsize=(8, 5))
sns.histplot(df["monthly_fee"], bins=30, kde=True)
plt.title("Monthly Fee Distribution")
plt.xlabel("Monthly Fee")
plt.ylabel("Frequency")
plt.show()
plt.figure(figsize=(8, 5))
sns.countplot(x="contract_type", data=df)
plt.title("Contract Type Count")
plt.xlabel("Contract Type")
plt.ylabel("Count")
plt.show()
plt.figure(figsize=(8, 5))
sns.countplot(x="payment_method", data=df)
plt.title("Payment Method Frequency")
plt.xlabel("Payment Method")
plt.ylabel("Count")
plt.show()
# Bivariate Analysis
plt.figure(figsize=(8, 5))
sns.countplot(x="contract_type", hue="churn", data=df)
plt.title("Contract Type vs Churn")
plt.xlabel("Contract Type")
plt.ylabel("Count")
plt.show()
plt.figure(figsize=(8, 5))
sns.boxplot(x="churn", y="monthly_fee", data=df)
plt.title("Monthly Fee vs Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Fee")
plt.show()
plt.figure(figsize=(8, 5))
sns.countplot(x="payment_failures", hue="churn", data=df)
plt.title("Payment Failures vs Churn")
plt.xlabel("Payment Failures")
plt.ylabel("Count")
plt.show()
plt.figure(figsize=(8, 5))
sns.countplot(x="discount_applied", hue="churn", data=df)
plt.title("Discount Applied vs Churn")
plt.xlabel("Discount Applied")
plt.ylabel("Count")
plt.show()
# Demographic Analysis
plt.figure(figsize=(8, 5))
sns.countplot(x="gender", hue="churn", data=df)
plt.title("Gender vs Churn")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()
# Financial Analysis
plt.figure(figsize=(8, 5))
sns.countplot(x="churn", hue="country", data=df)
plt.title("Country vs Churn")
plt.xlabel("Country")
plt.ylabel("Count")
plt.show()
plt.figure(figsize=(8, 5))
sns.boxplot(x="churn", y="total_revenue", data=df)
plt.title("Total Revenue vs Churn")
plt.xlabel("Churn")
plt.ylabel("Total Revenue")
plt.show()
# Correlation Analysis
df["churn"] = df["churn"].map({"Yes": 1, "No": 0})
correlation_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(8, 5))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
# Insights
high_charge_threshold = df["monthly_fee"].quantile(0.75)
high_charge_customers = df[df["monthly_fee"] > high_charge_threshold]
print(f"Number of high-charge customers: {high_charge_customers.shape[0]}")
Long_term_contracts = df[df["contract_type"] != "Month-to-month"]
print(f"Number of Long-term contract customers: {Long_term_contracts.shape[0]}")
# Business Recommendations
print("Recommendations:")
print("1. Offer discounts to high-charges customers to reduce churn.")
print("2. Promote Long-term contracts to enhance customer retention.")
print("3. Improve service quality based on common complaint types to enhance customer satisfaction.")
# Saved Cleaned Data
df.to_excel("cleaned_customer_churn_data.xlsx", index=False)
print("Cleaned data saved to 'cleaned_customer_churn_data.xlsx'")