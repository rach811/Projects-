#Amazon sales report PYTHON 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

#load the dataset
file_path = "Downloads/archive/Amazon Sale Report.xlsx"
df = pd.read_excel(file_path)

#first headering the data 
df.head(), df.info()

#view column names
print(df.columns)

# 4. Removing missing values & outliers
Q1 = df["Amount"].quantile(0.25)
Q3 = df["Amount"].quantile(0.75)
IQR = Q3 - Q1

clean_data = df[~((df["Amount"] < (Q1 - 1.5 * IQR)) | (df["Amount"] > (Q3 + 1.5 * IQR)))]

# Drop irrelevant columns
clean_data = df.drop(columns=["index", "Unnamed: 22", "promotion-ids", "fulfilled-by"])


# Drop rows where 'Amount' or 'Qty' is missing or zero (essential for profit analysis)
clean_data = clean_data.dropna(subset=['Amount', 'Qty'])
clean_data = clean_data[clean_data['Qty'] > 0]
clean_data = clean_data[clean_data['Amount'] > 0]

# Preview cleaned data
clean_data.info(), clean_data.head()

#STEP 2 
# Calculate Revenue per Unit
clean_data["Revenue_per_Unit"] = clean_data["Amount"] / clean_data["Qty"]

# Group by 'ship-state' and calculate average Revenue per Unit
state_revenue_per_unit = clean_data.groupby("ship-state")["Revenue_per_Unit"].mean().sort_values(ascending=False)

# Pick top 5 and bottom 5
top_5_states = state_revenue_per_unit.head(5)
bottom_5_states = state_revenue_per_unit.tail(5)

# Combine results
combined = pd.concat([top_5_states, bottom_5_states])

print("Top and Bottom 5 States by Revenue per Unit:")
print(combined)

# Plotting
plt.figure(figsize=(10, 6))
combined.plot(kind='bar', color=['green']*5 + ['red']*5)
plt.title('Top and Bottom 5 States by Revenue per Unit')
plt.ylabel('Average Revenue per Unit')
plt.xlabel('Ship State')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#STEP 3
# Group by country and category, calculate average revenue per unit
category_margin = (
    clean_data.groupby(["ship-state", "Category"])["Revenue_per_Unit"]
    .mean()
    .reset_index()
)

# For each country, get the top 3 categories by average revenue per unit
top_3_categories_per_state = (
    category_margin.sort_values(["ship-state", "Revenue_per_Unit"], ascending=[True, False])
    .groupby("ship-state")
    .head(3)
)

top_3_categories_per_state

#STEP 4 
# Calculate revenue per unit as a proxy for margin
clean_data["Revenue_per_Unit"] = clean_data["Amount"] / clean_data["Qty"]

# Aggregate by SKU
sku_summary = (
    clean_data.groupby("SKU")
    .agg({"Revenue_per_Unit": "mean", "Qty": "sum"})
    .reset_index()
)

# Define thresholds
high_margin_threshold = sku_summary["Revenue_per_Unit"].quantile(0.75)
low_stock_threshold = sku_summary["Qty"].quantile(0.25)

# Filter high-margin & understocked
understocked_high_margin = sku_summary[
    (sku_summary["Revenue_per_Unit"] >= high_margin_threshold) &
    (sku_summary["Qty"] <= low_stock_threshold)
]

print("High-margin but understocked SKUs:")
print(understocked_high_margin.sort_values("Revenue_per_Unit", ascending=False))

#STEP 5 - Sales Trend & Customer Behavior Analysis

# Convert 'order-date' to datetime
clean_data['Date'] = pd.to_datetime(clean_data['Date'], format="%m-%d-%y", errors='coerce')

# Drop rows with invalid dates
clean_data = clean_data.dropna(subset=['Date'])

# Extract month
clean_data['Month'] = clean_data['Date'].dt.to_period('M')

# 1. Monthly sales trend
monthly_sales = clean_data.groupby('Month')['Amount'].sum().reset_index()
monthly_sales['Month'] = monthly_sales['Month'].astype(str)

# Plot monthly sales trend
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_sales, x='Month', y='Amount', marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales Amount')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Create proxy for customer using postal code + city (assumes clean_data is my DataFrame)
clean_data['Customer_Proxy'] = (
    clean_data['ship-postal-code'].fillna('').astype(str) + '_' +
    clean_data['ship-city'].fillna('').astype(str)
)

# Ensure order date is datetime
clean_data['Date'] = pd.to_datetime(clean_data['Date'])


# Repeat Purchase Frequency of ASINs by Customer Proxy

# Count unique orders per (Customer Proxy, ASIN)
asin_customer_orders = clean_data.groupby(['Customer_Proxy', 'ASIN'])['Order ID'].nunique().reset_index(name='Order_Count')

# Determine if ASIN has been purchased more than once by same customer
repeat_asin_customers = (asin_customer_orders['Order_Count'] > 1).sum()
single_asin_customers = (asin_customer_orders['Order_Count'] == 1).sum()

# Print stats
print(f"Repeat ASIN Purchases by Customers: {repeat_asin_customers}")
print(f"One-time ASIN Purchases by Customers: {single_asin_customers}")

# Pie chart
labels = ['Repeat ASIN Purchases', 'One-time ASIN Purchases']
sizes = [repeat_asin_customers, single_asin_customers]
colors = ['#66b3ff', '#ff9999']

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('ASIN Purchase Frequency by Customer Proxy')
plt.axis('equal')
plt.tight_layout()
plt.show()

# Seasonality Analysis — Peak Sales Months

clean_data['Month_Num'] = clean_data['Date'].dt.month
monthly_avg = clean_data.groupby('Month_Num')['Amount'].mean().reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(data=monthly_avg, x='Month_Num', y='Amount', palette='viridis')
plt.title('Average Sales by Month')
plt.xlabel('Month')
plt.ylabel('Average Sales Amount')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Weekday Sales Analysis

clean_data['Weekday'] = clean_data['Date'].dt.day_name()
weekday_sales = clean_data.groupby('Weekday')['Amount'].sum().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

plt.figure(figsize=(10, 5))
sns.barplot(x=weekday_sales.index, y=weekday_sales.values, palette='coolwarm')
plt.title('Total Sales by Day of the Week')
plt.xlabel('Day of Week')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


