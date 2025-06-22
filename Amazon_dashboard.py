# amazon_dashboard.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(layout="wide")
st.title("📊 Amazon Sales Report Dashboard")

@st.cache_data
def load_amazon_data():
    # Get path directly in 'main' (no 'data' subfolder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "Amazon_Sale_Report.xlsx")  # Updated path
    
    # Debugging (check Streamlit logs)
    st.write(f"🔍 Looking for Amazon data at: {data_path}")  
    st.write(f"📂 Directory contents: {os.listdir(current_dir)}")  # Lists files in 'main'
    
    # Verify file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Amazon data file not found at: {data_path}")
    
    df = pd.read_excel(data_path)  # Still reading Excel
    
    # Data cleaning and processing
    df = df.drop(columns=["index", "Unnamed: 22", "promotion-ids", "fulfilled-by"])
    df = df.dropna(subset=['Amount', 'Qty'])
    df = df[(df['Qty'] > 0) & (df['Amount'] > 0)]
    df['Date'] = pd.to_datetime(df['Date'], format="%m-%d-%y", errors='coerce')
    df = df.dropna(subset=['Date'])
    df["Revenue_per_Unit"] = df["Amount"] / df["Qty"]
    
    df['Customer_Proxy'] = (
        df['ship-postal-code'].fillna('').astype(str) + '_' +
        df['ship-city'].fillna('').astype(str)
    )

    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    df['Month_Num'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.day_name()
    
    return df

# Load the data
df = load_amazon_data() 

# Sidebar filter
st.sidebar.header("Filters")
state = st.sidebar.selectbox("Select ship-state", ["All"] + sorted(df['ship-state'].dropna().unique()))
category = st.sidebar.selectbox("Select Category", ["All"] + sorted(df['Category'].unique()))

def filter_df(df):
    if state != "All":
        df = df[df['ship-state'] == state]
    if category != "All":
        df = df[df['Category'] == category]
    return df

filtered = filter_df(df)

# 1. Top & Bottom States by Margin
st.header("🏅 Top & Bottom States by Revenue/Unit")
top_5 = df.groupby("ship-state")["Revenue_per_Unit"].mean().nlargest(5)
bottom_5 = df.groupby("ship-state")["Revenue_per_Unit"].mean().nsmallest(5)

state_rev = pd.concat([top_5, bottom_5])
st.bar_chart(state_rev)

# 2. Top 3 Categories by State
st.header("📦 Top 3 Categories by Revenue/Unit")
cat_by_state = (
    df.groupby(["ship-state", "Category"])["Revenue_per_Unit"]
      .mean()
      .reset_index()
      .sort_values(["ship-state","Revenue_per_Unit"], ascending=[True, False])
      .groupby("ship-state")
      .head(3)
)
st.dataframe(cat_by_state[cat_by_state["ship-state"] == state] if state != "All" else cat_by_state)

# 3. High-Margin & Understocked SKUs
st.header("⚠️ High-Margin, Understocked SKUs")
sku_sum = df.groupby("SKU").agg({"Revenue_per_Unit":"mean","Qty":"sum"}).reset_index()
high_m = sku_sum["Revenue_per_Unit"].quantile(0.75)
low_q = sku_sum["Qty"].quantile(0.25)
sku_alerts = sku_sum[(sku_sum["Revenue_per_Unit"] >= high_m) & (sku_sum["Qty"] <= low_q)]
st.dataframe(sku_alerts.sort_values("Revenue_per_Unit", ascending=False))

# 4. Monthly Sales Trend
st.header("📈 Monthly Sales Trend")
monthly = df.groupby('Month')['Amount'].sum().reset_index()
fig, ax = plt.subplots(figsize=(10,4))
sns.lineplot(data=monthly, x='Month', y='Amount', marker="o", ax=ax)
ax.set_xticklabels(monthly['Month'], rotation=45)
st.pyplot(fig)

# 5. Customer Repeat Purchase (ASIN)
st.header("🔁 ASIN Repeat Purchases by Customer Proxy")
asin_orders = df.groupby(['Customer_Proxy','ASIN'])['Order ID'].nunique().reset_index(name='count')
repeat = (asin_orders['count']>1).sum()
one_time = (asin_orders['count']==1).sum()
st.metric("Repeat ASIN Orders", repeat)
st.metric("One-time ASIN Orders", one_time)

# 6. Weekday Sales Pattern
st.header("📅 Sales by Day of Week")
weekday = df.groupby('Weekday')['Amount'].sum().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
fig2, ax2 = plt.subplots(figsize=(8,4))
sns.barplot(x=weekday.index, y=weekday.values, palette="coolwarm", ax=ax2)
ax2.set_xticklabels(weekday.index, rotation=45)
st.pyplot(fig2)
