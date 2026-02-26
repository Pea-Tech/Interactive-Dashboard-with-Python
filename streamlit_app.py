
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Supply Chain Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("supply_chain_data_v2.csv")

df = load_data()

st.title("📦 Supply Chain Analytics Dashboard")

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Demand", int(df["Demand"].sum()))
col2.metric("Avg Supplier Rating", round(df["Supplier_Rating"].mean(),2))
col3.metric("Avg Customer Rating", round(df["Customer_Rating"].mean(),2))
col4.metric("Stockout Cases", int(df["Stockout"].sum()))

st.divider()

# Demand by product
st.subheader("Demand by Product")
prod_demand = df.groupby("Product")["Demand"].sum()
st.bar_chart(prod_demand)

# Profit distribution
st.subheader("Profit Distribution")
df["Profit"] = df["Price"] - df["Cost"]
st.histogram_chart = st.bar_chart(df["Profit"])

# Forecast demand
st.subheader("Demand Forecast")

ts = df["Demand"].reset_index(drop=True)
lag_df = pd.DataFrame({
    "lag1": ts.shift(1),
    "lag2": ts.shift(2),
    "lag3": ts.shift(3),
    "y": ts
}).dropna()

X = lag_df[["lag1","lag2","lag3"]]
y = lag_df["y"]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X,y)

future = np.array([[ts.iloc[-1], ts.iloc[-2], ts.iloc[-3]]])
prediction = model.predict(future)[0]

st.metric("Predicted Next Demand", round(prediction,2))

# Reorder alert
st.subheader("Reorder Alerts")
EOQ = np.sqrt((2 * df["Demand"].sum() * 50) / 2)
df["Reorder_Alert"] = df["Inventory_Level"] < EOQ
st.dataframe(df[df["Reorder_Alert"]==True][["Product","Inventory_Level"]].head(20))
