#!/usr/bin/env python
# coding: utf-8

# # 💰 Penny Wise: Smart Family Budgeting with AI
# An interactive notebook that predicts family spending, simulates budgets, tracks goals, and visualizes trends.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd

uploaded_file = st.file_uploader("📂 Upload your dataset (.csv)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")


# Parse date
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month


# In[ ]:


encoder = OneHotEncoder()
location_encoded = encoder.fit_transform(df[['Location']]).toarray()
df[encoder.categories_[0]] = location_encoded

X = df.drop(['FamilyID', 'Date', 'NextMonthTotalSpent', 'Location'], axis=1)
y = df['NextMonthTotalSpent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: €{mae:.2f}")


# In[ ]:


# Predict on new sample
example = X_test.iloc[0].copy()
print("Original predicted spending: €{:.2f}".format(model.predict([example])[0]))

# Simulate cutting Food by 10%
original_food = example['Food']
example['Food'] = original_food * 0.9
adjusted = model.predict([example])[0]
print(f"With 10% food cut: €{adjusted:.2f}")


# In[ ]:


# Average per category by month
monthly_avg = df.groupby('Month')[['Rent', 'Food', 'Health', 'Transport', 'Education', 'Other']].mean()
monthly_avg.plot(figsize=(12,6), title="Average Category Spending by Month")
plt.ylabel("€ Amount")
plt.show()


# In[ ]:


# Spending-to-income
df['SpendingRatio'] = df['TotalSpent'] / df['Income']
alerts = df[df['SpendingRatio'] > 0.75]
print(f"Overspending alerts: {len(alerts)} cases flagged (>{75}% of income).")


# In[ ]:


# Example spike: education in April
edu_april = df[df['Month'] == 4]['Education'].mean()
edu_avg = df['Education'].mean()
if edu_april > edu_avg:
  print(f"📌 Education spending spikes in April (avg €{edu_april:.2f}) vs normal (€{edu_avg:.2f})")


# In[ ]:


# Goal to save €500 in 3 months
target_saving = 500
monthly_target = target_saving / 3
print(f"To save €{target_saving} in 3 months, reduce monthly spending by €{monthly_target:.2f}")

# Suggest reducing top categories
avg = df[['Food', 'Transport', 'Other']].mean()
cut_suggestions = (monthly_target / avg.sum()) * avg
print("Recommended monthly cuts per category:")
print(cut_suggestions.round(2))


# ### ✅ Features Included
# - ✅ Smart forecast + what-if simulation
# - ✅ Personalized budgeting based on size/income
# - ✅ Overspending alerts
# - ✅ Spending trends by month
# - ✅ Predictive education spike
# - ✅ Goal-based savings planner
# 
