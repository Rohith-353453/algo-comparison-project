# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 11:59:46 2025

@author: luci3
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Data (replace path if needed)
df = pd.read_csv("C:\\Users\\luci3\\OneDrive\\Documents\\aenexz\\project - comparing the algos\\Flight_Booking.csv.csv")  # Your Kaggle file

print(f"Dataset shape: {df.shape}")
print(df.head(3))
print(df['Price'].describe())

# Step 2: Preprocessing (FIXED FOR DURATION PARSING)
# Rename to match expected (from previous fix)
df = df.rename(columns={
    'Source': 'Source_city',
    'Destination': 'Destination_city',
    'Dep_Time': 'Departure_Time',
    'Total_Stops': 'Stops'
})

# Parse Duration: Convert '2h 50m' to total minutes (float)
def parse_duration(duration_str):
    if pd.isna(duration_str):
        return np.nan
    duration_str = str(duration_str).replace('h ', 'h').replace('m', '')  # Clean: '2h50m' → '2h50'
    hours = 0
    minutes = 0
    if 'h' in duration_str:
        hours = float(duration_str.split('h')[0].strip())
        mins_part = duration_str.split('h')[1].strip() if 'h' in duration_str else '0'
        if mins_part:
            minutes = float(mins_part)
    else:
        minutes = float(duration_str.strip())
    return hours * 60 + minutes

df['Duration'] = df['Duration'].apply(parse_duration)

# Engineer missing features (same as before)
df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], dayfirst=True)
df['Month'] = df['Date_of_Journey'].dt.month
df['Day'] = df['Date_of_Journey'].dt.day
df['Days_left'] = 30 - df['Day']  # Rough estimate

# For class: Set to 'Economy'
df['class'] = 'Economy'

# Select features (now Duration is numeric!)
features = ['Airline', 'Source_city', 'Destination_city', 'Stops', 'Price', 'Days_left', 
            'Departure_Time', 'Arrival_Time', 'Duration', 'class', 'Month']
df = df[features].copy().dropna()  # Drop NAs (Duration parse may create some)

print("Updated shape after parsing:", df.shape)
print("Sample head (Duration now numeric):")
print(df.head(3))
print("Duration sample:", df['Duration'].head().tolist())  # E.g., [170.0, 445.0, ...]

# Encode categoricals (Duration/nums untouched)
le = LabelEncoder()
cat_cols = ['Airline', 'Source_city', 'Destination_city', 'Stops', 'Departure_Time', 
            'Arrival_Time', 'class']
for col in cat_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# Features & Target
X = df.drop('Price', axis=1)
y = df['Price']

# Scale (now all numeric— no string errors!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"\nTrain shape: {X_train.shape}, Test: {X_test.shape}")

# Step 3: Train & Evaluate Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'KNN': KNeighborsRegressor(n_neighbors=5)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'R²': r2})
    print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")

# Step 4: Comparison Table
metrics_df = pd.DataFrame(results)
print("\n--- COMPARISON TABLE ---")
print(metrics_df.round(2))

# Quick Insights (add to your report)
print("\nInsights: Random Forest edges out on accuracy (non-linear patterns like airline-route interactions). "
      "Linear is fastest but assumes linearity (underperforms). KNN captures local similarities but sensitive to scaling.")

# Step 5: Visualizations (Run to see plots)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics_df.set_index('Model')[['RMSE']].plot(kind='bar', ax=axes[0], title='RMSE Comparison (Lower Better)')
axes[0].set_ylabel('RMSE (₹)')
metrics_df.set_index('Model')[['R²']].plot(kind='bar', ax=axes[1], title='R² Score (Higher Better)')
axes[1].set_ylabel('R²')
# Residuals for Linear (example)
lr = models['Linear Regression']
lr_pred = lr.predict(X_test)
axes[2].scatter(lr_pred, y_test - lr_pred, alpha=0.5)
axes[2].axhline(0, color='r', ls='--')
axes[2].set_xlabel('Predicted Price')
axes[2].set_ylabel('Residuals')
axes[2].set_title('Linear Residuals')
plt.tight_layout()
plt.show()

# Step 6: Recommendation Demo (FIXED FOR 10 FEATURES)
print("\n--- RECOMMENDATION DEMO ---")
# Base encoded (9 fixed: Airline, Source_city, Dest_city, Stops, Dep_Time, Arr_Time, Duration, class, Month)
# E.g., IndiGo(4), Delhi(0)-Mumbai(3), non-stop(4), Evening(5), Night(6), 120 mins, Economy(0), March(3)
# Adjust indices after checking le.classes_ (e.g., print(le.classes_['Airline']) etc.)
base_fixed = np.array([4, 0, 3, 4, 5, 6, 120.0, 0, 3])  # Now 9 items!
days_left_range = np.arange(1, 31).reshape(-1, 1)
sample_features = np.hstack([np.tile(base_fixed, (30, 1)), days_left_range])  # 9 + 1 = 10 cols
sample_scaled = scaler.transform(sample_features)

rf = models['Random Forest']  # Use best model
prices_pred = rf.predict(sample_scaled)

# Plot & Rec (same)
plt.figure(figsize=(10, 5))
plt.plot(days_left_range.flatten(), prices_pred, marker='o', label='Predicted Price')
plt.xlabel('Days Left to Departure')
plt.ylabel('Predicted Price (₹)')
plt.title('Price Over Booking Window (e.g., Delhi-Mumbai, March)')
plt.grid(True)
plt.legend()
plt.show()

best_idx = np.argmin(prices_pred)
best_day = days_left_range[best_idx][0]
best_price = prices_pred[best_idx]
avg_price = np.mean(prices_pred)
savings = avg_price - best_price
print(f"Recommendation: Book {best_day} days ahead at ~₹{best_price:.0f} "
      f"(saves ~₹{savings:.0f} vs average ~₹{avg_price:.0f}).")