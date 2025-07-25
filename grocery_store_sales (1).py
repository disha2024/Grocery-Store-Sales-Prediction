# -*- coding: utf-8 -*-
"""Grocery_Store_Sales.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16p6VhD0LPiNp8H5TKyj3DESC-cB0X8Y1
"""

import pandas as pd                  # For loading the CSV
import numpy as np                   # For numerical operations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle     # For saving the model       
import pandas as pd
import os            

df = pd.read_csv("Stores.csv")

df.info()

df.head()

# Select features and target
X = df[['Store_Area', 'Items_Available', 'Daily_Customer_Count']]
y = df['Store_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(model.predict(X_train[:5]))  # Should show varying predictions

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))


store_area = int(input("Enter The Store Area : "))
items_available = int(input("Enter The items_available : "))
daily_customer_count = int(input("Enter The daily_customer_count : "))

input_data = np.array([[store_area, items_available, daily_customer_count]])
prediction = model.predict(input_data)[0]
print(f"Predected Sales Number : {int(prediction)}")

# Store input and prediction in a DataFrame
result_row = pd.DataFrame({
    'Store_Area': [store_area],
    'Items_Available': [items_available],
    'Daily_Customer_Count': [daily_customer_count],
    'Predicted_Sales': [prediction]
})



filename = input("Enter the User FileName with 📌 .csv extension : ")

if not  filename.lower().endswith('.csv'):
    print("✅ It is not a CSV file.")
    
else: 
    # If file exists, append; else create new with header
    if os.path.exists(filename):
        result_row.to_csv(filename, mode='a', header=False, index=False)
    else:
        result_row.to_csv(filename, mode='w', header=True, index=False)






# print("Root Mean Squared Error:", rmse)

# feature = X.columns
# importance = model.feature_importances_
# for f, i in zip(feature, importance):
#     print(f"{f}: {i:.4f}")

# #Plot Feature Importance
# import matplotlib.pyplot as plt
# plt.figure(figsize=(6,4))
# plt.bar(feature, importance, color='skyblue')
# plt.title("Feature Importance")
# plt.ylabel("Importance Score")
# plt.xlabel("Features")
# plt.tight_layout()
# plt.show()

# # Plot Actual vs Predicted
# plt.figure(figsize=(6,4))
# plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
# plt.xlabel("Actual Store Sales")
# plt.ylabel("Predicted Store Sales")
# plt.title("Actual vs Predicted Sales")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# import pickle

# with open("model.pkl", "wb") as f:
#     pickle.dump(model, f)

# print(model.predict([[1000, 1500, 300]]))

# print(model.predict([[100,38,196]]))

# print(model.predict([[756,48,97]]))

# print(model.predict([[489,68,97]]))