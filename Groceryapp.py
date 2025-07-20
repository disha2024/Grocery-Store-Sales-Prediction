import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Page setup
st.set_page_config(page_title="Grocery Sales Predictor", page_icon="🛒")
st.title("🛒 Grocery Store Sales Predictor")

# Upload CSV file
#uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])
df = pd.read_csv("Stores.csv")
if df is not None:
    

    # st.subheader("Sample Data")
    # st.dataframe(df.head())

    # Select features and target
    X = df[['Store_Area', 'Items_Available', 'Daily_Customer_Count']]
    y = df['Store_Sales']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Prediction input fields
    st.subheader("🔢 Enter Store Details to Predict Sales")

    store_area = st.number_input("Store Area (sq. ft)", min_value=100, max_value=10000, value=1500)
    items_available = st.number_input("Items Available", min_value=10, max_value=10000, value=1800)
    daily_customer_count = st.number_input("Daily Customer Count", min_value=10, max_value=5000, value=1200)

    custom_name = st.text_input("Enter filename to save (without .csv):", placeholder="User's Name ")


    if st.button("Predict Sales"):
        input_data = np.array([[store_area, items_available, daily_customer_count]])
        prediction = model.predict(input_data)[0]
        formatted_result = f"₹{prediction:,.2f}"
        st.success(f"📈 Estimated Daily Sales: {formatted_result}")
        
        # Create a DataFrame to download
        result_df = pd.DataFrame({
            "Store_Area": [store_area],
            "Items_Available": [items_available],
            "Daily_Customer_Count": [daily_customer_count],
            "Predicted_Sales (₹)": [round(prediction, 2)]
        })

        # Ask user for custom filename
         # Only show download button if filename is provided
        if custom_name.strip():
            csv_data = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Your CSV",
                data=csv_data,
                file_name=f"{custom_name.strip()}.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ Please enter a filename above to enable download.")

        




    # # Show RMSE
    # y_pred = model.predict(X_test)
    # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # st.subheader("Model Evaluation")
    # st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # # Feature Importance
    # st.subheader("📊 Feature Importance")
    # importance = model.feature_importances_
    # feature_names = X.columns
    # for f, i in zip(feature_names, importance):
    #     st.write(f"{f}: {i:.4f}")

    # fig1, ax1 = plt.subplots()
    # ax1.bar(feature_names, importance, color='skyblue')
    # ax1.set_title("Feature Importance")
    # ax1.set_ylabel("Importance Score")
    # ax1.set_xlabel("Features")
    # st.pyplot(fig1)

    # # Plot Actual vs Predicted
    # st.subheader("🔍 Actual vs Predicted Sales")
    # fig2, ax2 = plt.subplots()
    # ax2.scatter(y_test, y_pred, alpha=0.6, color='blue')
    # ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    # ax2.set_xlabel("Actual Store Sales")
    # ax2.set_ylabel("Predicted Store Sales")
    # ax2.set_title("Actual vs Predicted Sales")
    # ax2.grid(True)
    # st.pyplot(fig2)
else:
    st.warning("Please upload a dataset to start.")
