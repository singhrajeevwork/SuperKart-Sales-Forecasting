
import streamlit as st
import pandas as pd
import requests
import os


# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="SuperKart Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

st.title("📊 SuperKart Sales Forecasting")
st.write(
    "Predict product-store sales using the trained machine learning model."
)

# Backend API URL
#BACKEND_URL = "http://127.0.0.1:5000/predict"
BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:5000/predict"
)

# ============================================
# Input Features
# ============================================

REQUIRED_COLUMNS = [
    'Product_Weight',
    'Product_Sugar_Content',
    'Product_Allocated_Area',
    'Product_Type',
    'Product_MRP',
    'Store_Id',
    'Store_Establishment_Year',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type'
]


# ============================================
# Sidebar - Prediction Mode
# ============================================

st.sidebar.header("Prediction Mode")

prediction_mode = st.sidebar.radio(
    "Choose prediction type:",
    ["Single Prediction", "Batch Prediction"]
)


# ============================================
# Single Prediction
# ============================================

if prediction_mode == "Single Prediction":

    st.header("Single Product-Store Prediction")

    col1, col2 = st.columns(2)

    with col1:

        product_weight = st.number_input(
            "Product Weight",
            min_value=0.0,
            value=12.0
        )

        product_sugar_content = st.selectbox(
            "Product Sugar Content",
            ["Low Sugar", "Regular", "No Sugar"]
        )

        product_allocated_area = st.number_input(
            "Product Allocated Area",
            min_value=0.0,
            value=0.05
        )

        product_type = st.selectbox(
            "Product Type",
            [
                "Baking Goods",
                "Breads",
                "Breakfast",
                "Canned",
                "Dairy",
                "Frozen Foods",
                "Fruits and Vegetables",
                "Hard Drinks",
                "Health and Hygiene",
                "Household",
                "Meat",
                "Others",
                "Seafood",
                "Snack Foods",
                "Soft Drinks",
                "Starchy Foods"
            ]
        )

        product_mrp = st.number_input(
            "Product MRP",
            min_value=0.0,
            value=150.0
        )

    with col2:

        store_id = st.selectbox(
            "Store ID",
            ["OUT001", "OUT002", "OUT003", "OUT004"]
        )

        store_establishment_year = st.selectbox(
            "Store Establishment Year",
            [1987, 1998, 2004, 2009]
        )

        store_size = st.selectbox(
            "Store Size",
            ["Small", "Medium", "High"]
        )

        store_location_city_type = st.selectbox(
            "Store Location City Type",
            ["Tier 1", "Tier 2", "Tier 3"]
        )

        store_type = st.selectbox(
            "Store Type",
            [
                "Departmental Store",
                "Supermarket Type1",
                "Supermarket Type2",
                "Food Mart"
            ]
        )

    if st.button("Predict Sales", type="primary"):

        input_data = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar_content,
            "Product_Allocated_Area": product_allocated_area,
            "Product_Type": product_type,
            "Product_MRP": product_mrp,
            "Store_Id": store_id,
            "Store_Establishment_Year": store_establishment_year,
            "Store_Size": store_size,
            "Store_Location_City_Type": store_location_city_type,
            "Store_Type": store_type
        }

        try:

            response = requests.post(
                BACKEND_URL,
                json=input_data
            )

            if response.status_code == 200:

                result = response.json()

                predicted_sales = result["predicted_sales"]

                st.success(
                    f"Predicted Sales: {predicted_sales:,.2f}"
                )

            else:

                st.error(
                    f"Prediction failed: {response.json()}"
                )

        except Exception as e:

            st.error(
                f"Unable to connect to backend: {e}"
            )


# ============================================
# Batch Prediction
# ============================================

else:

    st.header("Batch Sales Prediction")

    st.write(
        "Upload a CSV file containing multiple product-store records."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        input_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(input_df)

        missing_columns = [
            col for col in REQUIRED_COLUMNS
            if col not in input_df.columns
        ]

        if missing_columns:

            st.error(
                f"Missing required columns: {missing_columns}"
            )

        else:

            if st.button("Predict Batch Sales", type="primary"):

                try:

                    response = requests.post(
                        BACKEND_URL,
                        json=input_df[REQUIRED_COLUMNS].to_dict(
                            orient="records"
                        )
                    )

                    if response.status_code == 200:

                        predictions = response.json()[
                            "predicted_sales"
                        ]

                        result_df = input_df.copy()

                        result_df["Predicted_Sales"] = predictions

                        st.subheader("Prediction Results")

                        st.dataframe(result_df)

                        csv_data = result_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download Predictions CSV",
                            data=csv_data,
                            file_name="superkart_sales_predictions.csv",
                            mime="text/csv"
                        )

                    else:

                        st.error(
                            f"Prediction failed: {response.json()}"
                        )

                except Exception as e:

                    st.error(
                        f"Unable to connect to backend: {e}"
                    )
