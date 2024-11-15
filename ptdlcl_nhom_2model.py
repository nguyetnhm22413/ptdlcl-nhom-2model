import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Tải mô hình từ tệp .pkl
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

with open('best_model2.pkl', 'rb') as f:
    best_model2 = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Tiêu đề và mô tả
st.title("Dự báo Rủi ro Giao hàng Trễ và Doanh số Khách hàng")

# Nhập dữ liệu đầu vào cho cả hai dự đoán
st.header("Dự đoán Rủi ro Giao hàng Trễ")
days_for_shipment_scheduled = st.number_input("Days for shipment (scheduled)")
order_item_product_price = st.number_input("Order Item Product Price")
order_item_quantity = st.number_input("Order Item Quantity")
latitude = st.number_input("Latitude")
longitude = st.number_input("Longitude")

# Input data frame cho mô hình phân loại
input_data_classification = pd.DataFrame({
    'Days for shipment (scheduled)': [days_for_shipment_scheduled],
    'Order Item Product Price': [order_item_product_price],
    'Order Item Quantity': [order_item_quantity],
    'Latitude': [latitude],
    'Longitude': [longitude]
})

# Tiến hành chuẩn hóa dữ liệu cho phân loại
input_data_classification_scaled = scaler.transform(input_data_classification)

# Dự đoán với mô hình phân loại
if st.button("Dự đoán Rủi ro Giao hàng Trễ"):
    prediction_classification = best_model.predict(input_data_classification_scaled)[0]
    st.write("Dự đoán Rủi ro Giao hàng Trễ:", "Có" if prediction_classification == 1 else "Không")

# Nhập các thông tin khác để dự đoán doanh số khách hàng
st.header("Dự đoán Doanh số Khách hàng")
category_id = st.number_input("Category Id")
customer_city = st.text_input("Customer City")
customer_country = st.text_input("Customer Country")
customer_segment = st.text_input("Customer Segment")
customer_state = st.text_input("Customer State")
product_price = st.number_input("Product Price")
order_region = st.text_input("Order Region")
market = st.text_input("Market")

# Input data frame cho mô hình hồi quy
input_data_regression = pd.DataFrame({
    'Type': ['DEBIT'],
    'Days for shipment (scheduled)': [days_for_shipment_scheduled],
    'Delivery Status': ['Advance shipping'],
    'Category Id': [category_id],
    'Category Name': ['Cleats'],
    'Customer City': [customer_city],
    'Customer Country': [customer_country],
    'Customer Segment': [customer_segment],
    'Customer State': [customer_state],
    'Latitude': [latitude],
    'Longitude': [longitude],
    'Order City': ['Bikaner'],
    'Order Country': ['India'],
    'Order Item Product Price': [order_item_product_price],
    'Order Item Quantity': [order_item_quantity],
    'Order Status': ['COMPLETE'],
    'Product Card Id': [1360],
    'Product Price': [product_price],
    'Order Region': [order_region],
    'Market': [market]
})

# Chuẩn hóa dữ liệu cho hồi quy
input_data_regression_scaled = scaler.transform(input_data_regression.select_dtypes(include=['number']))

# Dự đoán với mô hình hồi quy
if st.button("Dự đoán Doanh số Khách hàng"):
    prediction_regression = best_model2.predict(input_data_regression_scaled)[0]
    st.write("Dự đoán Doanh số Khách hàng:", prediction_regression)
