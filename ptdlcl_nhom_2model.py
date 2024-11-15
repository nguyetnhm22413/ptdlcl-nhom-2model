import os
import pickle
import streamlit as st
import pandas as pd

# Đường dẫn tới các tệp mô hình và scaler
model_path1 = 'best_model.pkl'
model_path2 = 'best_model2.pkl'
scaler_path = 'scaler.pkl'

# Khởi tạo các đối tượng mô hình và scaler
best_model = None
best_model2 = None
scaler = None

# Hàm tải mô hình và scaler
def load_model_and_scaler():
    global best_model, best_model2, scaler
    try:
        if os.path.exists(model_path1):
            with open(model_path1, 'rb') as f:
                best_model = pickle.load(f)
        else:
            st.error(f"Tệp mô hình '{model_path1}' không tồn tại hoặc không thể truy cập!")
        
        if os.path.exists(model_path2):
            with open(model_path2, 'rb') as f:
                best_model2 = pickle.load(f)
        else:
            st.error(f"Tệp mô hình '{model_path2}' không tồn tại hoặc không thể truy cập!")
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            st.error(f"Tệp scaler '{scaler_path}' không tồn tại hoặc không thể truy cập!")
    
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi tải mô hình hoặc scaler: {str(e)}")

# Gọi hàm tải mô hình và scaler
load_model_and_scaler()

# Kiểm tra xem mô hình và scaler đã được tải thành công chưa
if best_model is None or best_model2 is None or scaler is None:
    st.error("Không thể tải mô hình hoặc scaler. Vui lòng kiểm tra lại các tệp.")
else:
    # Phần còn lại của mã (dự đoán, xử lý dữ liệu, v.v.)
    st.title("Dự báo Rủi ro Giao hàng Trễ và Doanh số Khách hàng")

    # Nhập dữ liệu cho mô hình phân loại
    days_for_shipment_scheduled = st.number_input("Days for shipment (scheduled)", min_value=0, step=1)
    order_item_product_price = st.number_input("Order Item Product Price", min_value=0.0, step=0.01)
    order_item_quantity = st.number_input("Order Item Quantity", min_value=0, step=1)

    input_data_classification = pd.DataFrame({
        'Days for shipment (scheduled)': [days_for_shipment_scheduled],
        'Order Item Product Price': [order_item_product_price],
        'Order Item Quantity': [order_item_quantity]
    })

    # Chuẩn hóa dữ liệu phân loại
    try:
        input_data_classification_scaled = scaler.transform(input_data_classification.select_dtypes(include=['number']))
        st.write("Dữ liệu sau khi chuẩn hóa:", input_data_classification_scaled)

        # Dự đoán phân loại (rủi ro giao hàng trễ)
        if st.button("Dự đoán Rủi ro Giao hàng Trễ"):
            prediction_classification = best_model.predict(input_data_classification_scaled)[0]
            st.write("Dự đoán Rủi ro Giao hàng Trễ:", "Có" if prediction_classification == 1 else "Không")
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi chuẩn hóa dữ liệu: {str(e)}")

    # Nhập dữ liệu cho mô hình hồi quy
    category_id = st.number_input("Category Id", min_value=0, step=1)
    customer_city = st.text_input("Customer City")
    customer_country = st.text_input("Customer Country")
    customer_segment = st.text_input("Customer Segment")
    customer_state = st.text_input("Customer State")
    product_price = st.number_input("Product Price", min_value=0.0, step=0.01)
    order_region = st.text_input("Order Region")
    market = st.text_input("Market")

    input_data_regression = pd.DataFrame({
        'Category Id': [category_id],
        'Customer City': [customer_city],
        'Customer Country': [customer_country],
        'Customer Segment': [customer_segment],
        'Customer State': [customer_state],
        'Product Price': [product_price],
        'Order Region': [order_region],
        'Market': [market]
    })

    # Chuẩn hóa dữ liệu hồi quy
    try:
        input_data_regression_scaled = scaler.transform(input_data_regression.select_dtypes(include=['number']))
        st.write("Dữ liệu sau khi chuẩn hóa:", input_data_regression_scaled)

        # Dự đoán hồi quy (doanh số khách hàng)
        if st.button("Dự đoán Doanh số Khách hàng"):
            prediction_regression = best_model2.predict(input_data_regression_scaled)[0]
            st.write("Dự đoán Doanh số Khách hàng:", prediction_regression)
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi chuẩn hóa dữ liệu hồi quy: {str(e)}")
