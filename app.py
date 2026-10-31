
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor

# =====================================================
# 1. CẤU HÌNH TRANG
# =====================================================

st.set_page_config(
    page_title="Dự đoán giá nhà",
    layout="centered"
)

st.title("DỰ ĐOÁN GIÁ NHÀ")
st.write("Nhập thông tin căn nhà để dự đoán giá bán.")

st.divider()

# =====================================================
# 2. ĐỌC DỮ LIỆU
# =====================================================

@st.cache_data
def load_data():
    train = pd.read_excel("Nhom2_GiaNha_train.xlsx")
    test = pd.read_excel("Nhom2_GiaNha_test.xlsx")
    return train, test

try:
    train_df, test_df = load_data()
except Exception as e:
    st.error(f"Không thể đọc dữ liệu Excel: {e}")
    st.stop()

# =====================================================
# 3. CHỌN ĐẶC TRƯNG
# =====================================================

features = [
    'OverallQual',
    'GrLivArea',
    'LotArea',
    'TotalBsmtSF',
    'YearBuilt',
    'FullBath',
    'TotRmsAbvGrd'
]

target = 'SalePrice'

required_columns = features + [target]

missing = [c for c in required_columns if c not in train_df.columns]

if missing:
    st.error(f"File Train thiếu các cột: {missing}")
    st.stop()

missing_test = [c for c in features if c not in test_df.columns]

if missing_test:
    st.error(f"File Test thiếu các cột: {missing_test}")
    st.stop()

# =====================================================
# 4. TIỀN XỬ LÝ DỮ LIỆU
# =====================================================

train_df = train_df.copy()

for col in features + [target]:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')

train_df = train_df.dropna(subset=[target])

X_train = train_df[features]
y_train = train_df[target]

# =====================================================
# 5. HUẤN LUYỆN ANN VÀ SVM
# =====================================================

@st.cache_resource
def train_models(X, y):

    ann_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('ann', MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=42
        ))
    ])

    ann_model = TransformedTargetRegressor(
        regressor=ann_pipeline,
        transformer=StandardScaler()
    )

    svm_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('svm', SVR(
            kernel='rbf',
            C=100,
            epsilon=0.1,
            gamma='scale'
        ))
    ])

    svm_model = TransformedTargetRegressor(
        regressor=svm_pipeline,
        transformer=StandardScaler()
    )

    ann_model.fit(X, y)
    svm_model.fit(X, y)

    return ann_model, svm_model

with st.spinner("Đang khởi tạo mô hình dự đoán..."):
    ann_model, svm_model = train_models(X_train, y_train)

# =====================================================
# 6. CHỌN THUẬT TOÁN
# =====================================================

st.subheader("Chọn thuật toán")

algorithm = st.selectbox(
    "Thuật toán dự đoán",
    ["ANN", "SVM"]
)

if algorithm == "ANN":
    model = ann_model
else:
    model = svm_model

# =====================================================
# 7. NHẬP THÔNG SỐ CĂN NHÀ
# =====================================================

st.subheader("Thông số căn nhà")

chat_luong = st.select_slider(
    "Chất lượng tổng thể",
    options=list(range(1, 11)),
    value=7
)

dien_tich_su_dung = st.number_input(
    "Diện tích sử dụng (ft²)",
    min_value=1.0,
    value=1500.0,
    step=50.0
)

dien_tich_dat = st.number_input(
    "Diện tích đất (ft²)",
    min_value=1.0,
    value=8000.0,
    step=100.0
)

dien_tich_tang_ham = st.number_input(
    "Diện tích tầng hầm (ft²)",
    min_value=0.0,
    value=800.0,
    step=50.0
)

nam_xay_dung = st.number_input(
    "Năm xây dựng",
    min_value=1800,
    max_value=2100,
    value=2000,
    step=1
)

so_phong_tam = st.number_input(
    "Số phòng tắm",
    min_value=0,
    max_value=20,
    value=2,
    step=1
)

tong_so_phong = st.number_input(
    "Tổng số phòng",
    min_value=1,
    max_value=30,
    value=6,
    step=1
)

st.divider()

# =====================================================
# 8. NÚT DỰ ĐOÁN
# =====================================================

if st.button("DỰ ĐOÁN GIÁ NHÀ", use_container_width=True):

    if dien_tich_tang_ham > dien_tich_su_dung:
        st.error("Diện tích tầng hầm không được lớn hơn diện tích sử dụng.")

    else:

        input_data = pd.DataFrame([{
            'OverallQual': chat_luong,
            'GrLivArea': dien_tich_su_dung,
            'LotArea': dien_tich_dat,
            'TotalBsmtSF': dien_tich_tang_ham,
            'YearBuilt': nam_xay_dung,
            'FullBath': so_phong_tam,
            'TotRmsAbvGrd': tong_so_phong
        }])

        input_data = input_data[features]

        try:
            prediction = model.predict(input_data)[0]
            prediction = max(float(prediction), 0)

            st.success("DỰ ĐOÁN THÀNH CÔNG")

            st.subheader("Giá nhà dự đoán")

            st.markdown(
                f"<h2 style='text-align:center;'>"
                f"${prediction:,.2f}"
                f"</h2>",
                unsafe_allow_html=True
            )

            st.caption(f"Thuật toán sử dụng: {algorithm}")

        except Exception as e:
            st.error(f"Lỗi dự đoán: {e}")
