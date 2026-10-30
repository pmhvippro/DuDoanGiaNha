
import streamlit as st
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline


# Cau hinh trang
st.set_page_config(
    page_title="Dự đoán giá nhà",
    layout="centered"
)


# Du lieu
TRAIN_FILE = "Nhom2_GiaNha_train.xlsx"

FEATURES = [
    "BedroomAbvGr",
    "FullBath",
    "LotArea",
    "YearBuilt",
    "1stFlrSF",
    "2ndFlrSF",
    "OverallQual"
]

TARGET = "SalePrice"


# Ham huan luyen mo hinh
@st.cache_resource
def load_models():

    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(
            "Không tìm thấy file dữ liệu huấn luyện: "
            + TRAIN_FILE
        )

    data = pd.read_excel(TRAIN_FILE)

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "File Excel thiếu các cột: "
            + ", ".join(missing_columns)
        )

    data = data[required_columns].copy()

    for col in required_columns:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    data = data.dropna()

    if len(data) < 10:
        raise ValueError(
            "Dữ liệu huấn luyện không đủ."
        )

    X = data[FEATURES]
    y = data[TARGET]

    # Mo hinh ANN
    ann_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=1000,
            random_state=42
        ))
    ])

    # Mo hinh SVM
    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(
            kernel="rbf",
            C=100,
            gamma="scale"
        ))
    ])

    ann_model.fit(X, y)
    svm_model.fit(X, y)

    return {
        "ANN": ann_model,
        "SVM": svm_model
    }


# Giao dien chinh
st.title("DỰ ĐOÁN GIÁ NHÀ")

st.write(
    "Nhập các thông tin của căn nhà để dự đoán giá."
)


# Tai mo hinh
try:
    models = load_models()

except Exception as e:
    st.error(str(e))
    st.stop()


# Nhap thong tin nha
st.subheader("THÔNG TIN CĂN NHÀ")

bedroom = st.number_input(
    "Số phòng ngủ",
    min_value=0,
    max_value=20,
    value=3,
    step=1
)

fullbath = st.number_input(
    "Số phòng tắm",
    min_value=0,
    max_value=10,
    value=2,
    step=1
)

lotarea = st.number_input(
    "Diện tích đất",
    min_value=1,
    value=8000,
    step=100
)

yearbuilt = st.number_input(
    "Năm xây dựng",
    min_value=1800,
    max_value=2026,
    value=2000,
    step=1
)

overallqual = st.slider(
    "Chất lượng xây dựng",
    min_value=1,
    max_value=10,
    value=5
)


# Lua chon loai nha
st.subheader("LOẠI NHÀ")

house_type = st.selectbox(
    "Chọn loại nhà",
    [
        "Nhà có tầng 1 và tầng 2",
        "Nhà chỉ có tầng 1",
        "Nhà không có tầng 1 và tầng 2"
    ]
)


# Nhap dien tich tang
st.subheader("DIỆN TÍCH CÁC TẦNG")

firstflrsf = 0
secondflrsf = 0

if house_type == "Nhà có tầng 1 và tầng 2":

    firstflrsf = st.number_input(
        "Diện tích tầng 1",
        min_value=0,
        max_value=int(lotarea),
        value=min(1000, int(lotarea)),
        step=50,
        key="first_floor_both"
    )

    secondflrsf = st.number_input(
        "Diện tích tầng 2",
        min_value=0,
        max_value=int(lotarea),
        value=min(500, int(lotarea)),
        step=50,
        key="second_floor_both"
    )

elif house_type == "Nhà chỉ có tầng 1":

    firstflrsf = st.number_input(
        "Diện tích tầng 1",
        min_value=0,
        max_value=int(lotarea),
        value=min(1000, int(lotarea)),
        step=50,
        key="first_floor_only"
    )

    secondflrsf = 0

    st.write("Diện tích tầng 2: 0")

else:

    firstflrsf = 0
    secondflrsf = 0

    st.write("Diện tích tầng 1: 0")
    st.write("Diện tích tầng 2: 0")


# Chon mo hinh
st.subheader("MÔ HÌNH DỰ ĐOÁN")

model_name = st.selectbox(
    "Chọn mô hình dự đoán",
    ["ANN", "SVM"]
)


# Du doan gia nha
if st.button("DỰ ĐOÁN GIÁ NHÀ", use_container_width=True):

    input_data = pd.DataFrame(
        [[
            bedroom,
            fullbath,
            lotarea,
            yearbuilt,
            firstflrsf,
            secondflrsf,
            overallqual
        ]],
        columns=FEATURES
    )

    prediction = models[model_name].predict(
        input_data
    )[0]

    st.subheader("KẾT QUẢ DỰ ĐOÁN")

    st.write("Mô hình được sử dụng:", model_name)

    st.success(
        "Giá nhà dự đoán: {:,.0f}".format(prediction)
    )
