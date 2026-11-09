
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# 1. CẤU HÌNH TRANG
# ==========================================

st.set_page_config(
    page_title="Dự đoán giá nhà",
    page_icon=None,
    layout="centered"
)

st.title("DỰ ĐOÁN GIÁ NHÀ")
st.write("Ứng dụng sử dụng hai thuật toán ANN và K-means.")

# ==========================================
# 2. ĐƯỜNG DẪN DỮ LIỆU
# ==========================================

TRAIN_PATH = "Nhom2_GiaNha_train.xlsx"
TEST_PATH = "Nhom2_GiaNha_test.xlsx"

TARGET = "GiaNha"

# Hỗ trợ hai cách viết tên cột tầng hầm
BASEMENT_NAMES = [
    "DienTichTangHam",
    "DienTichTangHao"
]

FEATURES_BASE = [
    "DienTichDat",
    "ChatLuongTongThe",
    "NamXayDung",
    "DienTichSuDung",
    "SoPhongTam",
    "TongSoPhong"
]


# ==========================================
# 3. ĐỌC DỮ LIỆU VÀ HUẤN LUYỆN MÔ HÌNH
# ==========================================

@st.cache_resource
def train_models():

    train_df = pd.read_excel(TRAIN_PATH)
    test_df = pd.read_excel(TEST_PATH)

    # Tìm tên cột diện tích tầng hầm thực tế
    basement_col = next(
        (c for c in BASEMENT_NAMES if c in train_df.columns),
        None
    )

    if basement_col is None:
        raise ValueError(
            "Không tìm thấy cột DienTichTangHam "
            "hoặc DienTichTangHao trong file Train."
        )

    if basement_col not in test_df.columns:
        raise ValueError(
            f"File Test thiếu cột {basement_col}."
        )

    features = [
        FEATURES_BASE[0],
        FEATURES_BASE[1],
        "NamXayDung",
        basement_col,
        "DienTichSuDung",
        "SoPhongTam",
        "TongSoPhong"
    ]

    required = features + [TARGET]

    for col in required:
        if col not in train_df.columns:
            raise ValueError(f"Train thiếu cột: {col}")

    for col in required:
        if col not in test_df.columns:
            raise ValueError(f"Test thiếu cột: {col}")

    # Chuyển dữ liệu về dạng số
    for col in required:
        train_df[col] = pd.to_numeric(
            train_df[col], errors="coerce"
        )
        test_df[col] = pd.to_numeric(
            test_df[col], errors="coerce"
        )

    # Loại bỏ dòng thiếu giá nhà
    train_df = train_df.dropna(subset=[TARGET])
    test_df = test_df.dropna(subset=[TARGET])

    X_train = train_df[features]
    y_train = train_df[TARGET]

    X_test = test_df[features]
    y_test = test_df[TARGET]

    # Kiểm tra dữ liệu
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Dữ liệu Train hoặc Test không hợp lệ.")

    # --------------------------------------
    # MÔ HÌNH ANN
    # --------------------------------------

    ann_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ann", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=2000,
            early_stopping=True,
            random_state=42
        ))
    ])

    model_ann = TransformedTargetRegressor(
        regressor=ann_pipeline,
        transformer=StandardScaler()
    )

    model_ann.fit(X_train, y_train)

    # Đánh giá ANN
    y_pred = model_ann.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # --------------------------------------
    # MÔ HÌNH K-MEANS
    # --------------------------------------

    kmeans_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    X_train_scaled = kmeans_preprocessor.fit_transform(X_train)

    if len(X_train) < 3:
        raise ValueError("Cần ít nhất 3 mẫu Train để phân cụm.")

    kmeans_model = KMeans(
        n_clusters=3,
        init="k-means++",
        random_state=42,
        n_init=10
    )

    kmeans_model.fit(X_train_scaled)

    return {
        "features": features,
        "train": X_train,
        "test": X_test,
        "ann": model_ann,
        "kmeans": kmeans_model,
        "preprocessor": kmeans_preprocessor,
        "metrics": (mae, mse, rmse, r2),
        "train_size": len(X_train),
        "test_size": len(X_test)
    }


# ==========================================
# 4. KHỞI TẠO MÔ HÌNH
# ==========================================

try:
    models = train_models()

    model_ann = models["ann"]
    kmeans_model = models["kmeans"]
    preprocessor = models["preprocessor"]
    features = models["features"]

except Exception as e:
    st.error(f"Không thể khởi tạo mô hình: {e}")
    st.stop()


# ==========================================
# 5. HIỂN THỊ THÔNG TIN MÔ HÌNH
# ==========================================

st.success("Hai mô hình ANN và K-means đã sẵn sàng.")

st.write(
    f"Số mẫu Train: {models['train_size']:,} | "
    f"Số mẫu Test: {models['test_size']:,}"
)

st.divider()


# ==========================================
# 6. NHẬP THÔNG SỐ CĂN NHÀ
# ==========================================

st.subheader("NHẬP THÔNG SỐ CĂN NHÀ")

train_data = models["train"]

# Giá trị trung vị làm giá trị mặc định
defaults = train_data.median()

with st.form("house_form"):

    col1, col2 = st.columns(2)

    with col1:

        dien_tich_dat = st.number_input(
            "Diện tích đất",
            value=float(defaults["DienTichDat"]),
            min_value=0.0
        )

        chat_luong = st.number_input(
            "Chất lượng tổng thể",
            value=float(defaults["ChatLuongTongThe"]),
            min_value=0.0
        )

        nam_xay_dung = st.number_input(
            "Năm xây dựng",
            value=int(defaults["NamXayDung"]),
            min_value=0,
            step=1
        )

        dien_tich_ham = st.number_input(
            "Diện tích tầng hầm",
            value=float(defaults[features[3]]),
            min_value=0.0
        )

    with col2:

        dien_tich_su_dung = st.number_input(
            "Diện tích sử dụng",
            value=float(defaults["DienTichSuDung"]),
            min_value=0.0
        )

        so_phong_tam = st.number_input(
            "Số phòng tắm",
            value=int(defaults["SoPhongTam"]),
            min_value=0,
            step=1
        )

        tong_so_phong = st.number_input(
            "Tổng số phòng",
            value=int(defaults["TongSoPhong"]),
            min_value=0,
            step=1
        )

    submitted = st.form_submit_button(
        "DỰ ĐOÁN VÀ PHÂN CỤM",
        use_container_width=True
    )


# ==========================================
# 7. DỰ ĐOÁN VÀ PHÂN CỤM
# ==========================================

if submitted:

    if dien_tich_dat <= 0:
        st.error("Diện tích đất phải lớn hơn 0.")

    elif dien_tich_su_dung <= 0:
        st.error("Diện tích sử dụng phải lớn hơn 0.")

    elif dien_tich_ham > dien_tich_su_dung:
        st.error("Diện tích tầng hầm không được lớn hơn diện tích sử dụng.")

    elif so_phong_tam > tong_so_phong:
        st.error("Số phòng tắm không được lớn hơn tổng số phòng.")

    else:

        du_lieu_moi = pd.DataFrame([{
            "DienTichDat": dien_tich_dat,
            "ChatLuongTongThe": chat_luong,
            "NamXayDung": nam_xay_dung,
            features[3]: dien_tich_ham,
            "DienTichSuDung": dien_tich_su_dung,
            "SoPhongTam": so_phong_tam,
            "TongSoPhong": tong_so_phong
        }], columns=features)

        # Dự đoán giá nhà bằng ANN
        gia_du_doan = model_ann.predict(du_lieu_moi)[0]

        # Phân cụm bằng K-means
        du_lieu_scaled = preprocessor.transform(du_lieu_moi)

        cum_du_doan = kmeans_model.predict(
            du_lieu_scaled
        )[0]

        st.divider()

        st.subheader("KẾT QUẢ DỰ ĐOÁN")

        st.write("Giá nhà dự đoán bằng thuật toán ANN:")

        st.metric(
            label="Giá nhà dự đoán",
            value=f"{gia_du_doan:,.2f}"
        )

        st.caption(
            "Đơn vị giá được giữ nguyên theo dữ liệu gốc."
        )

        st.subheader("KẾT QUẢ PHÂN CỤM K-MEANS")

        st.metric(
            label="Căn nhà thuộc cụm",
            value=f"Cụm {cum_du_doan}"
        )

        st.write(
            "Mô hình đã so sánh các đặc điểm của căn nhà "
            "với tâm của 3 cụm để xác định nhóm phù hợp."
        )


# ==========================================
# 8. ĐÁNH GIÁ MÔ HÌNH ANN
# ==========================================

st.divider()

st.subheader("ĐÁNH GIÁ MÔ HÌNH ANN")

mae, mse, rmse, r2 = models["metrics"]

c1, c2 = st.columns(2)
c3, c4 = st.columns(2)

c1.metric("MAE", f"{mae:,.2f}")
c2.metric("MSE", f"{mse:,.2f}")
c3.metric("RMSE", f"{rmse:,.2f}")
c4.metric("R²", f"{r2:.4f}")

st.caption(
    "MAE, MSE, RMSE càng nhỏ thường càng tốt. "
    "R² càng gần 1 thường cho thấy mô hình phù hợp hơn."
)

# ==========================================
# 9. THỐNG KÊ PHÂN CỤM TRAIN
# ==========================================

st.divider()

st.subheader("THỐNG KÊ PHÂN CỤM K-MEANS")

train_clusters = kmeans_model.labels_

cluster_counts = pd.Series(
    train_clusters
).value_counts().sort_index()

cluster_table = pd.DataFrame({
    "Cụm": [f"Cụm {i}" for i in cluster_counts.index],
    "Số lượng căn nhà": cluster_counts.values
})

st.dataframe(
    cluster_table,
    use_container_width=True,
    hide_index=True
)

st.bar_chart(
    cluster_table.set_index("Cụm")
)
