import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pytorch_forecasting import TemporalFusionTransformer
from src.data_utils import load_config

config = load_config()
st.set_page_config(page_title="Coffee Forecast Dashboard", layout="wide")

@st.cache_resource
def get_model():
    return TemporalFusionTransformer.load_from_checkpoint(f"{config['paths']['checkpoint_dir']}/best_model.ckpt")

st.title("☕ Dự Báo Giá Cà Phê Robusta")

try:
    df = pd.read_csv(config['paths']['processed_data'])
    df['date'] = pd.to_datetime(df['date'])
    model = get_model()
    
    province = st.sidebar.selectbox("Chọn tỉnh thành:", df['province_id'].unique())
    
    # Lấy dữ liệu 90 ngày cuối để làm đầu vào dự báo
    df_province = df[df['province_id'] == province].sort_values('date')
    
    if st.sidebar.button("Thực hiện dự báo"):
        # Dự báo
        raw_predictions = model.predict(df, mode="prediction")
        idx = list(df['province_id'].unique()).index(province)
        forecast_values = raw_predictions[idx].numpy()
        
        # Tạo chuỗi thời gian tương lai
        future_dates = pd.date_range(df_province['date'].max() + pd.Timedelta(days=1), periods=len(forecast_values))
        
        # Vẽ biểu đồ
        fig = go.Figure()
        # Đường giá lịch sử
        fig.add_trace(go.Scatter(x=df_province['date'].tail(60), y=df_province['robusta_vn_price'].tail(60), name="Giá lịch sử"))
        # Đường giá dự báo
        fig.add_trace(go.Scatter(x=future_dates, y=forecast_values, name="Giá dự báo", line=dict(dash='dash', color='red')))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Hiển thị bảng giá dự báo
        st.subheader("Bảng giá dự báo chi tiết")
        st.write(pd.DataFrame({"Ngày": future_dates, "Giá dự báo (VNĐ/kg)": forecast_values}))

except Exception as e:
    st.warning("Vui lòng đảm bảo đã chạy train.py thành công và có file best_model.ckpt trong thư mục models.")
    st.error(e)
