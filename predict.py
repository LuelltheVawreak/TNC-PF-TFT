from pytorch_forecasting import TemporalFusionTransformer
import pandas as pd
from src.data_utils import load_config

def make_prediction():
    config = load_config()
    model = TemporalFusionTransformer.load_from_checkpoint(f"{config['paths']['checkpoint_dir']}/best_model.ckpt")
    data = pd.read_csv(config['paths']['processed_data'])
    
    # Lấy 90 ngày cuối cùng để dự báo 30 ngày tiếp theo
    new_prediction = model.predict(data, mode="prediction")
    print("Dự báo giá 30 ngày tới (theo từng tỉnh):")
    print(new_prediction)

if __name__ == "__main__":
    make_prediction()
