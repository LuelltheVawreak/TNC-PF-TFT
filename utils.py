import pandas as pd
import json
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer

def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)

def prepare_data(config):
    df_macro = pd.read_csv(config['paths']['macro_data'])
    df_province = pd.read_csv(config['paths']['province_data'])

    df_macro['date'] = pd.to_datetime(df_macro['date'], format='mixed')
    df_province['date'] = pd.to_datetime(df_province['date'], format='mixed')

    cols_to_use = df_macro.columns.difference(df_province.columns).tolist()
    cols_to_use.append('date')
    df = pd.merge(df_province, df_macro[cols_to_use], on='date', how='left')
    
    df = df.sort_values(by=['province_id', 'time_idx']).reset_index(drop=True)
    df = df.groupby('province_id').ffill().bfill()

    # Chuyển đổi kiểu dữ liệu phân loại
    cat_cols = ['province_id', 'month', 'quarter', 'day_of_week', 'is_vietnam_holiday']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).astype("category")
            
    df.to_csv(config['paths']['processed_data'], index=False)
    return df

def create_datasets(data, config):
    params = config['tft_params']
    feat = config['features']
    
    training_cutoff = data["time_idx"].max() - params['max_prediction_length']

    training_ds = TimeSeriesDataSet(
        data[data.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=feat['target'],
        group_ids=feat['group_ids'],
        min_encoder_length=params['max_encoder_length'] // 2,
        max_encoder_length=params['max_encoder_length'],
        min_prediction_length=1,
        max_prediction_length=params['max_prediction_length'],
        static_categoricals=feat['static_categoricals'],
        static_reals=feat['static_reals'],
        time_varying_known_categoricals=feat['time_varying_known_categoricals'],
        time_varying_known_reals=feat['time_varying_known_reals'],
        time_varying_unknown_reals=feat['time_varying_unknown_reals'],
        # HỌC TỪ REPO: GroupNormalizer giúp xử lý giá khác nhau giữa các tỉnh
        target_normalizer=GroupNormalizer(
            groups=feat['group_ids'], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation_ds = TimeSeriesDataSet.from_dataset(
        training_ds, data, predict=True, stop_random_sampling=True
    )
    
    return training_ds, validation_ds
