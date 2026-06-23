import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from src.data_utils import load_config, prepare_data, create_datasets

def run_training():
    config = load_config()
    data = prepare_data(config)
    training_ds, validation_ds = create_datasets(data, config)

    train_dl = training_ds.to_dataloader(train=True, batch_size=config['tft_params']['batch_size'], num_workers=0)
    val_dl = validation_ds.to_dataloader(train=False, batch_size=config['tft_params']['batch_size']*10, num_workers=0)

    # Khởi tạo model
    tft = TemporalFusionTransformer.from_dataset(
        training_ds,
        learning_rate=config['tft_params']['learning_rate'],
        hidden_size=config['tft_params']['hidden_size'],
        attention_head_size=config['tft_params']['attention_head_size'],
        dropout=config['tft_params']['dropout'],
        hidden_continuous_size=config['tft_params']['hidden_continuous_size'],
        loss=QuantileLoss(),
        reduce_on_plateau_patience=config['tft_params']['reduce_on_plateau_patience']
    )

    trainer = pl.Trainer(
        max_epochs=config['tft_params']['max_epochs'],
        accelerator="auto",
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10),
            ModelCheckpoint(dirpath=config['paths']['checkpoint_dir'], filename="best_model", monitor="val_loss", mode="min"),
            LearningRateMonitor(logging_interval="step")
        ],
    )

    tuner = Tuner(trainer)
    res = tuner.lr_find(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
    print(f"Suggested Learning Rate: {res.suggestion()}")
    tft.hparams.learning_rate = res.suggestion()

    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)

if __name__ == "__main__":
    run_training()
