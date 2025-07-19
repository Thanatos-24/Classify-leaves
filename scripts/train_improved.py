from dataloader import preprocess_image_label, preprocess_label
from torch.utils.data import DataLoader, Dataset
from dataprocess import train_valid_dataset, test_dataset
import hydra
import sys
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from net.Resnet18 import ResNet
from net.alexnet import AlexNet
import os


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(config):
    config = config["config"]
    
    # 添加早停机制
    # early_stopping = EarlyStopping(
    #     monitor="val_loss",
    #     patience=20,  # 增加patience，给模型更多时间
    #     mode="min",
    #     verbose=True
    # )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=config.hydra_path,
        filename="best-checkpoint",
        save_top_k=1,
        mode="min",
        verbose=True
    )
    
    train_image = preprocess_image_label(config.Datasetconfig.image_root, sigmaX=config.Datasetconfig.sigmaX)
    train_label = preprocess_label(config.Datasetconfig.image_label, mapping_path=config.Datasetconfig.mapping_path)
    train_ds = train_valid_dataset(train_image, train_label, mode='train')
    val_ds = train_valid_dataset(train_image, train_label, mode='valid')
    print("train_ds size:", len(train_ds))
    print("val_ds size:", len(val_ds))

    train_dl = DataLoader(
        train_ds,
        batch_size=config.Datasetconfig.batch_size,
        shuffle=True,
        num_workers=config.Datasetconfig.num_workers,
        pin_memory=True
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config.Datasetconfig.batch_size,
        shuffle=False,
        num_workers=config.Datasetconfig.num_workers,
        pin_memory=True
    )

    model = ResNet(
        **config["resnet18"]
    )
    logger = pl.loggers.CSVLogger(save_dir=config.hydra_path, name="logs")
    trainer = pl.Trainer(
        **config["trainer"],
        callbacks=[checkpoint_callback], 
        default_root_dir=config.hydra_path,
        logger=logger,
        log_every_n_steps=50
    )

    trainer.fit(
        model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl
    )
    print("Training complete.")

if __name__ == "__main__":
    main() 