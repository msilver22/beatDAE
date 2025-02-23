import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from src.data_module import MNISTDataModule
from src.model import DAE

def main():
    
    dm = MNISTDataModule(batch_size=32)

    model = DAE()

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='dae-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
    )

    trainer = Trainer(
        max_epochs=100,
        accelerator='auto',
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    trainer.fit(model, dm)
    torch.save(model.state_dict(), 'dae_weights.pth')

if __name__ == '__main__':
    main()
