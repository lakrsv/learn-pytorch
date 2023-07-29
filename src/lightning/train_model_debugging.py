import os

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        # Print input output layer dimensions
        self.example_input_array = torch.Tensor(1, 28*28)
        self.encoder = encoder
        self.decoder = decoder
        self.save_hyperparameters(ignore=['encoder', 'decoder'])

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("validation_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


torch.set_float32_matmul_precision('medium')

checkpoint_dir = '../../models/train_model_debugging/'
batch_size = 64
max_epochs = 100

transform = transforms.ToTensor()

train_dataset = MNIST(os.getcwd(), train=True, download=True, transform=transform)
test_dataset = MNIST(os.getcwd(), train=False, download=True, transform=transform)

train_dataset_size = int(len(train_dataset) * 0.8)
validation_dataset_size = int(len(train_dataset) * 0.2)

seed = torch.Generator().manual_seed(42)
train_dataset, validation_dataset = torch.utils.data.random_split(
    train_dataset,
    [train_dataset_size, validation_dataset_size],
    generator=seed)

autoencoder = LitAutoEncoder(Encoder(), Decoder())

# Fast Dev Run Example
trainer = pl.Trainer(max_epochs=max_epochs, default_root_dir=checkpoint_dir,
                     callbacks=[EarlyStopping(monitor="validation_loss", mode="min")], fast_dev_run=True)
trainer.fit(model=autoencoder,
            train_dataloaders=DataLoader(train_dataset, batch_size),
            val_dataloaders=DataLoader(validation_dataset, batch_size))

# # Fast Dev Run With 7 Batches
# trainer = pl.Trainer(max_epochs=max_epochs, default_root_dir=checkpoint_dir,
#                      callbacks=[EarlyStopping(monitor="validation_loss", mode="min")], fast_dev_run=7)
# trainer.fit(model=autoencoder,
#             train_dataloaders=DataLoader(train_dataset, batch_size),
#             val_dataloaders=DataLoader(validation_dataset, batch_size))
#
# # Use only 10% of training data and 1% of val data
# trainer = pl.Trainer(max_epochs=max_epochs, default_root_dir=checkpoint_dir,
#                      callbacks=[EarlyStopping(monitor="validation_loss", mode="min")], limit_train_batches=0.1, limit_val_batches=0.01)
# trainer.fit(model=autoencoder,
#             train_dataloaders=DataLoader(train_dataset, batch_size),
#             val_dataloaders=DataLoader(validation_dataset, batch_size))
#
# # Use 10 batches of train and 5 batches of val
# trainer = pl.Trainer(max_epochs=max_epochs, default_root_dir=checkpoint_dir,
#                      callbacks=[EarlyStopping(monitor="validation_loss", mode="min")], limit_train_batches=10, limit_val_batches=5)
# trainer.fit(model=autoencoder,
#             train_dataloaders=DataLoader(train_dataset, batch_size),
#             val_dataloaders=DataLoader(validation_dataset, batch_size))
#
# # Sanity check
# trainer = pl.Trainer(max_epochs=max_epochs, default_root_dir=checkpoint_dir,
#                      callbacks=[EarlyStopping(monitor="validation_loss", mode="min")], num_sanity_val_steps=2)
# trainer.fit(model=autoencoder,
#             train_dataloaders=DataLoader(train_dataset, batch_size),
#             val_dataloaders=DataLoader(validation_dataset, batch_size))

