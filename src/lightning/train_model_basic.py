import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt


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
        self.encoder = encoder
        self.decoder = decoder
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


torch.set_float32_matmul_precision('medium')

checkpoint_dir = '../../models/train_model_basic/'
checkpoint_name = None # 'lightning_logs/version_7/checkpoints/epoch=9-step=9380.ckpt'
batch_size = 64
max_epochs = 10

dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset, batch_size)

autoencoder = LitAutoEncoder(Encoder(), Decoder())

if checkpoint_name is not None:
    autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint_dir + checkpoint_name, encoder=Encoder(), decoder=Decoder())

trainer = pl.Trainer(max_epochs=max_epochs, default_root_dir=checkpoint_dir)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)

# Show result
with torch.no_grad():
    autoencoder.eval()
    sample_idx = torch.randint(len(dataset), size=(1,)).item()
    (img, label) = dataset[sample_idx]

    rows, cols = (1, 2)
    figure = plt.figure(figsize=(8, 8))

    figure.add_subplot(rows, cols, 1)
    plt.title("Original")
    plt.imshow(torch.squeeze(img.clone()))

    figure.add_subplot(rows, cols, 2)
    plt.title("New")
    new_img = img.clone().reshape(1, 1, 28, 28)
    new_img = trainer.predict(autoencoder, new_img)[0]
    new_img = new_img.view((28, 28))
    plt.imshow(new_img)

    plt.show()
