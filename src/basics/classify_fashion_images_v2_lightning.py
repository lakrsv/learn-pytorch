from os import path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import transforms, TrivialAugmentWide
import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner

FEATURE_LABELS = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

TRAINING_DATA = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=None
)

AUGMENT_AMOUNT = 9

EVAL_DATA = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)
MODEL_PATH = "../../models/classify_fashion_images_latest_v2_lightning_model.pt"

# Enable cuDNN auto-tuner
torch.backends.cudnn.benchmark = True
# Configure CUDAfloat32 matmul precision
torch.set_float32_matmul_precision('medium')


class CustomImageDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.images = []
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        variant_img, img_label = self.images[idx]
        if self.transform:
            variant_img = self.transform(variant_img)
        if self.target_transform:
            img_label = self.target_transform(img_label)
        return variant_img, img_label

    def add(self, variant_img, img_label):
        self.images.append((variant_img, img_label))


class NeuralNetwork(pl.LightningModule):
    def __init__(self, train_dataset, eval_dataset, batch_size, learning_rate):
        super().__init__()
        self.model = nn.Sequential(
            # 1st Convolutional Layer
            nn.Conv2d(1, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),

            # 2nd Convolutional Layer
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            # Linear Layer
            nn.Flatten(),
            nn.Linear(1600, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )
        self.loss_module = nn.CrossEntropyLoss()

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(model.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        (images, labels) = batch
        # Compute prediction and loss
        predictions = self.model(images)
        loss = self.loss_module(predictions, labels)

        accuracy = (predictions.argmax(dim=-1) == labels).float().mean()

        self.log("train_accuracy", accuracy, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=self.batch_size)

    def validation_step(self, batch, batch_idx):
        (images, labels) = batch
        predictions = self.model(images).argmax(dim=-1)
        accuracy = (labels == predictions).float().mean()

        self.log("validation_accuracy", accuracy, on_epoch=True, prog_bar=True)


state_dict = None
has_model_weights = path.isfile(MODEL_PATH)
if has_model_weights:
    print("Loading latest model from disk")
    state_dict = torch.load(MODEL_PATH)

training_data = TRAINING_DATA
eval_data = EVAL_DATA

should_train = True
if has_model_weights:
    should_train = input('Would you like to train the model? (y/N)\n').lower() == 'y'
if should_train:
    AUGMENT_AMOUNT = AUGMENT_AMOUNT if input("Would you like to generate augmented images? (y/N)\n") == 'y' else 0
    augmented_training_data = CustomImageDataset(transform=transforms.ToTensor())
    augmenter = TrivialAugmentWide()
    original_image_amount = len(TRAINING_DATA)
    target_image_amount = len(TRAINING_DATA) + len(TRAINING_DATA) * AUGMENT_AMOUNT

    if AUGMENT_AMOUNT > 0:
        print(f"Creating augmented images from training data set, original length: {original_image_amount} images, "
              f"target length: {target_image_amount} images")

    for i in range(len(TRAINING_DATA)):
        image, label = TRAINING_DATA[i]
        augmented_training_data.add(image, label)
        for j in range(AUGMENT_AMOUNT):
            augmented_image = augmenter(image)
            augmented_training_data.add(augmented_image, label)
            if len(augmented_training_data) % len(TRAINING_DATA) == 0:
                print(f"Progress [{len(augmented_training_data)}/{target_image_amount}]")
    training_data = augmented_training_data

learning_rate = 1e-3
batch_size = 1024
epochs = int(1000 / (1 + AUGMENT_AMOUNT))

model = NeuralNetwork(training_data, eval_data, batch_size, learning_rate)
if state_dict is not None:
    model.load_state_dict(state_dict)

trainer = pl.Trainer(max_epochs=epochs)
tuner = Tuner(trainer)
tuner.lr_find(model)

if should_train:
    trainer.fit(model=model)
    print("Done training model")
    torch.save(model.state_dict(), MODEL_PATH)

print("Rendering model evaluation results")

# Render it
figure = plt.figure(figsize=(14, 14))
cols, rows = 6, 6
total, correct, average_confidence = cols * rows, 0, 0.0
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(eval_data), size=(1,)).item()
    img, label = eval_data[sample_idx]
    img = torch.reshape(img, (1, 1, 1, 28, 28))
    logits = trainer.predict(model, img)[0]
    pred_probab = nn.Softmax(dim=1)(logits)
    label_prediction_confidence = pred_probab.amax(1)
    label_prediction = pred_probab.argmax(1)

    average_confidence += label_prediction_confidence.item()

    actual_label_name = FEATURE_LABELS[label]
    predicted_label_name = FEATURE_LABELS[label_prediction.item()]

    prediction_correct = actual_label_name == predicted_label_name
    if prediction_correct:
        correct += 1

    figure.add_subplot(rows, cols, i)
    color = ('blue' if prediction_correct else 'red')
    plt.title(f"{(label_prediction_confidence.item() * 100):>0.1f}% {predicted_label_name}", fontdict={'color': color})
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

average_confidence /= total
print(f"Average Confidence: {(average_confidence * 100):>0.1f}% Correct: [{correct}/{total}]")

plt.show()
