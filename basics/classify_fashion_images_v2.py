from os import path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import transforms, TrivialAugmentWide

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

# training_data_transform = transforms.Compose([
#     transforms.ToTensor()
# ])
#
# eval_data_transform = transforms.Compose([
#     transforms.ToTensor()
# ])

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
MODEL_PATH = "./classify_fashion_images_latest_v2_model.pt"


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


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # self._flatten = nn.Flatten()
        self.convolutional_relu_stack = nn.Sequential(

            # 1st Convolutional Layer
            # Create 32 output channels from 1 gray channel
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
            # Flatten all dimensions except batch (start_dim = 1 by default)
            nn.Flatten(),
            nn.Linear(1600, 128),
            nn.ReLU(),
            # Zero some of the inputs to prevent co-adaption of neurons
            nn.Dropout(0.3),
            # Reshape to get the output features
            nn.Linear(128, 10),

            # Simple example
            # nn.Linear(IMAGE_SIZE, LAYER_SIZE),
            # nn.ReLU(),
            # nn.Linear(LAYER_SIZE, LAYER_SIZE),
            # nn.ReLU(),
            # nn.Linear(LAYER_SIZE, len(FEATURE_LABELS))
        )

    def forward(self, x):
        # x = self._flatten(x)
        logits = self.convolutional_relu_stack(x)
        return logits


class Runner():
    def __init__(self, model, optimizer, loss_fn, device, training_data, test_data, batch_size):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.training_dataloader = DataLoader(training_data, batch_size)
        self.eval_dataloader = DataLoader(test_data, batch_size)

    def train(self):
        self.model.train()
        for batch, (X, y) in enumerate(self.training_dataloader):
            # Move X,y to device
            X = X.to(self.device)
            y = y.to(self.device)
            # Compute prediction and loss
            prediction = self.model(X)
            loss = self.loss_fn(prediction, y)
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"Loss: {loss:>7f} [{current:>5d}/{len(self.training_dataloader.dataset):>5d}]")

    def evaluate(self):
        self.model.eval()
        num_batches = len(self.eval_dataloader)
        eval_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in self.eval_dataloader:
                # Move X,y to device
                X = X.to(self.device)
                y = y.to(self.device)
                prediction = self.model(X)
                eval_loss += self.loss_fn(prediction, y).item()
                correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

        eval_loss /= num_batches
        correct /= len(self.eval_dataloader.dataset)
        print(f"\nEvaluation Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {eval_loss:>8f}\n")

    def predict(self, img):
        self.model.eval()
        with torch.no_grad():
            X = img.to(self.device)
            # Create batch of one
            X = torch.reshape(X, (1, 1, 28, 28))
            prediction = self.model(X)
            return prediction

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = NeuralNetwork().to(device)

has_model_weights = path.isfile(MODEL_PATH)
if has_model_weights:
    print("Loading latest model from disk")
    model.load_state_dict(torch.load(MODEL_PATH))

loss_fn = nn.CrossEntropyLoss().to(device)

# https://towardsdatascience.com/7-tips-to-choose-the-best-optimizer-47bb9c1219e
# Stochastic Gradient Descent
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# Adam (Best among adaptive optimizers for most use cases)
# Less fine tuning needed
learning_rate = 1e-2
weight_decay = 1e-5
optimizer = torch.optim.Adam(model.parameters())

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

batch_size = 64
epochs = int(1000/(1 + AUGMENT_AMOUNT))

runner = Runner(model, optimizer, loss_fn, device, training_data, eval_data, batch_size)

# Enable cuDNN auto-tuner
torch.backends.cudnn.benchmark = True

if should_train:
    for t in range(epochs):
        print(f"Epoch {t+1} [{t + 1}/{epochs}]\n-------------------------------")
        runner.train()
        runner.evaluate()
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
    logits = runner.predict(img)
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
