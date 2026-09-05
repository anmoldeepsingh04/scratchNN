import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

start = time.perf_counter()

# configurations
batch_size = 10
lr = 0.001
epochs = 5

device = torch.device("cpu") #torch.device("mps" if torch.mps.is_available() else "cpu")

# model definition
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

start_data = time.perf_counter()

# dataset loading
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root = "data",
    train = True,
    download = True,
    transform = transform
)

test_dataset = datasets.MNIST(
    root = "data",
    train = False,
    download = True,
    transform = transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    shuffle = True
)

test_loader = DataLoader(
    test_dataset,
    batch_size = batch_size,
    shuffle = False
)

end_data = time.perf_counter()

# loss and optimizer definition
model = NeuralNetwork().to(device)
loss_fxn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = lr)

# training loop
def train():
    model.train()

    for epoch in range(epochs):
        cum_loss = 0.0
        # start_time = time.perf_counter()

        for batch_id, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fxn(outputs, labels)
            loss.backward()
            optimizer.step()
            cum_loss += loss.item()
        # epoch_time = time.perf_counter() - start_time

        # print(
        #     f"Epoch [{epoch + 1}/{epochs}]\n"
        #     f"Loss: {cum_loss / len(train_loader):.6f}\n"
        #     f"Time: {epoch_time:.3f}s\n"
        # )
    torch.save(model.state_dict(), "models/mnist_pytorch.pth")

# testing loop
def test():
    model.eval()
    correct = 0
    total = 0
    cum_loss = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fxn(outputs, labels)
            cum_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # accuracy = 100.0 * correct/total
    # average_loss = cum_loss/len(test_loader)

    # print(f"Test accuracy: {accuracy:0.2f}\n")
    # print(f"Test loss: {average_loss:0.6f}\n")

# main function
if __name__ == "__main__":
    print(f"Using device: {device}\n")
    start_train = time.perf_counter()
    train()
    end_train = time.perf_counter()

    start_test = time.perf_counter()
    test()
    end_test = time.perf_counter()
    # for using saved models
    # model = NeuralNetwork()
    # model.load_state_dict(torch.load("models/mnist_pytorch.pth", weights_only=True))
    # model.eval()
    end = time.perf_counter()

    data_loading_time = end_data - start_data
    train_time = end_train - start_train
    test_time = end_test - start_test
    exec_time = end - start
    print(f"Data loading time: {data_loading_time:.3f} \n",
          f"Training time: {train_time:.3f} \n",
          f"Inference time: {test_time:.3f} \n",
          f"Total time: {exec_time:.3f} \n")
      