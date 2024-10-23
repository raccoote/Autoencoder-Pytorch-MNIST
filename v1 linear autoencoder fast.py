import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random
import time
epochs = 20
batch_size = 64


class Autoencoder(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=128, latent_dim=64):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # sigmoid for range [0, 1] if input is normalized
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# function to prepare data for a given pair of digits
def prepare_data(input_digit, output_digit, train=True):
    mnist_data = datasets.MNIST(root="./data", train=train, transform=transform, download=True)
    data_input = mnist_data.data[(mnist_data.targets == input_digit)]
    data_output = mnist_data.data[(mnist_data.targets == output_digit)]

    min_samples = min(len(data_input), len(data_output))
    data_input = data_input[:min_samples].view(-1, 28*28).float() / 255.0
    data_output = data_output[:min_samples].view(-1, 28*28).float() / 255.0

    return data_input, data_output



start_time = time.time()
losses_per_pair = {}  # to store losses for each pair
# train for each digit pair
autoencoder_models = {}
for input_digit in range(10):
    output_digit = (input_digit + 1) % 10  # output digit is the next digit (wrapping around from 9 to 0)

    train_input, train_output = prepare_data(input_digit, output_digit, train=True)
    test_input, test_output = prepare_data(input_digit, output_digit, train=False)

    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    pair_losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(0, len(train_input), batch_size):
            inputs = train_input[i:i+batch_size]
            outputs = train_output[i:i+batch_size]
            optimizer.zero_grad()
            predicted = model(inputs)
            loss = criterion(predicted, outputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        pair_losses.append(epoch_loss)
        if (epoch+1)%10==0:
            print(f"Pair [{input_digit}-{output_digit}] - Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    losses_per_pair[(input_digit, output_digit)] = pair_losses
    autoencoder_models[(input_digit, output_digit)] = model  # save the trained model for this digit pair


end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time} secs")
# plotting loss 
plt.figure(figsize=(10, 6))
for pair, losses in losses_per_pair.items():
    plt.plot(range(1, epochs + 1), losses, label=f"Pair {pair}")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch for Each Digit Pair')
plt.legend()
plt.grid(True)
plt.show()



# show image results
mnist_test = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
model = autoencoder_models[(input_digit, output_digit)]
model.eval()
num_examples = 7
random_indices = np.random.choice(len(mnist_test), num_examples, replace=False)
plt.figure(figsize=(12, 6))

for i, idx in enumerate(random_indices):
    original_img = mnist_test[idx][0].numpy().reshape(28, 28) * 255.0
    plt.subplot(2, num_examples, i + 1)
    plt.imshow(original_img, cmap='gray')
    plt.title(f"Original {mnist_test[idx][1]}")
    plt.axis('off')

    input_digit = mnist_test[idx][1]
    output_digit = (input_digit + 1) % 10
    
    model = autoencoder_models[(input_digit, output_digit)]
    inputs = mnist_test[idx][0].view(-1, 28*28).float() / 255.0
    predicted = model(inputs.unsqueeze(0)).cpu().detach().numpy().reshape(28, 28) * 255.0

    plt.subplot(2, num_examples, num_examples + i + 1)
    plt.imshow(predicted, cmap='gray')
    plt.title(f"Reconstructed {output_digit}")
    plt.axis('off')

plt.tight_layout()
plt.show()
