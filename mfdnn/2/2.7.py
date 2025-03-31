import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import matplotlib.pyplot as plt


'''
Step 1: Load the entire MNIST dataset (LOOK HERE)
'''

train_set = datasets.MNIST(root='./mnist_data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_set = datasets.MNIST(root='./mnist_data/',
                              train=False,
                              transform=transforms.ToTensor())


'''
Step 2: Since there are 10 classes, the output should be 10. (LOOK HERE)
'''
class softmax(nn.Module) :
    def __init__(self, input_dim=28*28) :
        super().__init__()
        self.linear = nn.Linear(input_dim, 10, bias=True)

    def forward(self, x) :
        return self.linear(x.float().view(-1, 28*28))


'''
Step 3: Create the model, specify loss function and optimizer (LOOK HERE)
'''
model = softmax()
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


'''
Step 4: Train model with SGD (same step)
'''
train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)

import time
start = time.time()
for epoch in range(5) :
    for images, labels in train_loader :
        optimizer.zero_grad()
        train_loss = loss_function(model(images), labels)
        train_loss.backward()
        optimizer.step()
end = time.time()
print(f"Time ellapsed in training is: {end - start}")


'''
Step 5: Test model (Evaluate the accuracy)
'''
test_loss, correct = 0, 0

test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

for ind, (image, label) in enumerate(test_loader) :
    output = model(image)
    test_loss += loss_function(output, label).item()
    pred = output.max(1, keepdim=True)[1]
    correct += pred.eq(label.view_as(pred)).sum().item()


print('[Test set] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss /len(test_loader), correct, len(test_loader),
        100. * correct / len(test_loader)))