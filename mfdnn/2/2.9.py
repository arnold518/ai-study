import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
Step 1
'''
train_dataset = datasets.MNIST(root='./mnist_data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False,
                              transform=transforms.ToTensor())

'''
Step 2: Define the neural network class (LOOK HERE)
'''
class MLP4(nn.Module) :
    '''
    Initialize model
        input_dim : dimension of given input data
    '''
    def __init__(self, input_dim=28*28) :
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim//2, bias=True)
        self.linear2 = nn.Linear(input_dim//2, input_dim//4, bias=True)
        self.linear3 = nn.Linear(input_dim//4, input_dim//8, bias=True)
        self.linear4 = nn.Linear(input_dim//8, 10, bias=True)

    ''' forward given input x '''
    def forward(self, x) :
        x = x.float().view(-1, 28*28)
        x = nn.functional.relu(self.linear(x))
        x = nn.functional.relu(self.linear2(x))
        x = nn.functional.relu(self.linear3(x))
        x = self.linear4(x)
        return x
'''
Step 3 Instantiate model and send it to device (LOOK HERE)
'''
model = MLP4().to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)


'''
Step 4 Load batch, send it to device, and perform SGD (LOOK HERE)
'''
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)

import time
start = time.time()
for epoch in range(10) :
    for images, labels in train_loader :
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        train_loss = loss_function(model(images), labels)
        train_loss.backward()

        optimizer.step()
end = time.time()
print("Time ellapsed in training is: {}".format(end - start))



'''
Step 5 Load batch, send it to device, and peform test (LOOK HERE)
'''
test_loss, correct, total = 0, 0, 0

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1024, shuffle=False)

for images, labels in test_loader :
    images, labels = images.to(device), labels.to(device)

    output = model(images)
    test_loss += loss_function(output, labels).item()

    pred = output.max(1, keepdim=True)[1]
    correct += pred.eq(labels.view_as(pred)).sum().item()

    total += labels.size(0)

print('[Test set] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss /total, correct, total,
        100. * correct / total))
