import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

import matplotlib.pyplot as plt
from numpy import linspace
from random import shuffle


'''
Step 1: Prepare dataset (same step)
'''
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
label_1 = classes.index('plane')
label_2 = classes.index('car')

train_set = datasets.CIFAR10(root='./cifar_data/', train=True, transform=transforms.ToTensor(), download=True)

# Use data with two labels
train_set.targets = torch.tensor(train_set.targets)
idx = (train_set.targets == label_1) + (train_set.targets == label_2)
train_set.data = train_set.data[idx]
train_set.targets = train_set.targets[idx]
train_set.targets[train_set.targets == label_1] = -1
train_set.targets[train_set.targets == label_2] = 1


test_set = datasets.CIFAR10(root='./cifar_data/', train=False, transform=transforms.ToTensor(), )

# Use data with two labels
test_set.targets = torch.tensor(test_set.targets)
idx = (test_set.targets == label_1) + (test_set.targets == label_2)
test_set.data = test_set.data[idx]
test_set.targets = test_set.targets[idx]
test_set.targets[test_set.targets == label_1] = -1
test_set.targets[test_set.targets == label_2] = 1


'''
Step 2: Define the neural network class. (LOOK HERE)
'''
class MLP4(nn.Module) :
    '''
    Initialize model
        input_dim : dimension of given input data
    '''
    # CIFAR-10 data is 32*32 images with 3 RGB channels
    def __init__(self, input_dim=3*32*32) :
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim//2, bias=True)
        self.linear2 = nn.Linear(input_dim//2, input_dim//4, bias=True)
        self.linear3 = nn.Linear(input_dim//4, input_dim//8, bias=True)
        self.linear4 = nn.Linear(input_dim//8, 1, bias=True)

    ''' forward given input x '''
    def forward(self, x) :
        x = x.float().view(-1, 3*32*32)
        x = nn.functional.relu(self.linear(x))
        x = nn.functional.relu(self.linear2(x))
        x = nn.functional.relu(self.linear3(x))
        x = self.linear4(x)
        return x


'''
Step 3: Create the model, specify loss function and optimizer. (LOOK HERE)
'''
model = MLP4()  # Define Neural Network Models

def logistic_loss(output, target):
    return torch.mean(-torch.nn.functional.logsigmoid(target.reshape(-1)*output.reshape(-1)))
loss_function = logistic_loss   # Specify loss function

optimizer = torch.optim.SGD(model.parameters(), lr=3*1e-2)  # specify SGD with learning rate


'''
Step 4: Train model with SGD (same step)
'''
# Use DataLoader class
train_loader = DataLoader(dataset=train_set, batch_size=1024, shuffle=True)
import time
start = time.time()
# Train the model
for epoch in range(100) :
    for images, labels in train_loader :

        # Clear previously computed gradient
        optimizer.zero_grad()
        # then compute gradient with forward and backward passes
        train_loss = loss_function(model(images), labels.float())
        train_loss.backward()

        # perform SGD step (parameter update)
        optimizer.step()
end = time.time()
print(f"Time ellapsed in training is: {end - start}")


'''
Step 5: Test model (same step)
'''
test_loss, correct = 0, 0
misclassified_ind = []
correct_ind = []

# Test data
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

# Evaluate accuracy using test data
for ind, (image, label) in enumerate(test_loader) :

    # Forward pass
    output = model(image)

    # Calculate cumulative loss
    test_loss += loss_function(output, label.float()).item()

    # Make a prediction
    if output.item() * label.item() >= 0 :
        correct += 1
        correct_ind += [ind]
    else:
        misclassified_ind += [ind]

# Print out the results
print('[Test set] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss /len(test_loader), correct, len(test_loader),
        100. * correct / len(test_loader)))


'''
Step 6: Show some incorrectly classified images and some correctly classified ones (same step)
'''
# Misclassified images
shuffle(misclassified_ind)
fig = plt.figure(1, figsize=(15, 6))
fig.suptitle('Misclassified Figures', fontsize=16)

for k in range(3) :
    image = test_set.data[misclassified_ind[k]].astype('uint8')
    ax = fig.add_subplot(1, 3, k+1)
    true_label = test_set.targets[misclassified_ind[k]]

    if true_label == -1 :
        ax.set_title('True Label: {}\nPrediction: {}'.format(classes[label_1], classes[label_2]))
    else :
        ax.set_title('True Label: {}\nPrediction: {}'.format(classes[label_2], classes[label_1]))
    plt.imshow(image)
plt.savefig("result/2.6.1.png")
plt.show()

# Correctly classified images
shuffle(correct_ind)
fig = plt.figure(2, figsize=(15, 6))
fig.suptitle('Correctly-classified Figures', fontsize=16)

for k in range(3) :
    image = test_set.data[correct_ind[k]].astype('uint8')
    ax = fig.add_subplot(1, 3, k+1)
    true_label = test_set.targets[correct_ind[k]]

    if true_label == -1 :
        ax.set_title('True Label: {}\nPrediction: {}'.format(classes[label_1], classes[label_1]))
    else :
        ax.set_title('True Label: {}\nPrediction: {}'.format(classes[label_2], classes[label_2]))
    plt.imshow(image)
plt.savefig("result/2.6.2.png")
plt.show()


'''
Step 7: Inspect model parameters
'''

print("MLP model parameters")
param_num = 0
for parameter in model.parameters():
    print(parameter.shape)
    param_num += parameter.numel()
print(f"MLP has a total of {param_num} parameters.")