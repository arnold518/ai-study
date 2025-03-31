# Binary Classification with Logistic Regression

$$
f_{a, b}(x)=
\mu\left(\left[\begin{array}{c}
0 \\
a^{\top} x+b
\end{array}\right]\right)=
\left[\begin{array}{c}
\frac{1}{1+e^{a^{\top} x+b}} \\
\frac{e^{a^{\top} x+b}}{1+e^{a^{\top} x+b}}
\end{array}\right]=\left[\begin{array}{c}
\frac{1}{1+e^{a^{\top} x+b}} \\
\frac{1}{1+e^{-\left(a^{\top} x+b\right)}}
\end{array}\right]\begin{array}{c}
= \mathbb{P}(y=-1) \\
= \mathbb{P}(y=+1)
\end{array}
$$

$$
\begin{gathered}
\underset{a \in \mathbb{R}^{p}, b \in \mathbb{R}}{\mathrm{minimize}} \sum_{i=1}^{N} D_{\mathrm{KL}}\left(\mathcal{P}\left(Y_{i}\right) \| f_{a, b}\left(X_{i}\right)\right) \\
\mathbb{\Updownarrow} \\
\underset{a \in \mathbb{R}^{p}, b \in \mathbb{R}}{\mathrm{minimize}} \sum_{i=1}^{N} H\left(\mathcal{P}\left(Y_{i}\right), f_{a, b}\left(X_{i}\right)\right)+(\text { terms independent of } a, b) \\
\mathbb{\Updownarrow} \\
\underset{a \in \mathbb{R}^{p}, b \in \mathbb{R}}{\mathrm{minimize}} \sum_{i=1}^{N} \log \left(1+\exp \left(-Y_{i}\left(a^{\top} X_{i}+b\right)\right)\right) \\
\mathbb{\Updownarrow} \\
\underset{a \in \mathbb{R}^{p}, b \in \mathbb{R}}{\mathrm{minimize}} \frac{1}{N} \sum_{i=1}^{N} - \mathrm{logsigmoid}\left(Y_{i}\left(a^{\top} X_{i}+b\right)\right)
\end{gathered}
$$


## [`2.1.py`](2.1.py)

Basic logistic regression with pytorch.

Classify images with label 4 and 9 from MNIST data.
Use $-\mathrm{logsigmoid}(x)$ as loss function.

- **Step 1** : Prepare dataset.

    ```py
    train_set = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
    test_set = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor())
    ```

- **Step 2** : Define the neural network class.

    ```py
    class LR(nn.Module) :
        # MNIST data is 28x28 images
        def __init__(self, input_dim=28*28) :
            super().__init__()
            self.linear = nn.Linear(input_dim, 1, bias=True)

        def forward(self, x) :
            return self.linear(x.float().view(-1, 28*28))
    ```

- **Step 3** : Create the model, specify loss function and optimizer.

    ```py
    model = LR()   # Define a Neural Network Model

    def logistic_loss(output, target):
        return -torch.nn.functional.logsigmoid(target*output)

    loss_function = logistic_loss   # Specify loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)   # specify SGD with learning rate
    ```

- **Step 4** : Train model with SGD.

    ```py
    for _ in range(1000) :
        # Sample a random data for training
        ind = randint(0, len(train_set.data)-1)
        image, label = train_set.data[ind], train_set.targets[ind]

        # Clear previously computed gradient
        optimizer.zero_grad()

        # then compute gradient with forward and backward passes
        train_loss = loss_function(model(image), label.float())
        train_loss.backward()

        # perform SGD step (parameter update)
        optimizer.step()
    ```

- **Step 5** : Test model. (Evaluate the accuracy)

    ```py
    # Evaluate accuracy using test data
    for ind in range(len(test_set.data)) :
        image, label = test_set.data[ind], test_set.targets[ind]

        # evaluate model
        output = model(image)

        # Calculate cumulative loss
        test_loss += loss_function(output, label.float()).item()

        # Make a prediction
        if output.item() * label.item() >= 0 :
            correct += 1
            correct_ind += [ind]
        else:
            misclassified_ind += [ind]
    ```

## [`2.2.py`](2.2.py)

Simplified code of [`2.1.py`](2.1.py) using dataloader utility.

The dataloader creates a iterator that end when all data has been processed.

Original data was `uint8` with `0 ~ 255`.
DataLoader scales the image by a factor of 255 and converts the data type to a float with range `0 ~ 1`.

- **Step 4** : Train model with SGD.

    ```py
    # Use DataLoader class (choose from below options)

    # 1. SGD
    from torch.utils.data import RandomSampler
    train_loader = DataLoader(dataset=train_set, batch_size=1, sampler=RandomSampler(train_set, replacement=True))

    # 2. cyclic SGD
    train_loader = DataLoader(dataset=train_set, batch_size=1)

    # 3. shuffled cyclic SGD
    train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)

    # Train the model
    for image, label in train_loader :
        # Clear previously computed gradient
        optimizer.zero_grad()

        # then compute gradient with forward and backward passes
        train_loss = loss_function(model(image), label.float())
        train_loss.backward()

        # perform SGD step (parameter update)
        optimizer.step()
    ```

## [`2.3.py`](2.3.py)

[`2.2.py`](2.2.py) code run with multiple epochs.

- **Step 4** : Train model with SGD.

    ```py
    train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True)

    # Train the model for 3 epochs
    for epoch in range(3) :
        for image, label in train_loader :
            # Clear previously computed gradient
            optimizer.zero_grad()

            # then compute gradient with forward and backward passes
            train_loss = loss_function(model(image), label.float())
            train_loss.backward()

            # perform SGD step (parameter update)
            optimizer.step()
    ```

## [`2.4.py`](2.4.py)

[`2.3.py`](2.3.py) code modified using batch update.

- **Step 3** : Create the model, specify loss function and optimizer.

    ```py
    def logistic_loss(output, target):
        # output has dim [B,1]
        # target has dim [B]
        # dimensions as is don't match!
        # convert output dim to [B]
        # conert target dim to [B]
        # elementwise product (* is elementwise product in Python)
        # after logsigmoid, dim is [B]
        # after mean, dim is [1]
        return torch.mean(-torch.nn.functional.logsigmoid(target.reshape(-1)*output.reshape(-1)))
    ```

- **Step 4** : Train model with SGD.

    ```py
    train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)

    # Train the model (for 3 epochs)
    for epoch in range(3) :
        for images, labels in train_loader :
            # Clear previously computed gradient
            optimizer.zero_grad()

            # then compute gradient with forward and backward passes
            train_loss = loss_function(model(images), labels.float())
            train_loss.backward()

            # perform SGD step (parameter update)
            optimizer.step()
    ```

## [`2.5.py`](2.5.py)

CIFAR10 (two classes, label plane and car) with logistic regression trained with random permutation cyclic batch SGD.

## [`2.6.py`](2.6.py)

[`2.5.py`](2.5.py) with multilayer perceptron (MLP).

- **Step 2** : Define the neural network class.

    ```py
    class MLP4(nn.Module) :
        # CIFAR-10 data is 32*32 images with 3 RGB channels
        def __init__(self, input_dim=3*32*32) :
            super().__init__()
            self.linear = nn.Linear(input_dim, input_dim//2, bias=True)
            self.linear2 = nn.Linear(input_dim//2, input_dim//4, bias=True)
            self.linear3 = nn.Linear(input_dim//4, input_dim//8, bias=True)
            self.linear4 = nn.Linear(input_dim//8, 1, bias=True)

        def forward(self, x) :
            x = x.float().view(-1, 3*32*32)
            x = nn.functional.relu(self.linear(x))
            x = nn.functional.relu(self.linear2(x))
            x = nn.functional.relu(self.linear3(x))
            x = self.linear4(x)
            return x
    ```

# Multi-Class Classification with Softmax Regression

$$
\mu\left(f_{A, b}(x)\right)=\frac{1}{\sum_{i=1}^{k} e^{a_{i}^{\top} x+b_{i}}}\left[\begin{array}{c}
e^{a_{1}^{\top} x+b_{1}} \\
e^{a_{2}^{\top} x+b_{2}} \\
\vdots \\
e^{a_{k}^{\top} x+b_{k}}
\end{array}\right]
$$

$$
\begin{gathered}
\underset{A \in \mathbb{R}^{k \times n}, b \in \mathbb{R}^{k}}{\mathrm{minimize}} \sum_{i=1}^{N} D_{\mathrm{KL}}\left(\mathcal{P}\left(Y_{i}\right) \| \mu\left(f_{A, b}\left(X_{i}\right)\right)\right) \\
\mathbb{\Updownarrow} \\
\underset{A \in \mathbb{R}^{k \times n}, b \in \mathbb{R}^{k}}{\mathrm{minimize}} \sum_{i=1}^{N} H\left(\mathcal{P}\left(Y_{i}\right), \mu\left(f_{A, b}\left(X_{i}\right)\right)\right) \\
\mathbb{\Updownarrow} \\
\underset{A \in \mathbb{R}^{k \times n}, b \in \mathbb{R}^{k}}{\mathrm{minimize}} \frac{1}{N} \sum_{i=1}^{N}-\log \left(\mu_{Y_{i}}\left(f_{A, b}\left(X_{i}\right)\right)\right) \\
\mathbb{\Updownarrow} \\
\underset{A \in \mathbb{R}^{k \times n}, b \in \mathbb{R}^{k}}{\mathrm{minimize}} \frac{1}{N} \sum_{i=1}^{N}-\log \left(\frac{\exp \left(a_{Y_{i}}^{\top} X_{i}+b_{Y_{i}}\right)}{\sum_{j=1}^{k} \exp \left(a_{j}^{\top} X_{i}+b_{j}\right)}\right) \\
\mathbb{\Updownarrow} \\
\underset{A \in \mathbb{R}^{k \times n}, b \in \mathbb{R}^{k}}{\mathrm{minimize}} \frac{1}{N} \sum_{i=1}^{N} \ell^{\mathrm{CE}}\left(f_{A, b}\left(X_{i}\right), Y_{i}\right)
\end{gathered}
$$

## [`2.7.py`](2.7.py)

Basic softmax regression with pytorch.

- **Step 2** : Define the neural network class.

    ```py
    class softmax(nn.Module) :
        def __init__(self, input_dim=28*28) :
            super().__init__()
            self.linear = nn.Linear(input_dim, 10, bias=True)

        def forward(self, x) :
            return self.linear(x.float().view(-1, 28*28))
    ```

- **Step 3** : Create the model, specify loss function and optimizer.

    ```py
    model = softmax()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    ```
# GPU computing on PyTorch

## [`2.8.py`](2.8.py)

Simple demonstration of GPU computing on pytorch.

```py
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

N = 8192
A = torch.normal(0, 1/np.sqrt(N), (N, N))   # 8*8192^2=512Mb data
x = torch.normal(0.0, 1.0, (N, 1))

A = A.to(device)
x = x.to(device)
for _ in range(100): x = A@x

x = x.to("cpu")
```

## [`2.9.py`](2.9.py)

[`2.6.py`](2.6.py) modified to classify MNIST, using GPU.

```py
model = MLP4().to(device)

for images, labels in train_loader :
    images, labels = images.to(device), labels.to(device)
```