import numpy as np
import matplotlib.pyplot as plt

N, p = 300, 30
np.random.seed(0)
U,_ = np.linalg.qr(np.random.randn(N,N))
V,_ = np.linalg.qr(np.random.randn(p,p))
sig = np.random.rand(p)+0.2
sig[1] = 5
X = U[:,:p]@np.diag(sig)@V
Y = np.random.randn(N)

K = 30  #number of epochs


# Stochastic Gradient Descent
theta = np.zeros(p)
alpha = 0.1
f_val_SGD = []
for _ in range(K*N):
    ind = np.random.randint(N)
    theta -= alpha*(X[ind,:]@theta-Y[ind])*X[ind,:]
    f_val_SGD.append(np.linalg.norm(X@theta-Y)**2)


# Cyclic Stochastic Gradient Descent
theta = np.zeros(p)
alpha = 0.1
f_val_cyclic = []
for j in range(K*N):
    ind = j % N
    theta -= alpha*(X[ind,:]@theta-Y[ind])*X[ind,:]
    f_val_cyclic.append(np.linalg.norm(X@theta-Y)**2)


# Shuffled Cyclic Stochastic Gradient Descent
theta = np.zeros(p)
alpha = 0.1
f_val_shuffle_cyclic = []
for j in range(K*N):
    if j%N == 0:  # reshuffle every epoch
        perm = np.random.permutation(np.arange(N))
    ind = perm[j%N]
    theta -= alpha*(X[ind,:]@theta-Y[ind])*X[ind,:]
    f_val_shuffle_cyclic.append(np.linalg.norm(X@theta-Y)**2)


plt.rc('text', usetex=True)
plt.plot(np.arange(K*N)/N, f_val_SGD, color = "green", label = "SGD")
plt.plot(np.arange(K*N)/N, f_val_cyclic, color = "black", label = "Cyclic SGD")
plt.plot(np.arange(K*N)/N, f_val_shuffle_cyclic, color = "blue", label = "Shuffled Cylic SGD")
plt.plot(list(range(K)), np.linalg.norm(X@np.linalg.inv(X.T@X)@X.T@Y-Y)**2*np.ones(K), color = "red", label = "Optimal Value")
plt.xlabel('Epochs')
plt.ylabel(r'$f(\theta^k)$')
plt.legend()
plt.savefig("1.3.1.png")
plt.show()
