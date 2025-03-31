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

K = 30  # number of epochs


# Gradient Descent
theta = np.zeros(p)
alpha = 10
f_val_GD = []
for _ in range(K):
    theta -= (alpha/N)*X.T@(X@theta-Y)
    f_val_GD.append(np.linalg.norm(X@theta-Y)**2)


# Stochastic Gradient Descent
theta = np.zeros(p)
alpha = 0.1
f_val_SGD = []
for _ in range(K*N):
    ind = np.random.randint(N)
    theta -= alpha*(X[ind,:]@theta-Y[ind])*X[ind,:]
    f_val_SGD.append(np.linalg.norm(X@theta-Y)**2)


plt.rc('text', usetex=True)
plt.plot(np.arange(K*N)/N, f_val_SGD, color = "green", label = "Stochastic Gradient Descent")
plt.plot(np.arange(0,K*N,N)/N, f_val_GD, color = "blue", linestyle = "", marker = "o", label = "Gradient Descent")
plt.plot(np.arange(K*N)/N, np.linalg.norm(X@np.linalg.inv(X.T@X)@X.T@Y-Y)**2*np.ones(K*N), color = "red", label = "Optimal Value")
plt.xlabel('Epochs')
plt.ylabel(r'$f(\theta^k)$')
plt.legend()
plt.savefig("result/1.1.1.png")
plt.show()
