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
B = 10  #size of minibatch


# Stochastic Gradient Descent
theta = np.zeros(p)
alpha = 0.1
f_val_SGD = []
for _ in range(K*N):
    ind = np.random.randint(N)
    theta -= alpha*(X[ind,:]@theta-Y[ind])*X[ind,:]
    f_val_SGD.append(np.linalg.norm(X@theta-Y)**2)


# Minibatch Stochastic Gradient Descent with Replacement
theta = np.zeros(p)
alpha = 0.1
f_val_MB = []
for _ in range(K*N//B):
    grad = np.zeros(p)
    for _ in range(B):
        ind = np.random.randint(N)
        grad += (X[ind,:]@theta-Y[ind])*X[ind,:]
    theta -= alpha*grad
    f_val_MB.append(np.linalg.norm(X@theta-Y)**2)


# Minibatch Stochastic Gradient Descent without Replacement
theta = np.zeros(p)
alpha = 0.1
f_val_RP = []
for _ in range(K*N//B):
    grad = np.zeros(p)
    perm = np.random.permutation(np.arange(N))
    for b in range(B):
        ind = perm[b]
        grad += (X[ind,:]@theta-Y[ind])*X[ind,:]
    theta -= alpha*grad
    f_val_RP.append(np.linalg.norm(X@theta-Y)**2)


plt.rc('text', usetex=True)
plt.plot(np.arange(K*N)/N, f_val_SGD, color = "green", label = "SGD")
plt.plot(np.arange(K*N//B)*B/N, f_val_MB, color = "black", label = "Minibatch SGD w/ replacement")
plt.plot(np.arange(K*N//B)*B/N, f_val_RP, color = "blue", label = "Minibatch SGD w/o replacement")
plt.plot(np.arange(K*N)/N, np.linalg.norm(X@np.linalg.inv(X.T@X)@X.T@Y-Y)**2*np.ones(K*N), color = "red", label = "Optimal Value")
plt.xlabel('Epochs')
plt.ylabel(r'$f(\theta^k)$')
plt.legend()
plt.savefig("1.2.1.png")
plt.show()
