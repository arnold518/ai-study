import torch
import numpy as np
import time

print(f"torch.cuda.is_available() : {torch.cuda.is_available()}")
print(f"torch.cuda.device_count() : {torch.cuda.device_count()}")

print(f"torch.__version__ : {torch.__version__}")

t = torch.tensor([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3]
], dtype=torch.float32)

# device = "cpu"
device = "cuda:0"
# device = "cuda:5" # error if you have fewer than 6 GPUs

t_dev = t.to(device)

print(f"t.device : {t.device}")
print(f"t_dev.device : {t_dev.device}")
print()

# =========

def f(device, A, x) :

    A = A.to(device)
    x = x.to(device)   # error if A is sent to GPU but x is not sent to GPU

    start = time.time()
    for _ in range(100):
        x = A@x   # matrix-vector product
    end = time.time()
    print(f"<{device}> Time ellapsed in loop is: {end - start}")

    x = x.to("cpu")
    print(f"result : {np.linalg.norm(x)}")

N = 8192
A = torch.normal(0, 1/np.sqrt(N), (N, N))   # 8*8192^2=512Mb data
x = torch.normal(0.0, 1.0, (N, 1))

f("cpu", A, x)
f(device, A, x)