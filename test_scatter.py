from time import perf_counter

import numpy as np
import torch

repeats = 100
p = [
    0.9,
    0.09,
    0.009,
    0.0009,
    0.0001,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
n = len(p)
labels = torch.tensor(np.random.choice(n, size=200000, p=p)).to("cuda")

assert labels.dtype == torch.long

torch.cuda.synchronize()
t0 = perf_counter()
for _ in range(repeats):
    scatter_add_long_results = torch.zeros(
        n, dtype=torch.float32, device=labels.device
    ).scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
torch.cuda.synchronize()
t1 = perf_counter()
print("scatter_add_float32: ", (t1 - t0) / repeats)

torch.cuda.synchronize()
t0 = perf_counter()
for _ in range(repeats):
    scatter_add_results = torch.zeros(
        n, dtype=torch.float64, device=labels.device
    ).scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float64))
torch.cuda.synchronize()
t1 = perf_counter()
print("scatter_add_float64: ", (t1 - t0) / repeats)

torch.cuda.synchronize()
t0 = perf_counter()
for _ in range(repeats):
    scatter_add_long_results = torch.zeros(
        n, dtype=torch.float16, device=labels.device
    ).scatter_add_(0, labels, torch.ones_like(labels, dtype=torch.float16))
torch.cuda.synchronize()
t1 = perf_counter()
print("scatter_add_float16: ", (t1 - t0) / repeats)
