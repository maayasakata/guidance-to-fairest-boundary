import torch.nn.functional as F
import torch

def interval(x, tau, k=10.0):
    return torch.where(tau >= 0, torch.sigmoid(k * x), torch.sigmoid(-k * x))

def gen_loss(yhat, t, a, pa, k=10):
    device = yhat.device
    dtype  = yhat.dtype
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=device, dtype=dtype)
    else:
        t = t.to(device=device, dtype=dtype)
    if not torch.is_tensor(pa):
        pa = torch.tensor(pa, device=device, dtype=dtype)
    else:
        pa = pa.to(device=device, dtype=dtype)

    if isinstance(a, str):
        a = 0 if a == "0" else 1
    a = int(a)

    s = 2*a - 1
    tau = -torch.log((pa - s*t)/(pa + s*t))
    coef = interval(yhat, tau, k=k)

    dist = torch.where(tau > 0, F.relu(tau - yhat), F.relu(yhat - tau))
    return (coef * dist).mean() 
