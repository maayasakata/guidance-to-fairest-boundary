import torch
from typing import List
from torch.func import grad, jvp

def tree_dot(a, b):
    return sum((a[k] * b[k]).sum() for k in a.keys())


def hvp(loss: torch.Tensor, params: List[torch.Tensor], vec: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    HVP: (∇^2_{param} loss) @ vec
    """
    g = grad(loss)
    _, Hv = jvp(g, (params,), (vec,))
    return Hv

def cross_hvp(loss: torch.Tensor, params_left: List[torch.Tensor], params_right: List[torch.Tensor], vec: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Cross-HVP: ∇_{left}( ∇_{right} loss · vec )
    """
    grad_r = grad(loss, argnums=1)  # ∇_{right} loss
    def scalar(left, right):
        gh = grad_r(left, right)
        return tree_dot(gh, vec)
    
    grad_g = grad(scalar, argnums=0)  # ∇_{left}({right} loss)
    return grad_g(params_left, params_right)


def list_dot(a, b):
    s = 0.0
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        s = s + (x * y).sum()
    return s

def hvp_autograd(g_right, params_right, vec_right, retain_graph=True):
    gv = list_dot(g_right, vec_right)
    Hv = torch.autograd.grad(gv, params_right, retain_graph=retain_graph, allow_unused=True)
    return Hv

def cross_hvp_autograd(g_right, params_left, vec_right, retain_graph=False):
    gv = list_dot(g_right, vec_right)
    g_left = torch.autograd.grad(gv, params_left, retain_graph=retain_graph, allow_unused=True)
    return g_left
