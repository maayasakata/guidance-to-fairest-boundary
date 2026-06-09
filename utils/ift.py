import torch
import torch.nn as nn
import math

from utils.utils import cal_thresh_logit

def phi_fn(z0: torch.Tensor, z1: torch.Tensor, tau0: torch.Tensor, tau1: torch.Tensor, k: float = 1.0,) -> torch.Tensor:

    term1 = torch.sigmoid(k*(z1 - tau1)).mean()
    term0 = torch.sigmoid(k*(z0 - tau0)).mean()
    return term1 - term0 


class ImplicitT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, z1, k, eps):

        logit0 = z0.detach().cpu().numpy()
        logit1 = z1.detach().cpu().numpy()
        _, _, t, _ = cal_thresh_logit(logit0, logit1, 0)
        t = torch.tensor(t, device=z0.device)
        len0, len1 = len(logit0), len(logit1)
        p1 = len1 / (len0 + len1)
        p0 = 1 - p1
        p0t = torch.tensor(p0, device=z0.device, dtype=z0.dtype)
        p1t = torch.tensor(p1, device=z0.device, dtype=z0.dtype)
        bound = torch.minimum(p0t, p1t) - 1e-4
        t = torch.clamp(t, -bound, bound)

        ctx.save_for_backward(z0, z1) 
        ctx.t = t if isinstance(t, torch.Tensor) else torch.tensor(t, device=z0.device)
        ctx.p0 = p0
        ctx.p1 = p1
        ctx.k = float(k)
        ctx.eps = float(eps)
        return t

    @staticmethod
    def backward(ctx, grad_out):
    
        z0, z1 = ctx.saved_tensors 
        t = ctx.t                 
        p0 = ctx.p0
        p1 = ctx.p1
        k = ctx.k

        with torch.enable_grad():
            t_req  = t.detach().requires_grad_(True)
            z0_req = z0.detach().requires_grad_(True)
            z1_req = z1.detach().requires_grad_(True)

            p0t = torch.as_tensor(p0, device=t_req.device, dtype=t_req.dtype)
            p1t = torch.as_tensor(p1, device=t_req.device, dtype=t_req.dtype)
            bound = torch.minimum(p0t, p1t) - 1e-4  
            t_req = torch.clamp(t_req, -bound, bound)
            eps_log = 1e-8
            r0 = (p0t + t_req) / (p0t - t_req)
            r1 = (p1t - t_req) / (p1t + t_req)
            r0 = torch.clamp(r0, min=eps_log, max=1/eps_log)
            r1 = torch.clamp(r1, min=eps_log, max=1/eps_log)
            tau0 = -torch.log(r0)
            tau1 = -torch.log(r1)

            phi = phi_fn(z0_req, z1_req, tau0, tau1, k=k)

            dphi_dz0, dphi_dz1, dphi_dt = torch.autograd.grad(phi, (z0_req, z1_req, t_req),
                retain_graph=False,
                create_graph=False,   
                allow_unused=False
            )

        sign = torch.sign(dphi_dt)
        abs_dphi_dt = dphi_dt.abs().clamp(min=1e-4)
        inv_dphi_dt = 1.0 / (sign * abs_dphi_dt)
        dt_dz0 = - dphi_dz0 * inv_dphi_dt
        dt_dz1 = - dphi_dz1 * inv_dphi_dt

        grad_z0 = grad_out * dt_dz0
        grad_z1 = grad_out * dt_dz1

        return grad_z0, grad_z1, None, None


class ImplicitTModule(nn.Module):
    def __init__(self, k=1.0, eps=1e-6):
        super().__init__()
        self.k = float(k)
        self.eps = float(eps)

    def forward(self, z0, z1):
        return ImplicitT.apply(z0, z1, self.k, self.eps)

