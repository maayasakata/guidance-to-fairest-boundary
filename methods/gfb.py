import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from hydra.utils import instantiate
import random
import copy

from methods.base import MethodBase
from utils.utils import generate_tradeoff, make_fig, cal_avg_curve
from utils.ift import ImplicitTModule
from utils.autograd_hvp import hvp_autograd, cross_hvp_autograd
from metrics import Hypervolume, DDP


class GFB(MethodBase):
    def __init__(self, dataset, savedir, **kwargs):
        super().__init__(dataset, savedir, **kwargs)
        models = self.models
        self.models = {"0": models["0"].cls, "1": models["1"].cls}
        self.generators = {"0": models["0"].gen, "1": models["1"].gen}
        wd = 1e-4
        optimizers = self.optimizers
        self.optimizers = {
            "0": optimizers["f"]["optimizer"](self.models["0"].parameters(), lr=float(optimizers["f"]["args"]["lr0"]), weight_decay=wd),
            "1": optimizers["f"]["optimizer"](self.models["1"].parameters(), lr=float(optimizers["f"]["args"]["lr1"]), weight_decay=wd),
        }
        self.optim_gens = {
            "0": optimizers["g"]["optimizer"](self.generators["0"].parameters(), lr=float(optimizers["g"]["args"]["lr0"])),
            "1": optimizers["g"]["optimizer"](self.generators["1"].parameters(), lr=float(optimizers["g"]["args"]["lr1"])),
        }

        self.criterion_gen = instantiate(kwargs["methods"]["method"]["args"]["criterion_gen"])
        # Trade-off weight λ between the distribution-matching loss and the prediction loss.
        self.C = float(kwargs["methods"]["method"]["args"]["C"])

        # Weight for the moving average.
        self.theta = float(kwargs["methods"]["method"]["args"]["theta"])

        # Step size for the auxiliary variable.
        self.gamma = float(kwargs["methods"]["method"]["args"]["gamma"])

        self.ift = ImplicitTModule(k=10, eps=1e-6)

        self.gen0 = [p for p in self.generators["0"].parameters() if p.requires_grad]
        self.gen1 = [p for p in self.generators["1"].parameters() if p.requires_grad]
        self.head0 = [p for p in self.models["0"].parameters() if p.requires_grad]
        self.head1 = [p for p in self.models["1"].parameters() if p.requires_grad]
        self.leng = len(self.gen0)
        self.lenh = len(self.head0)
        head0 = [p for p in self.models["0"].parameters() if p.requires_grad]
        head1 = [p for p in self.models["1"].parameters() if p.requires_grad]
        gen0  = [p for p in self.generators["0"].parameters() if p.requires_grad]
        gen1  = [p for p in self.generators["1"].parameters() if p.requires_grad]

        self.head_params = head0 + head1
        self.gen_params  = gen0 + gen1 

        save_path = os.path.join(self.savedir, self.folder, self.dataset)
        os.makedirs(save_path, exist_ok=True)

    def _forward(self, x, a):
        x = self.process_x(x)
        x_gen = self.generators[a](x)
        yhat = self.models[a](x_gen)
        return yhat
    
    def cal_pa(self, z0, z1):
        len0 = len(z0)
        len1 = len(z1)
        p0 = len0 / (len0 + len1)
        p1 = 1 - p0
        return p0, p1
    
    def compute_quantities(self, batch0, batch1):
        x0, y0, _ = batch0
        x1, y1, _ = batch1
        yhat0 = self._forward(x0, "0")
        yhat1 = self._forward(x1, "1")
        p0, p1 = self.cal_pa(yhat0, yhat1)

        L_pred0 = self.criterion(yhat0, y0.to(self.device, non_blocking=True).view(-1, 1).float())
        L_pred1 = self.criterion(yhat1, y1.to(self.device, non_blocking=True).view(-1, 1).float())

        # implicit gradient
        t = self.ift(yhat0, yhat1)

        L_dist0 = self.criterion_gen(yhat0, t, "0", p0)
        L_dist1 = self.criterion_gen(yhat1, t, "1", p1)
        return y0, yhat0, y1, yhat1, L_pred0, L_pred1, L_dist0, L_dist1, t

    def makeD(self, L_pred, L_gen, z):
        # Compute the inner, auxiliary, and outer update directions for the bi-level optimization.
        head_params = self.head_params
        gen_params = self.gen_params
        
        nabla_2g = torch.autograd.grad(L_pred, head_params, create_graph=True, retain_graph=True)
        nabla_f_all = torch.autograd.grad(L_gen, head_params + gen_params, retain_graph=True, create_graph=False)
        nabla_2f, nabla_1f = nabla_f_all[:len(head_params)], nabla_f_all[len(head_params):] 

        nabla_22g = hvp_autograd(nabla_2g, head_params, z, retain_graph=True)
        nabla_12g = cross_hvp_autograd(nabla_2g, gen_params, z, retain_graph=False)

        def safe_sub(a, b):
            if a is None and b is None:
                return None
            if a is None:
                return -b
            if b is None:
                return a
            return a - b
        
        Dy = list(nabla_2g)
        Dz = [safe_sub(a, b) for a, b in zip(nabla_22g, nabla_2f)]
        Dx = [safe_sub(a, b) for a, b in zip(nabla_1f, nabla_12g)]

        return Dy, Dz, Dx
    
    @torch.no_grad()
    def update_w(self, w, Dw):
        # update auxiliary
        for wi, dwi in zip(w, Dw):
            wi.add_(-self.gamma * dwi)

    @torch.no_grad()
    def sgd_step(self, params, direction, optimizer):
        optimizer.zero_grad()
        for p, d in zip(params, direction):
            p.grad = d
        optimizer.step()

    @torch.no_grad()
    def makeh(self, h_bef, Dx, theta):
        # Compute moving average.
        h = tuple((1 - theta) * hb + theta * dx for hb, dx in zip(h_bef, Dx))
        return h

    @torch.no_grad()
    def validation(self, val_loader):
        self.generators["0"].eval()
        self.generators["1"].eval()
        self.models["0"].eval()
        self.models["1"].eval()

        y_true, y_pred = [], []
        for i, (x, y, z) in enumerate(val_loader):
            a = str(int(z[0].item()))
            yhat = self._forward(x, a)

            y_true.append(y.detach().cpu().view(-1))
            y_pred.append(yhat.detach().cpu().view(-1))

        y_true = torch.cat(y_true).numpy() if len(y_true) > 0 else np.array([])
        y_pred = torch.cat(y_pred).numpy() if len(y_pred) > 0 else np.array([])
        
        return y_true, y_pred
    
    @torch.no_grad()
    def test(self, train_loader, test_loader):
        self.generators["0"].eval()
        self.generators["1"].eval()
        self.models["0"].eval()
        self.models["1"].eval()

        eta, eta_test, y_true = [], [], []
        for i, (x, _, z) in enumerate(train_loader):
            yhat = self._forward(x, str(int(z[0].item())))
            eta.append(yhat.detach().cpu().view(-1))

            if (i+1) % 10 == 0 or (i+1) == len(train_loader):
                print(f"[{i+1}/{len(train_loader)}]", end="\r")
        
        for i, (x, y, z) in enumerate(test_loader):
            yhat = self._forward(x, str(int(z[0].item())))
            eta_test.append(yhat.detach().cpu().view(-1))
            y_true.append(y.detach().cpu().view(-1))

            if (i+1) % 10 == 0 or (i+1) == len(test_loader):
                print(f"[{i+1}/{len(test_loader)}]", end="\r")

        eta_ = torch.cat(eta).numpy() if len(eta) > 0 else np.array([])
        eta_test_ = torch.cat(eta_test).numpy() if len(eta_test) > 0 else np.array([])
        y_true_ = torch.cat(y_true).numpy() if len(y_true) > 0 else np.array([])

        return eta_, eta_test_, y_true_
    
    def infinite_loader(self, loader):
        while True:
            for batch in loader:
                yield batch
        
    def train_step(self, **kwargs):
        train_loader0, train_loader1 = kwargs["train_loader0"], kwargs["train_loader1"]
        val_loader0, val_loader1 = kwargs["val_loader0"], kwargs["val_loader1"]

        if len(train_loader0) <= len(train_loader1):
            short_loader, long_loader = train_loader0, train_loader1
            short_is_0 = True
        else:
            short_loader, long_loader = train_loader1, train_loader0
            short_is_0 = False

        long_iter = self.infinite_loader(long_loader)
        counter = 0
        patience = 10 
        stop_train = False

        w = [torch.zeros_like(p) for p in self.head_params]
        h = [torch.zeros_like(p) for p in self.gen_params]

        for i in range(self.epochs):
            #---lists for record---#
            logit0_list, logit1_list = [], []
            #----------------------#

            for j, batch_short in enumerate(short_loader):
                batch_long = next(long_iter)

                if short_is_0:
                    batch0, batch1 = batch_short, batch_long
                else:
                    batch0, batch1 = batch_long, batch_short

                _, yhat0, _, yhat1, L_pred0, L_pred1, L_dist0, L_dist1, t_batch = self.compute_quantities(batch0, batch1)

                logit0_list.extend(yhat0.detach().cpu())
                logit1_list.extend(yhat1.detach().cpu())

                L_pred = (L_pred0 + L_pred1) * 0.5
                L_gen0 = (1-self.C) * L_dist0 + self.C * L_pred0
                L_gen1 = (1-self.C) * L_dist1 + self.C * L_pred1
                L_gen = (L_gen0 + L_gen1) * 0.5

                Dh, Dw, Dg = self.makeD(L_pred, L_gen, w)
                h0, h1 = h[:self.leng], h[self.leng:]
                Dh0, Dh1 = Dh[:self.lenh], Dh[self.lenh:]

                #--- update parameters ---#
                self.sgd_step(self.gen0, h0, self.optim_gens["0"])
                self.sgd_step(self.gen1, h1, self.optim_gens["1"])
                self.sgd_step(self.head0, Dh0, self.optimizers["0"])
                self.sgd_step(self.head1, Dh1, self.optimizers["1"])
                self.update_w(w, Dw)
                h = self.makeh(h, Dg, self.theta)

                if (j+1) % 10 == 0 or (j+1) == len(short_loader):
                    print(f"Epoch: [{i+1}/{self.epochs}] train: [{j+1}/{len(short_loader)}]", end="\r")

            logit0, logit1 = np.array(logit0_list), np.array(logit1_list)   

            #---validation---#
            if i%1 == 0 or (i+1) == self.epochs:
                with torch.no_grad():            
                    Y0_val, logit0_val = self.validation(val_loader0)
                    Y1_val, logit1_val = self.validation(val_loader1)

                ddp_val_ = DDP(logit0_val, logit1_val, 0, 0)
                delta_list_val = np.linspace(0, abs(ddp_val_), 50)
                tradeoff_curve_val = generate_tradeoff(delta_list_val, logit0, logit1, logit0_val, logit1_val, Y0_val, Y1_val)
                hv_val = Hypervolume(tradeoff_curve_val)
                
                if i == 0:
                    print("first hv:", hv_val)
                    max_hv = hv_val
                    param0, param1 = self.log_param()

                elif hv_val > max_hv:
                    print("Update max hv:", hv_val)
                    max_hv = hv_val
                    param0, param1 = self.log_param()
                    counter = 0
                
                else: 
                    counter += 1
                    print(f"counter: {counter}", end="\r")
                    if counter >= patience:
                        print("Early stopping triggered.")
                        stop_train = True
                        break

            if stop_train:
                break   

        return param0, param1

    def log_param(self):
        bestnet0 = copy.deepcopy(self.models["0"].state_dict())
        bestnet1 = copy.deepcopy(self.models["1"].state_dict())
        bestgen0 = copy.deepcopy(self.generators["0"].state_dict())
        bestgen1 = copy.deepcopy(self.generators["1"].state_dict())
        param0 = {"f": bestnet0, "g": bestgen0}
        param1 = {"f": bestnet1, "g": bestgen1}
        return param0, param1

    def save_param(self, param0, param1, seed):
        save_dir = f'{self.savedir}/saved_models/{self.dataset}/gfb'
        os.makedirs(save_dir, exist_ok=True)
        torch.save(param0["f"], f'{save_dir}/bestmodel0_seed{seed}.pth')
        torch.save(param0["g"], f'{save_dir}/bestgen0_seed{seed}.pth')
        torch.save(param1["f"], f'{save_dir}/bestmodel1_seed{seed}.pth')
        torch.save(param1["g"], f'{save_dir}/bestgen1_seed{seed}.pth')
    
    def infer_step(self, seed, **kwargs):
        train_loader0, train_loader1 = kwargs["thresh_loader0"], kwargs["thresh_loader1"]
        test_loader0, test_loader1 = kwargs["test_loader0"], kwargs["test_loader1"]
    
        self.generators["0"].load_state_dict(torch.load(f"{self.savedir}/saved_models/{self.dataset}/gfb/bestgen0_seed{seed}.pth"))
        self.generators["1"].load_state_dict(torch.load(f"{self.savedir}/saved_models/{self.dataset}/gfb/bestgen1_seed{seed}.pth"))
        self.models["0"].load_state_dict(torch.load(f"{self.savedir}/saved_models/{self.dataset}/gfb/bestmodel0_seed{seed}.pth"))
        self.models["1"].load_state_dict(torch.load(f"{self.savedir}/saved_models/{self.dataset}/gfb/bestmodel1_seed{seed}.pth"))
        
        self.generators["0"].eval()
        self.generators["1"].eval()
        self.models["0"].eval()
        self.models["1"].eval()

        logit0, logit0_test, y0_test = self.test(train_loader0, test_loader0)
        logit1, logit1_test, y1_test = self.test(train_loader1, test_loader1)
        
        delta_max = abs(DDP(logit0, logit1, tau0=0, tau1=0))
        delta_list = np.linspace(0, delta_max, 10)
       
        seed_results = []
        current_res = generate_tradeoff(delta_list, logit0, logit1, logit0_test, logit1_test, y0_test, y1_test)
        seed_results.append(current_res)
    
        seed_df = pd.DataFrame(current_res)
        print(seed_df)
        hv = Hypervolume(seed_df)
        print(f"hypervlome on seed{seed}:", hv)
        seed_results.append(seed_df)
        seed_df.to_csv(f"{self.savedir}/{self.folder}/{self.dataset}/result_seed{seed}.csv", index=False)

        fig = make_fig(seed_df)
        fig.savefig(f"{self.savedir}/{self.folder}/{self.dataset}/results_plot_seed{seed}.png", dpi=300)


    def all_curve(self, **kwargs):
        from registry import dataset_registry
        logit0s, logit1s = [], []
        logit0s_test, logit1s_test = [], []
        y0s, y1s = [], []
        delta_max_list = []
        for seed in range(kwargs["seeds"]):
            seed = seed * 5
            print(f"Currently wroking on seed{seed}")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            dataset_class = dataset_registry[self.dataset]
            dataset = dataset_class()
            _, _, test_loader0, thresh_loader0, _, _, test_loader1, thresh_loader1 = dataset.load(random_state=seed)
        
            self.generators["0"].load_state_dict(torch.load(f"{self.savedir}/saved_models/{self.dataset}/gfb/bestgen0_seed{seed}.pth"))
            self.generators["1"].load_state_dict(torch.load(f"{self.savedir}/saved_models/{self.dataset}/gfb/bestgen1_seed{seed}.pth"))
            self.models["0"].load_state_dict(torch.load(f"{self.savedir}/saved_models/{self.dataset}/gfb/bestmodel0_seed{seed}.pth"))
            self.models["1"].load_state_dict(torch.load(f"{self.savedir}/saved_models/{self.dataset}/gfb/bestmodel1_seed{seed}.pth"))

            logit0, logit0_test, y0_test = self.test(thresh_loader0, test_loader0)
            logit1, logit1_test, y1_test = self.test(thresh_loader1, test_loader1)
            logit0s.append(logit0)
            logit1s.append(logit1)
            logit0s_test.append(logit0_test)
            logit1s_test.append(logit1_test)
            y0s.append(y0_test)
            y1s.append(y1_test)

            delta_max = abs(DDP(logit0, logit1, tau0=0, tau1=0))
            delta_max_list.append(delta_max)
            
        delta_list = np.linspace(0, np.mean(delta_max_list), 10)
        result_all, hv_mean, hv_var, per_seed_dfs = cal_avg_curve(delta_list, kwargs["seeds"], logit0s, logit1s, logit0s_test, logit1s_test, y0s, y1s)
        result_all.to_csv(f"{self.savedir}/results_avg/{self.dataset}/result_avg_gfb.csv", index=False)

        out_dir_each = f"{self.savedir}/results_avg/{self.dataset}/each"
        os.makedirs(out_dir_each, exist_ok=True)
        for df_seed in per_seed_dfs:
            seed_val = int(df_seed["seed"].iloc[0])
            df_seed.to_csv(f"{out_dir_each}/result_gfb_seed{seed_val}.csv", index=False)

        print(f"hypervlome on GFB:", hv_mean)
        print(f"hypervolume variance on GFB:", hv_var)
    
        return result_all, hv_mean