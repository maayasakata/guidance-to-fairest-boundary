import matplotlib.pyplot as plt
import pandas as pd
# from deap import creator, base
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import os
from hydra import utils 

from registry import  method_registry
from metrics.Hypervolume import Hypervolume


def _nondominated_2d_accmax_ddpmin(P: np.ndarray) -> np.ndarray:
    if P.size == 0:
        return P

    idx = np.lexsort((P[:, 1], -P[:, 0]))
    S = P[idx]

    best_ddp = np.inf
    keep = []
    for i in range(S.shape[0]):
        ddp = S[i, 1]
        if ddp < best_ddp:
            keep.append(i)
            best_ddp = ddp
    return S[np.array(keep, dtype=int)]

def reference_point(tradeoff_list, *, acc_col="acc", ddp_col="ddp", use_union_pf=False):
    union = []
    n_list = []
    
    for df in tradeoff_list:
        P = df[[acc_col, ddp_col]].to_numpy(dtype=float)
        if P.size == 0:
            continue

        P_pf = _nondominated_2d_accmax_ddpmin(P)
        n_list.append(P_pf.shape[0])

        P_used = P_pf if use_union_pf else P
        union.append(P_used)

    print("P_used:", P_used)
    if len(union) == 0:
        raise ValueError("tradeoff_list has no points.")
    if max(n_list) < 2:
        raise ValueError("Need at least 2 PF points to define r (n>=2).")

    U = np.vstack(union)
    n = max(n_list)

    acc_min = float(np.min(U[:, 0]))
    acc_max = float(np.max(U[:, 0]))
    ddp_min = float(np.min(U[:, 1]))
    ddp_max = float(np.max(U[:, 1]))

    acc_span = acc_max - acc_min
    ddp_span = ddp_max - ddp_min

    acc_ref = acc_min - acc_span / (n - 1)
    ddp_ref = ddp_max + ddp_span / (n - 1)

    tradeoff_list_reg = []
    for curve in tradeoff_list:
        Pn = curve.copy()
        Pn["acc"] = (Pn["acc"] - acc_min) / acc_span
        Pn["ddp"] = (Pn["ddp"] - ddp_min) / ddp_span
        tradeoff_list_reg.append(Pn)

    acc_reg = acc_ref - acc_min
    ddp_reg = ddp_ref - ddp_max + 1

    norm = dict(
        acc_min=acc_min, acc_span=acc_span,
        ddp_min=ddp_min, ddp_span=ddp_span,
        ddp_max=ddp_max,
        n=n
    )

    return [acc_ref, ddp_ref], [acc_reg, ddp_reg], tradeoff_list_reg, norm


@hydra.main(config_path="conf", config_name="base", version_base="1.1")
def main(cfg: DictConfig):
    dataset_name = "compas" # "adult", "celeba", "utkface"
    savedir = "/path/to/your/output"
    methods_list = ["gfb"]
    seeds = 5

    METHOD_STYLE = {
    "gfb":     dict(color="tab:blue", marker="s", label="GFB"),
    }

    MARKER_SIZE = 80
    results_list = []
    fig, ax = plt.subplots(figsize=(8, 8))

    for method_name in methods_list:
        style = METHOD_STYLE.get(method_name, dict(color="black", marker="o", label=method_name))
        print(f"--- processing method: {method_name} ---")
        OmegaConf.set_struct(cfg, False)
        method_cfg = OmegaConf.load(f"{savedir}/conf/methods/{method_name}.yaml")
        with open_dict(cfg):
            cfg.methods = method_cfg
        data_cfg = OmegaConf.load(os.path.join(utils.get_original_cwd(), "conf", "datasets", f"{dataset_name}.yaml"))
        with open_dict(cfg):
            cfg.datasets = data_cfg

        method_key = cfg.methods.method.type  
        model_name = cfg.datasets.model_for[method_key]  
        model_cfg = OmegaConf.load(f"{savedir}/conf/models/{model_name}.yaml")

        with open_dict(cfg):
            cfg.models = model_cfg

        context = {**cfg, "seeds": seeds,}

        method_class = method_registry[method_name]
        method = method_class(dataset_name, savedir, **context)
        result_all, hv = method.all_curve(**context)
        results_list.append(result_all)
        print(f"plot of {method_name}: {result_all}")
        
        ax.scatter(result_all["acc"], result_all["ddp"], s=MARKER_SIZE, color=style["color"], marker=style["marker"], 
                label=style["label"], alpha=0.9, edgecolors="none",)

    hv_list = []
    r, r_reg, tradeoff_list = reference_point(results_list, use_union_pf=True)
    ax.scatter(r[0], r[1], s=400, color="black", marker="*", label="reference point", alpha=0.9, edgecolors="none",)
    print("reference point: ", r)
    for i, method_name in enumerate(methods_list):
        Pn = tradeoff_list[i].copy()
        hv_fair = Hypervolume(Pn, ref_point=r_reg)
        hv_list.append({"method": f"{method_name}", "hv": hv_fair,})
    print(hv_list)

    all_hv_df = pd.DataFrame(hv_list)
    all_hv_df.to_csv(f"{savedir}/results_avg/{dataset_name}/result_hv.csv", index=False)

    plt.xlabel('Accuracy', fontsize=20)
    plt.ylabel('DDP', fontsize=20)
    plt.grid(True)
    ax.legend(loc="best", fontsize=18, markerscale=1.4, frameon=True,)
    plt.tight_layout()
    plt.savefig(f"{savedir}/results_avg/{dataset_name}/results_plot_all.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()