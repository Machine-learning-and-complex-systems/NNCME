"""Utilities to reconstruct and load saved neural-network checkpoints."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from nncme.networks.gru import GRU
from nncme.networks.nade import NADE
from nncme.networks.transformer import TraDE


def build_net_from_args(meta_args: Dict) -> torch.nn.Module:
    """Instantiate the appropriate network class from metadata."""

    net_type = meta_args.get("net") or meta_args.get("net_type")
    if net_type == "rnn":
        return GRU(**meta_args)
    if net_type == "transformer":
        return TraDE(**meta_args)
    if net_type == "NADE":
        return NADE(**meta_args)
    raise ValueError(f"Unknown net type in metadata: {net_type}")


def load_checkpoint(nets_path: str, tstep: Optional[int] = None, device: str = "cpu") -> Tuple[torch.nn.Module, Dict, Tuple[int, ...]]:
    """Load a checkpointed network for a specific time step."""

    package = torch.load(nets_path, map_location=torch.device(device))
    meta = package.get("_meta", {})
    args_dict = meta.get("args", {})

    tkeys = []
    for key in package.keys():
        if key == "_meta":
            continue
        try:
            tkeys.append(int(key))
        except Exception:
            if isinstance(key, int):
                tkeys.append(key)
    if not tkeys:
        raise RuntimeError("No Tstep checkpoints found in nets_dict.")
    available = tuple(sorted(tkeys))

    use_t = tstep if tstep is not None else available[-1]
    if use_t not in available:
        raise KeyError(f"Requested Tstep {use_t} not in available {available}")

    cfg = dict(args_dict)
    cfg["net"] = meta.get("net_type")
    net = build_net_from_args(cfg)

    record = package.get(str(use_t), package.get(use_t))
    if record is None:
        raise KeyError(f"Checkpoint for Tstep {use_t} not found.")
    state = record["net"]
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net, meta, available


# nets_dict.pt path
nets_path = r"...\out\Schlogl\Schlogl_L2_S2_M85_T201_dt1e-05_batch1000\nd1_nw16_NADE_TDVP_lr1_epoch1_Losskl_Samplingalpha_IniDistdelta_Para1_bias_cg1\out_img\nets_dict.pt"

device = 'cuda' # 'cpu' or 'cuda'

_, meta, available_tsteps = load_checkpoint(nets_path, device=device)
# 创建字典保存所有模型
all_nets = {}
for tstep in available_tsteps:
    net, _, _ = load_checkpoint(nets_path, tstep=tstep, device=device)
    all_nets[tstep] = net
    print(f"loaded t= {tstep} models")
# usage example:
tstep_to_use = 100  # 选择特定时间步
if tstep_to_use in all_nets:
    net = all_nets[tstep_to_use]
    samples = net.sample(100)[0] #torch.Size([100, 2])
