import json
import os
from pathlib import Path

import numpy as np
import torch.distributed as dist
from accelerate.logging import get_logger
from torch.utils.data import DataLoader

from starVLA.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)


def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")


def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"):
    # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset

        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=4,
            # shuffle=True
        )
        if dist.get_rank() == 0:
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader

    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        return vlm_train_dataloader

    elif dataset_py == "vlm_sft_qwen3_datasets":
        from starVLA.dataloader.vlm_sft_qwen3_datasets import (
            make_vlm_dataloader as make_vlm_sft_qwen3_dataloader,
        )

        vlm_data_module = make_vlm_sft_qwen3_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        return vlm_train_dataloader

    elif dataset_py == "wallx_vla_dataset":
        from starVLA.dataloader.wallx_cotrain_datasets import collate_fn_vla, get_vla_dataset

        vla_dataset_cfg = cfg.datasets.vla_data
        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn_vla,
            num_workers=int(vla_dataset_cfg.get("num_workers", 4)),
        )
        return vla_train_dataloader

    elif dataset_py == "wallx_vlm_dataset":
        from starVLA.dataloader.wallx_cotrain_datasets import collate_fn_vlm, get_vlm_dataset

        vlm_dataset_cfg = cfg.datasets.vlm_data
        vlm_dataset = get_vlm_dataset(data_cfg=vlm_dataset_cfg)
        vlm_train_dataloader = DataLoader(
            vlm_dataset,
            batch_size=cfg.datasets.vlm_data.per_device_batch_size,
            collate_fn=collate_fn_vlm,
            num_workers=int(vlm_dataset_cfg.get("num_workers", 4)),
        )
        return vlm_train_dataloader
    elif dataset_py == "wallx_vlm_signal_dataset":
        from starVLA.dataloader.wallx_cotrain_datasets import collate_fn_vlm_signal, get_vlm_signal_dataset

        vlm_signal_dataset_cfg = cfg.datasets.vlm_signal_data
        vlm_signal_dataset = get_vlm_signal_dataset(data_cfg=vlm_signal_dataset_cfg)
        vlm_signal_train_dataloader = DataLoader(
            vlm_signal_dataset,
            batch_size=cfg.datasets.vlm_signal_data.per_device_batch_size,
            collate_fn=collate_fn_vlm_signal,
            num_workers=int(vlm_signal_dataset_cfg.get("num_workers", 4)),
        )
        return vlm_signal_train_dataloader

    elif dataset_py == "wallx_router_dataset":
        from starVLA.dataloader.wallx_cotrain_datasets import collate_fn_router, get_router_dataset

        router_dataset_cfg = cfg.datasets.router_data
        router_dataset = get_router_dataset(data_cfg=router_dataset_cfg)
        router_train_dataloader = DataLoader(
            router_dataset,
            batch_size=cfg.datasets.router_data.per_device_batch_size,
            collate_fn=collate_fn_router,
            num_workers=int(router_dataset_cfg.get("num_workers", 4)),
        )
        return router_train_dataloader

    elif dataset_py == "wallx_subtask_dataset":
        from starVLA.dataloader.wallx_cotrain_datasets import collate_fn_router, get_subtask_dataset

        subtask_dataset_cfg = cfg.datasets.subtask_data
        subtask_dataset = get_subtask_dataset(data_cfg=subtask_dataset_cfg)
        subtask_train_dataloader = DataLoader(
            subtask_dataset,
            batch_size=cfg.datasets.subtask_data.per_device_batch_size,
            collate_fn=collate_fn_router,
            num_workers=int(subtask_dataset_cfg.get("num_workers", 4)),
        )
        return subtask_train_dataloader
    elif dataset_py == "go2_waypoint_router_dataset":
        from starVLA.dataloader.go2_waypoint_dataset import (
            collate_fn_go2,
            get_go2_waypoint_dataset,
        )

        router_dataset_cfg = cfg.datasets.router_data
        router_dataset = get_go2_waypoint_dataset(data_cfg=router_dataset_cfg)
        return DataLoader(
            router_dataset,
            batch_size=router_dataset_cfg.per_device_batch_size,
            collate_fn=collate_fn_go2,
            num_workers=int(router_dataset_cfg.get("num_workers", 4)),
            shuffle=bool(router_dataset_cfg.get("shuffle", True)),
        )
