import re

from pathlib import Path

# You can add multimodal datasets here and register a short nickname to ${data_dict}.
# The data format should follow the general multimodal VLM format, for example:
# https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-finetune/README.md

json_root = f"./playground/Datasets/LLaVA-OneVision-COCO/llava_jsons"
image_root = f"./playground/Datasets/LLaVA-OneVision-COCO/images"

SHAREGPT4V_COCO = {
    "annotation_path": f"{json_root}/sharegpt4v_coco.json",
    "data_path": f"{image_root}/",
}

GQA_LOCAL = {
    "annotation_path": "/diff/wallx_workspace/wallx_data_ckp/datasets/gqa/gqa_cleaned.json",
    "data_path": "/diff/wallx_workspace/wallx_data_ckp/datasets/gqa/images",
}

VISUAL_GENOME_LOCAL = {
    "annotation_path": "/diff/wallx_workspace/wallx_data_ckp/datasets/visual_genome/visual_genome_clean.json",
    "data_path": "/diff/wallx_workspace/wallx_data_ckp/datasets/visual_genome/images",
}

VSI_SMALL = {
    "annotation_path": "/diff/wallx_workspace/wallx_data_ckp/datasets/VSI-small/vsi.json",
    "data_path": "/diff/wallx_workspace/wallx_data_ckp/datasets/VSI-small",
}

data_dict = {
    "sharegpt4v_coco": SHAREGPT4V_COCO,
    "gqa_local": GQA_LOCAL,
    "visual_genome_local": VISUAL_GENOME_LOCAL,
    "vsi_small": VSI_SMALL,
}

def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0

def data_list(dataset_names):
    if dataset_names == ["all"]:
        dataset_names = list(data_dict.keys())
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list

if __name__ == "__main__":
    print(data_list)
