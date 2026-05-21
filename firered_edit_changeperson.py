import os
import random
import yaml
import torch
import multiprocessing as mp

from PIL import Image
from diffusers import DiffusionPipeline


# ============================================================
# Config
# ============================================================

YAML_PATH = "keywords-CN.yaml"

INPUT_DIR = "/DATA_71/b59900515/data/candidate_hair/tgt"
OUTPUT_DIR = "/DATA_71/b59900515/data/candidate_hair/ref_firered"

MODEL_PATH = "FireRedTeam/FireRed-Image-Edit-1.1"

STEPS = 20
TRUE_CFG_SCALE = 4.0
GUIDANCE_SCALE = 1.0

SEED = None  # None 表示每张图随机
KEEP_SIZE = True
OUT_EXT = ".jpg"

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")

OVERWRITE = False
MAX_SAMPLE = None  # None 表示处理全部图片；整数表示只处理排序后的前 N 张


# ============================================================
# Global speed settings
# ============================================================

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True


# ============================================================
# Utils
# ============================================================

def ensure_list_str(cfg, key):
    v = cfg.get(key, None)
    if not isinstance(v, list) or len(v) == 0 or not all(isinstance(x, str) and x.strip() for x in v):
        raise TypeError(f"[ERROR] YAML 字段 `{key}` 必须是非空的 list[str]，实际值: {v}")
    return v


def load_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required_keys = [
        "color",
        "clothing",
        "posture",
        "location",
        "environmental_elements",
        "lighting",
        "hair_type_1",
        "hair_type_2",
    ]

    for key in required_keys:
        ensure_list_str(cfg, key)

    return cfg


def build_prompt(idx, sample):
    colors_list = sample["color"]
    clothes_list = sample["clothing"]
    posture_list = sample["posture"]
    location_list = sample["location"]
    environmental_elements_list = sample["environmental_elements"]
    lighting_list = sample["lighting"]
    hairtype_list = sample["hair_type_1"] + sample["hair_type_2"]

    clothes_color = random.choice(colors_list)
    clothes = random.choice(clothes_list)
    posture = random.choice(posture_list)
    location = random.choice(location_list)
    element = random.choice(environmental_elements_list)
    lighting = random.choice(lighting_list)
    hairtype = random.choice(hairtype_list)

    prompt = (
        f"保持头发的颜色样式均不变，"
        f"换成一个新的人物肖像照，穿着{clothes_color}的{clothes}，保持头发的颜色样式均不变，"
        f"姿势是{posture}，场景更换成{location}，"
        f"背景中有{element}，4K，电影级构图。"
    )

    negative_prompt = "变化发色，躯体畸形"

    return prompt, negative_prompt


def collect_image_files(input_folder):
    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(SUPPORTED_EXTS)
    ]
    image_files.sort(key=lambda x: x.lower())
    return image_files


def apply_max_sample(image_files, max_sample):
    if max_sample is None:
        return image_files
    if max_sample < 0:
        raise ValueError("MAX_SAMPLE must be None or a non-negative integer.")
    return image_files[:max_sample]


def split_data_round_robin(image_files, world_size):
    shards = [[] for _ in range(world_size)]

    for idx, filename in enumerate(image_files):
        rank = idx % world_size
        shards[rank].append((idx, filename))

    return shards


def load_pipeline(model_path, device):
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    return pipe


def make_generator(device, seed_value):
    return torch.Generator(device=device).manual_seed(seed_value)


# ============================================================
# Worker
# ============================================================

def worker_process(rank, world_size, shard, cfg):
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    print(
        f"[Worker {rank}] Start on {device}, "
        f"num_images={len(shard)}",
        flush=True,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipe = load_pipeline(MODEL_PATH, device)

    total = len(shard)

    for local_idx, (global_idx, filename) in enumerate(shard, start=1):
        in_path = os.path.join(INPUT_DIR, filename)
        base = os.path.splitext(filename)[0]
        out_path = os.path.join(OUTPUT_DIR, f"{base}{OUT_EXT}")

        if os.path.exists(out_path) and not OVERWRITE:
            print(
                f"[Worker {rank}] [{local_idx}/{total}] Skip existing: {out_path}",
                flush=True,
            )
            continue

        try:
            src = Image.open(in_path).convert("RGB")
        except Exception as e:
            print(f"[Worker {rank}] [WARN] 加载失败: {filename} | {e}", flush=True)
            continue

        w, h = src.size
        if not KEEP_SIZE:
            w, h = 768, 768
            src = src.resize((w, h), Image.Resampling.LANCZOS)

        # 每张图使用独立随机状态，保证多卡下也稳定
        random.seed(global_idx + 12345)
        prompt, negative_prompt = build_prompt(global_idx, cfg)

        if SEED is None:
            seed_value = random.randint(1, 2 ** 63)
        else:
            seed_value = SEED + global_idx

        generator = make_generator(device, seed_value)

        print(f"\n[Worker {rank}] PROCESSING: {in_path}", flush=True)
        print(f"[Worker {rank}] Prompt: {prompt}", flush=True)
        print(f"[Worker {rank}] Seed: {seed_value}", flush=True)

        try:
            with torch.inference_mode():
                out = pipe(
                    image=src,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=h,
                    width=w,
                    num_inference_steps=STEPS,
                    true_cfg_scale=TRUE_CFG_SCALE,
                    guidance_scale=GUIDANCE_SCALE,
                    num_images_per_prompt=1,
                    generator=generator,
                )

            out_img = out.images[0]
            out_img.save(out_path)

            print(
                f"[Worker {rank}] [{local_idx}/{total}] 保存成功: {out_path}",
                flush=True,
            )

        except Exception as e:
            print(
                f"[Worker {rank}] [ERROR] 处理失败: {filename} | {e}",
                flush=True,
            )

    print(f"[Worker {rank}] Finished.", flush=True)


# ============================================================
# Main
# ============================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device found.")

    cfg = load_yaml(YAML_PATH)
    image_files = collect_image_files(INPUT_DIR)
    image_files = apply_max_sample(image_files, MAX_SAMPLE)

    if len(image_files) == 0:
        raise RuntimeError(f"No images found in: {INPUT_DIR}")

    num_gpus = torch.cuda.device_count()
    world_size = num_gpus

    print(f"[INFO] Detected GPUs: {num_gpus}")
    print(f"[INFO] Total images: {len(image_files)}")
    print(f"[INFO] Input dir: {INPUT_DIR}")
    print(f"[INFO] Output dir: {OUTPUT_DIR}")

    if world_size == 1:
        worker_process(
            rank=0,
            world_size=1,
            shard=list(enumerate(image_files)),
            cfg=cfg,
        )
        return

    shards = split_data_round_robin(image_files, world_size)

    processes = []

    for rank in range(world_size):
        p = mp.Process(
            target=worker_process,
            args=(rank, world_size, shards[rank], cfg),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("[INFO] All workers finished.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
