import os
import random
import yaml
import torch
import multiprocessing as mp

from PIL import Image
from diffusers import Flux2KleinPipeline


# ============================================================
# Config
# ============================================================

YAML_PATH = "keywords-CN.yaml"

INPUT_DIR = "/DATA_71/b59900515/data/candidate_hair/tgt"
OUTPUT_BASE_DIR = "/DATA_71/b59900515/data/candidate_hair/ref_flux2_klein"

MODEL_PATH = "black-forest-labs/FLUX.2-klein-base-9B"

STEPS = 4
GUIDANCE_SCALE = 4.0

SEED = None  # None 表示每张图随机
KEEP_SIZE = True
OUT_EXT = ".jpg"

SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")

OVERWRITE = False
MAX_SAMPLE = None  # None 表示处理全部图片；整数表示只处理排序后的前 N 张
VALID_ID_EDIT_MODES = ("age", "gender", "ethnicity")
ID_EDIT_MODE = os.environ.get("ID_EDIT_MODE", "age").strip().lower()
OUTPUT_DIR = f"{OUTPUT_BASE_DIR}_{ID_EDIT_MODE}"


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
        "age",
        "gender",
        "ethnicity",
    ]

    for key in required_keys:
        ensure_list_str(cfg, key)

    return cfg


def validate_id_edit_mode(edit_mode):
    if edit_mode not in VALID_ID_EDIT_MODES:
        valid_modes = ", ".join(VALID_ID_EDIT_MODES)
        raise ValueError(f"ID_EDIT_MODE must be one of: {valid_modes}")
    return edit_mode


def build_identity_clause(sample, edit_mode):
    edit_mode = validate_id_edit_mode(edit_mode)

    if edit_mode == "age":
        age = random.choice(sample["age"])
        return f"将人物年龄阶段改为{age}，保持原图中可见的性别表达和人种外观尽量自然一致，"

    if edit_mode == "gender":
        gender = random.choice(sample["gender"])
        return f"将人物性别表达改为{gender}，保持原图中可见的年龄阶段和人种外观尽量自然一致，"

    ethnicity = random.choice(sample["ethnicity"])
    return f"将人物身份改为具有{ethnicity}外貌特征的人物，保持原图中可见的年龄阶段和性别表达尽量自然一致，"


def build_prompt(idx, sample, edit_mode=None):
    edit_mode = validate_id_edit_mode(edit_mode or ID_EDIT_MODE)
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
    identity_clause = build_identity_clause(sample, edit_mode)

    prompt = (
        f"保持输入图中人物的发型、发色、头发长度、刘海形状和头发轮廓完全不变，"
        f"{identity_clause}"
        f"只改变面部身份、年龄感、性别表达或肤色五官特征，"
        f"不要改变发型，不要改变发色，不要改变头发长度，不要添加帽子或头饰，"
        f"人物肖像照，真实摄影风格，清晰面部，4K。"
    )

    negative_prompt = (
        "改变发型，改变发色，改变头发长度，头发变短，头发变长，刘海变化，卷直变化，"
        "戴帽子，头饰，遮挡头发，头发缺失，躯体畸形，面部扭曲，低质量"
    )

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
    pipe = Flux2KleinPipeline.from_pretrained(
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
                    height=h,
                    width=w,
                    num_inference_steps=STEPS,
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

    validate_id_edit_mode(ID_EDIT_MODE)
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
