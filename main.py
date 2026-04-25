import os
import re
import random
import argparse
from dataclasses import dataclass
from datetime import datetime
from PIL import ImageDraw
import json
import torch
import torchvision

from src.utils import seed_everything, load_config, suppress_print, preprocess_prompt, draw_orientation, ignore_kwargs
from src.flux_pipeline import FluxSchnellPipeline
from src.reward_model import get_reward_model


@ignore_kwargs
@dataclass
class Config:
    seed: int = 0
    negative_prompt: str = None
    height: int = 512
    width: int = 512


def sanitize_filename(s: str, max_len: int = 50) -> str:
    """Convert a string into a safe folder/file name."""
    s = re.sub(r'[^a-zA-Z0-9\s\-_]', '', s)
    s = s.replace(' ', '_')
    return s[:max_len]


def flatten_orientation(orient):
    """Convert nested orientation list into a single underscore-separated string."""
    if isinstance(orient, list):
        return '_'.join(str(flatten_orientation(x)) for x in orient)
    return str(orient)


def main(CFG, args):
    device = torch.device("cuda:0")
    task_name = args.config.split("/")[-1].split(".")[0]

    # ---- Generate random seed if not provided or set to 0 ----
    if CFG.seed == 0:
        # Use a random seed between 1 and 2**32-1
        generated_seed = random.randint(1, 2**32 - 1)
        print(f"No seed provided (value 0). Generating random seed: {generated_seed}")
        CFG.seed = generated_seed
    else:
        print(f"Using config‑provided seed: {CFG.seed}")

    # Set seed for reproducibility (torch, numpy, random)
    seed_everything(CFG.seed)
    print(f"Final seed used for this run: {CFG.seed}")

    # ---- Load data early to build dynamic folder name ----
    with open(args.data_path, 'r') as f:
        data = json.load(f)

    orientations = data['orientations'][0][0]
    prompt = data["prompts"]
    phrases = data['phrases']

    orientation_str = flatten_orientation(orientations)
    safe_prompt = sanitize_filename(prompt)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    unique_folder = f"seed{CFG.seed}_{safe_prompt}_{orientation_str}_{timestamp}"
    base_dir = args.save_dir
    args.save_dir = os.path.join(base_dir, unique_folder)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Output will be saved to: {args.save_dir}")

    # ---- Pipeline and reward model ----
    with suppress_print():
        pipe = FluxSchnellPipeline(device, CFG)
        reward_model = get_reward_model(task_name)(torch.float32, device, CFG)

    # Prepare data
    data['orientations'] = orientations
    prompt = preprocess_prompt(prompt, phrases, orientations)

    pipe.load_encoder()
    negative_prompt = CFG.get("negative_prompt", None)
    pipe.encode_prompt(prompt, negative_prompt, phrases=phrases)
    reward_model.register_data(data)
    pipe.unload_encoder()

    generator = torch.Generator(device=device).manual_seed(CFG.seed)
    _, best_sample, best_reward = pipe.sample(
        height=CFG.height, width=CFG.width,
        reward_model=reward_model, generator=generator
    )

    image = torchvision.transforms.ToPILImage()(best_sample[0].float().cpu().clamp(0, 1))
    image.save(os.path.join(args.save_dir, "output.png"))

    if args.save_reward:
        draw = ImageDraw.Draw(image)
        text = f"{best_reward.item():.5f}" if hasattr(best_reward, "item") else f"{best_reward:.5f}"
        draw.rectangle([0, 0, 60, 20], fill=(0, 0, 0, 128))
        draw.text((5, 2), text, fill=(255, 255, 255))

        estimated_angles = reward_model.get_angle(best_sample)
        estimated_bboxes = reward_model.estimated_bboxes

        image = draw_orientation(image, estimated_bboxes, estimated_angles)
        image.save(os.path.join(args.save_dir, "output_orientation_rendered.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/orientation_grounding.yaml")
    parser.add_argument("--data_path", default="./data/single.json")
    parser.add_argument("--save_reward", action="store_true")
    parser.add_argument("--save_dir", default="./outputs")

    args, extras = parser.parse_known_args()
    CFG = load_config(args.config, cli_args=extras)

    CFG.save_dir = args.save_dir
    main(CFG, args)
