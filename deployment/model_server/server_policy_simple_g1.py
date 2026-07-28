from __future__ import annotations

import argparse
import gc
import logging
import os
import socket
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from DiT4DiT.model.framework.__init__ import build_framework
from DiT4DiT.model.framework.share_tools import dict_to_namespace, read_mode_config


MIN_MAX_ACTION_DIMS = slice(0, 32)
MEAN_STD_ACTION_DIMS = slice(32, 36)
DEFAULT_OPENOVEN_PROMPT = "move forward to the oven and open it"


def _load_model(
    ckpt_path: Path,
    base_model: Path | None,
    device: torch.device,
    use_bf16: bool,
    cpu_offload_text_encoder: bool,
):
    model_config, norm_stats = read_mode_config(ckpt_path)
    if base_model is not None:
        model_config["framework"]["cosmos25"]["base_model"] = str(base_model)

    cfg = dict_to_namespace(model_config)
    cfg.trainer.pretrained_checkpoint = None
    model = build_framework(cfg=cfg)
    model.norm_stats = norm_stats
    model.simple_input_image_size = tuple(model_config["datasets"]["vla_data"].get("image_size", [224, 224]))

    try:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict
    gc.collect()

    if cpu_offload_text_encoder:
        if use_bf16:
            model.backbone_interface.extractor.transformer.to(dtype=torch.bfloat16)
            model.backbone_interface.extractor.vae.to(dtype=torch.bfloat16)
            model.action_model.to(dtype=torch.bfloat16)
        model.backbone_interface.extractor.transformer.to(device)
        model.backbone_interface.extractor.vae.to(device)
        model.action_model.to(device)
        model.backbone_interface.extractor.text_encoder.to(device="cpu", dtype=torch.float32)
    else:
        if use_bf16:
            model = model.to(torch.bfloat16)
        model.to(device)
    return model.eval()


def _resolve_stats(norm_stats: dict[str, Any], unnorm_key: str | None) -> dict[str, Any]:
    if unnorm_key is None:
        if len(norm_stats) != 1:
            raise ValueError(f"--unnorm_key is required, available keys: {list(norm_stats)}")
        unnorm_key = next(iter(norm_stats))
    if unnorm_key not in norm_stats:
        raise ValueError(f"Unknown unnorm key {unnorm_key!r}, available keys: {list(norm_stats)}")
    return norm_stats[unnorm_key]


def _normalize_state(state: np.ndarray, state_stats: dict[str, Any]) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    min_v = np.asarray(state_stats["min"], dtype=np.float32)
    max_v = np.asarray(state_stats["max"], dtype=np.float32)
    normalized = np.zeros_like(state, dtype=np.float32)
    mask = max_v != min_v
    normalized[..., mask] = 2.0 * (state[..., mask] - min_v[mask]) / (max_v[mask] - min_v[mask]) - 1.0
    normalized[..., ~mask] = 0.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def _unnormalize_openoven_actions(normalized_actions: np.ndarray, action_stats: dict[str, Any]) -> np.ndarray:
    normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
    actions = np.zeros_like(normalized_actions, dtype=np.float32)

    min_v = np.asarray(action_stats["min"], dtype=np.float32)
    max_v = np.asarray(action_stats["max"], dtype=np.float32)
    mean_v = np.asarray(action_stats["mean"], dtype=np.float32)
    std_v = np.asarray(action_stats["std"], dtype=np.float32)

    min_max = np.clip(normalized_actions[..., MIN_MAX_ACTION_DIMS], -1.0, 1.0)
    actions[..., MIN_MAX_ACTION_DIMS] = (
        0.5
        * (min_max + 1.0)
        * (max_v[MIN_MAX_ACTION_DIMS] - min_v[MIN_MAX_ACTION_DIMS])
        + min_v[MIN_MAX_ACTION_DIMS]
    )
    actions[..., MEAN_STD_ACTION_DIMS] = (
        normalized_actions[..., MEAN_STD_ACTION_DIMS] * std_v[MEAN_STD_ACTION_DIMS]
        + mean_v[MEAN_STD_ACTION_DIMS]
    )
    return actions.astype(np.float32)


def _resize_image(image: Any, image_size: tuple[int, int]) -> Image.Image:
    width, height = int(image_size[1]), int(image_size[0])
    if isinstance(image, Image.Image):
        return image.convert("RGB").resize((width, height))

    if isinstance(image, np.ndarray):
        array = image
        if not array.flags.writeable:
            array = array.copy()
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB").resize((width, height))

    return Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB").resize((width, height))


class SimpleG1DiT4DiTPolicy:
    def __init__(
        self,
        ckpt_path: Path,
        base_model: Path | None,
        device: torch.device,
        use_bf16: bool,
        unnorm_key: str | None,
        cpu_offload_text_encoder: bool,
        cache_prompt_embeds: bool,
        cached_prompt: str,
    ):
        self._model = _load_model(ckpt_path, base_model, device, use_bf16, cpu_offload_text_encoder)
        stats = _resolve_stats(self._model.norm_stats, unnorm_key)
        self._state_stats = stats["state"]
        self._action_stats = stats["action"]
        self._image_size = getattr(self._model, "simple_input_image_size", (224, 224))
        self._cached_prompt_embeds = self._precompute_prompt_embeds(cached_prompt) if cache_prompt_embeds else None

    def _precompute_prompt_embeds(self, prompt: str) -> torch.Tensor:
        extractor = self._model.backbone_interface.extractor
        device = extractor.device
        dtype = extractor.transformer.dtype
        with torch.inference_mode():
            prompt_embeds, _ = extractor.encode_prompt(
                prompt=[prompt],
                guidance_scale=1.0,
                num_videos_per_prompt=1,
                device=device,
                dtype=dtype,
                max_sequence_length=extractor.max_sequence_length,
            )
        extractor.text_encoder = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cached prompt embeds for fixed SIMPLE prompt: %r", prompt)
        return prompt_embeds

    def predict_action(self, payload: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, np.ndarray]:
        if payload is None:
            payload = kwargs

        batch_images = payload.get("batch_images")
        instructions = payload.get("instructions")
        state = payload.get("state")

        if batch_images is None:
            raise ValueError("SIMPLE payload must contain `batch_images`")
        if instructions is None:
            raise ValueError("SIMPLE payload must contain `instructions`")
        if state is None:
            raise ValueError("SIMPLE payload must contain `state`")

        normalized_state = _normalize_state(np.asarray(state, dtype=np.float32), self._state_stats)
        examples = [
            {
                "image": [_resize_image(image, self._image_size) for image in images],
                "lang": instruction,
                "state": sample_state,
                **({"prompt_embeds": self._cached_prompt_embeds} if self._cached_prompt_embeds is not None else {}),
            }
            for images, instruction, sample_state in zip(batch_images, instructions, normalized_state, strict=True)
        ]

        with torch.inference_mode():
            output = self._model.predict_action(examples)

        normalized_actions = output["normalized_actions"]
        actions = _unnormalize_openoven_actions(normalized_actions, self._action_stats)
        return {"actions": actions[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a DiT4DiT G1 wholebody policy for SIMPLE websocket evaluation.")
    parser.add_argument(
        "--ckpt_path",
        default="/home/d013/桌面/CKPT/DIT4DIT/OPENOVEN/checkpoints/steps_40000_pytorch_model.pt",
    )
    parser.add_argument("--base_model", default=None, help="Local Cosmos-Predict2.5-2B directory. Overrides checkpoint config.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10090)
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--cpu_offload_text_encoder", action="store_true")
    parser.add_argument("--cache_prompt_embeds", action="store_true")
    parser.add_argument("--cached_prompt", default=DEFAULT_OPENOVEN_PROMPT)
    parser.add_argument("--unnorm_key", default=None)
    parser.add_argument("--idle_timeout", type=int, default=-1)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    base_model = Path(args.base_model).expanduser().resolve() if args.base_model else None
    if base_model is not None and not base_model.exists():
        raise FileNotFoundError(f"Cosmos base model directory not found: {base_model}")

    for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    policy = SimpleG1DiT4DiTPolicy(
        ckpt_path=ckpt_path,
        base_model=base_model,
        device=device,
        use_bf16=args.use_bf16,
        unnorm_key=args.unnorm_key,
        cpu_offload_text_encoder=args.cpu_offload_text_encoder,
        cache_prompt_embeds=args.cache_prompt_embeds,
        cached_prompt=args.cached_prompt,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating SIMPLE DiT4DiT server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"env": "simple_g1_openoven", "checkpoint": str(ckpt_path)},
    )
    logging.info("server running ...")
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
