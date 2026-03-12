import torch

from src.diffusers.pipelines.diffuman4d.pipeline_diffuman4d import Diffuman4DPipeline
from src.diffusers.models.wan_vae import AutoencoderKLWan
from src.diffusers.models.unets.unet_multiview_condition import UNetMultiviewConditionModel
from diffusers.schedulers import DDIMScheduler  # 或你项目实际 scheduler

# ===== 1) build modules =====
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

vae = AutoencoderKLWan.from_pretrained(
    pretrained_model_name_or_path="/home/tkw/Project/Diffuman4D/checkpoints",
    vae_filename="Wan2.1_VAE.pth",
    z_dim=16,
).to(device)

unet = UNetMultiviewConditionModel(
    sample_size=32,
    in_channels=16 + 6 + 1,   # latent + plucker + cond_mask，仅示例，按你真实通道改
    out_channels=16,
    block_out_channels=(320, 640, 1280, 1280),
    attention_head_dim=(5, 10, 20, 20),
    cross_attention_dim=1280,
    num_3d_attn_blocks=3,
    enable_tem_embeds=True,
    enable_pose_encoder=False,  # 如启用需提供 skeleton 5D/6D
).to(device, dtype=dtype)

scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
)

pipe = Diffuman4DPipeline(
    vae=vae,
    unet=unet,
    scheduler=scheduler,
).to(device)

# ===== 2) fake inputs (multi-view) =====
B, V, T, C, H, W = 1, 2, 9, 3, 256, 256
pixel_values = torch.randn(B, V, T, C, H, W, device=device, dtype=dtype).clamp(-1, 1)
plucker_embeds = torch.randn(B, V, T, 6, H, W, device=device, dtype=dtype)
cond_masks = torch.zeros(B, V, T, 1, H, W, device=device, dtype=dtype)
cond_masks[:, :, T // 2 :, ...] = 1.0  # 前半输入帧，后半目标帧（示例）

domains = ["spatial"] * B

# ===== Step A: latent output smoke =====
with torch.no_grad():
    latents = pipe(
        pixel_values=pixel_values,
        plucker_embeds=plucker_embeds,
        cond_masks=cond_masks,
        domains=domains,
        num_inference_steps=1,
        guidance_scale=1.0,
        output_type="latent",
    )
print("[A] latent shape:", latents.shape)

# ===== Step B: decode output smoke =====
with torch.no_grad():
    images = pipe(
        pixel_values=pixel_values,
        plucker_embeds=plucker_embeds,
        cond_masks=cond_masks,
        domains=domains,
        num_inference_steps=1,
        guidance_scale=1.0,
        output_type="pt",  # 会走 post_process/decode
    )
# 可能是 list/np/pt，按你 image_processor 配置判断
print("[B] decode type:", type(images))

# ===== Step C: sliding smoke =====
# 先准备 timestep_indices，长度要等于 B*V*T'
# 这里先用 A 步返回 latent 的 N 作为长度
timestep_indices = torch.zeros(latents.shape[0], device=device, dtype=torch.long)

with torch.no_grad():
    out = pipe.sliding_iterative_denoise(
        pixel_values=pixel_values,
        plucker_embeds=plucker_embeds,
        cond_masks=cond_masks,
        timestep_indices=timestep_indices,
        domain="spatial",          # 或 "temporal"
        window_size=2,
        sliding_stride=1,
        num_denoising_steps=1,
        alternation_rounds=1,
        guidance_scale=1.0,
    )
print("[C] sliding keys:", out.keys())
print("[C] sliding latents shape:", out["latents"].shape)

# ===== Step D: temporal sliding smoke (half-half layout) =====
# Temporal mode assumes [cond_half, target_half] after VAE temporal compression.
# Use T=13 so Wan latent length is T'=4, then set mask half-half in raw frame space.
T_temporal = 13
pixel_values_t = torch.randn(B, V, T_temporal, C, H, W, device=device, dtype=dtype).clamp(-1, 1)
plucker_embeds_t = torch.randn(B, V, T_temporal, 6, H, W, device=device, dtype=dtype)
cond_masks_t = torch.zeros(B, V, T_temporal, 1, H, W, device=device, dtype=dtype)
cond_masks_t[:, :, T_temporal // 2 :, ...] = 1.0

with torch.no_grad():
    latents_t = pipe(
        pixel_values=pixel_values_t,
        plucker_embeds=plucker_embeds_t,
        cond_masks=cond_masks_t,
        domains=["temporal"] * B,
        num_inference_steps=1,
        guidance_scale=1.0,
        output_type="latent",
    )
timestep_indices_t = torch.zeros(latents_t.shape[0], device=device, dtype=torch.long)

with torch.no_grad():
    out_t = pipe.sliding_iterative_denoise(
        pixel_values=pixel_values_t,
        plucker_embeds=plucker_embeds_t,
        cond_masks=cond_masks_t,
        timestep_indices=timestep_indices_t,
        domain="temporal",
        window_size=2,
        sliding_stride=1,
        num_denoising_steps=1,
        alternation_rounds=1,
        guidance_scale=1.0,
    )
print("[D] temporal sliding keys:", out_t.keys())
print("[D] temporal sliding latents shape:", out_t["latents"].shape)
