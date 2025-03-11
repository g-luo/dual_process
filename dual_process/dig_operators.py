import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms

from dual_process import dig_helpers

IGNORE_INDEX = -100

# =====================================
#     Image Generator Predicted x0
# =====================================
def encode_images(pipe, image, generator_kwargs):
    device, dtype = pipe.vae.device, pipe.vae.dtype
    image = pipe.image_processor.preprocess(
        image,
        height=generator_kwargs["height"],
        width=generator_kwargs["width"]
    )
    image = image.to(device).to(dtype)
    scale_factor = pipe.vae.config.get("scaling_factor", 1) or 1
    shift_factor = pipe.vae.config.get("shift_factor", 0) or 0
    latents = pipe.vae.encode(image).latent_dist.sample()
    latents = (latents - shift_factor) * scale_factor
    if hasattr(pipe, "_pack_latents"):
        backbone = dig_helpers.get_backbone(pipe)
        num_channels_latents = backbone.config.in_channels // 4
        latents = pipe._pack_latents(
            latents=latents,
            batch_size=image.shape[0],
            num_channels_latents=num_channels_latents,
            height=latents.shape[-2],
            width=latents.shape[-1]
        )
    return latents

def decode_images(pipe, latents, generator_kwargs):
    if hasattr(pipe, "_unpack_latents"):
        latents = pipe._unpack_latents(
            latents=latents, 
            vae_scale_factor=pipe.vae_scale_factor, 
            height=generator_kwargs["height"],
            width=generator_kwargs["width"]
        )
    scale_factor = pipe.vae.config.get("scaling_factor", 1) or 1
    shift_factor = pipe.vae.config.get("shift_factor", 0) or 0
    latents = (latents / scale_factor) + shift_factor
    latents = latents.to(pipe.vae.device)
    latents = latents.to(pipe.vae.dtype)
    image = pipe.vae.decode(latents).sample
    return image

def get_x0(pipe, latents, noise_pred, i, t, generator_kwargs):
    train_scheduler = dig_helpers.get_train_scheduler(pipe)
    scheduler_cls = train_scheduler._class_name
    if scheduler_cls == "FlowMatchEulerDiscreteScheduler":
        sigma = train_scheduler.sigmas[i]
        pred_x0 = latents - sigma * noise_pred
    elif scheduler_cls == "DDIMScheduler":
        train_scheduler.alphas_cumprod = train_scheduler.alphas_cumprod.to(t.device)
        outputs = train_scheduler.step(noise_pred, t, latents)
        pred_x0 = outputs.pred_original_sample
    else:
        raise NotImplementedError
    pred_x0 = decode_images(pipe, pred_x0, generator_kwargs)
    return pred_x0

# =====================================
#            VLM Prompting
# =====================================
def create_vlm_prompt(vlm_processor, question, answer, vlm_template, ref_images=[], image_dim=None, overlay_mode=None):
    num_images = max(1, len(ref_images))
    # If overlay then feed only one image to VLM
    if overlay_mode:
        num_images = 1
    # Prepare VLM input
    prefix = vlm_template["prefix"].format(
        image_token=vlm_template["image_token"]*num_images, 
        question=question
    )
    blank_images = [Image.new('RGB', image_dim, color='white')] * num_images
    prefix_ids = vlm_processor(text=prefix, images=blank_images, return_tensors="pt")["input_ids"]
    all_ids = vlm_processor(text=prefix + answer, images=blank_images, return_tensors="pt")["input_ids"]
    # Find the indices where prefix_ids and all_ids differ
    search_idxs = (prefix_ids != all_ids[:, :prefix_ids.shape[1]]).nonzero(as_tuple=True)[1].tolist()
    search_idxs += list(range(prefix_ids.shape[1], all_ids.shape[1]))
    guidance_prompt = {
        "input_ids": all_ids,
        "search_token_idxs": search_idxs,
        "image_dim": image_dim,
        "question": question,
        "answer": answer,
        "ref_images": ref_images,
        "overlay_mode": overlay_mode
    }
    return guidance_prompt

# =====================================
#         VLM Image Processing
# =====================================
def _preprocess_qwen_image(vlm_processor, patches):
    # Create differentiable version of qwen preprocessing
    temporal_patch_size = vlm_processor.image_processor.temporal_patch_size
    patch_size = vlm_processor.image_processor.patch_size
    merge_size = vlm_processor.image_processor.merge_size
    # Pad to temporal dimension
    if patches.shape[0] % temporal_patch_size != 0:
        patches = torch.cat([patches] * temporal_patch_size, dim=0)
    channel = patches.shape[1]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h, grid_w = patches.shape[2] // patch_size, patches.shape[3] // patch_size
    # Interpolate to multiplier of patch size
    patches = torch.nn.functional.interpolate(patches, size=(grid_h * patch_size, grid_w * patch_size), mode="bilinear")
    patches = patches.view(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
    )
    return flatten_patches, (grid_t, grid_h, grid_w)

def preprocess_qwen_image(vlm_processor, patches):
    pixel_values, image_grid_thw = zip(*[_preprocess_qwen_image(vlm_processor, patch[None, ...]) for patch in patches])
    pixel_values = torch.stack(pixel_values)
    image_grid_thw = torch.tensor(image_grid_thw)
    return pixel_values, image_grid_thw

def preprocess_vlm_image(pipe, vlm_processor, pred_x0, image_dim=None):
    pixel_values = pred_x0
    vlm_kwargs = {}
    # Renormalize
    pixel_values = dig_helpers.renormalize(pixel_values, (pixel_values.min(), pixel_values.max()), (0, 1))
    mean = vlm_processor.image_processor.image_mean
    std = vlm_processor.image_processor.image_std
    pixel_values = torchvision.transforms.functional.normalize(pixel_values, mean, std)
    # Resize
    processor_size = getattr(vlm_processor.image_processor, "crop_size", None) or getattr(vlm_processor.image_processor, "size", None)
    if processor_size is not None:
        h, w = processor_size.get("height"), processor_size.get("width")
        if h is not None and w is not None:
            image_dim = (h, w)
    if image_dim is not None:
        pixel_values = torch.nn.functional.interpolate(pixel_values, size=image_dim, mode="bilinear")
    # idefics2
    if "Idefics2" in vlm_processor.image_processor.image_processor_type:
        pixel_values = pixel_values[:, None]
    # qwenvl
    elif "Qwen" in vlm_processor.image_processor.image_processor_type:
        pixel_values, image_grid_thw = preprocess_qwen_image(vlm_processor, pixel_values)
        vlm_kwargs["image_grid_thw"] = image_grid_thw
    # pixtral
    elif "Pixtral" in vlm_processor.image_processor.image_processor_type:
        vlm_kwargs["image_sizes"] = torch.tensor([image_dim] * len(pixel_values)).to(pixel_values.device)
    return pixel_values, vlm_kwargs

def create_visual_prompt(pipe, guidance_prompt, pred_x0, ref_images, mode="pt"):
    if pipe is None:
        preprocess_fn = lambda x: transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])(x)[None, ...]
        postprocess_fn = lambda x: transforms.ToPILImage()(
            dig_helpers.renormalize(x[0], (-1, 1), (0, 1))
        )
        device = "cpu"
    else:
        preprocess_fn = pipe.image_processor.preprocess
        postprocess_fn = lambda x: pipe.image_processor.postprocess(x, output_type="pil")[0]
        device = pipe.device
    if ref_images:
        ref_image = ref_images[-1]
        overlay_mode = guidance_prompt.get("overlay_mode") # ["transparent", "solid"]
        assert overlay_mode in ["transparent", "solid"], "For overlay must specify overlay_mode!"
        assert len(ref_images) == 1, "For overlay only one ref_image is supported!"
        if not torch.is_tensor(pred_x0):
            pred_x0 = preprocess_fn(pred_x0)
        # Convert ref_image to tensor
        blend_x0 = preprocess_fn(ref_image)
        blend_x0 = torch.nn.functional.interpolate(blend_x0, size=pred_x0.shape[2:], mode="bilinear")
        blend_x0 = blend_x0.to(device)
        pred_x0 = pred_x0.to(device)
        blend_mask = torch.ones_like(blend_x0)
        # For background regions don't overlay
        if overlay_mode == "solid":
            background_color = 1 # white
            background_mask = (blend_x0 == background_color).all(dim=1)
            background_mask = background_mask[:, None].repeat(1, 3, 1, 1)
            blend_mask[background_mask] = 0
        elif overlay_mode == "transparent":
            blend_mask = blend_mask * 0.5 # 50% transparent
        else:
            raise NotImplementedError
        # Blend images
        pred_x0 = blend_mask * blend_x0 + (1 - blend_mask) * pred_x0
        if mode == "pt":
            return pred_x0
        elif mode == "pil":
            return postprocess_fn(pred_x0)
        else:
            raise NotImplementedError
    return pred_x0

def get_vlm_image(pipe, edit, pred_x0, guidance_prompt):
    vlm, vlm_processor = edit["vlm"], edit["vlm_processor"]
    ref_images = guidance_prompt.get("ref_images", [])
    image_dim = guidance_prompt.get("image_dim")

    if not torch.is_tensor(pred_x0):
        pred_x0 = pipe.image_processor.preprocess(pred_x0)
        pred_x0 = pred_x0.to(vlm.device)

    pred_x0 = create_visual_prompt(pipe, guidance_prompt, pred_x0, ref_images, mode="pt")
    pixel_values, vlm_kwargs = preprocess_vlm_image(pipe, vlm_processor, pred_x0, image_dim)
    return pixel_values, vlm_kwargs

# =====================================
#        VLM Argument Processing
# =====================================
def get_optimize_mask(edit, guidance_prompt):
    vlm, vlm_processor = edit["vlm"], edit["vlm_processor"]
    input_ids = guidance_prompt["input_ids"]
    input_ids = input_ids.to(vlm.device)
    search_token_idxs = guidance_prompt.get("search_token_idxs")
    mask = torch.zeros_like(input_ids)
    mask[:, search_token_idxs] = 1
    return input_ids, mask[0].bool()

def get_vlm_args(pipe, edit, pred_x0, guidance_idx=None):
    if guidance_idx is None:
        guidance_idx = np.random.choice(range(len(edit["guidance_prompts"])))
    guidance_prompt = edit["guidance_prompts"][guidance_idx]
    input_ids, optimize_mask = get_optimize_mask(edit, guidance_prompt)
    pixel_values, vlm_kwargs = get_vlm_image(pipe, edit, pred_x0, guidance_prompt)
    return input_ids, pixel_values, vlm_kwargs, optimize_mask

# =====================================
#              VLM Loss
# =====================================
def loss_logits(logits, labels):
    labels = labels.to(logits.device)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=IGNORE_INDEX)
    return loss

def loss_vlm(pipe, edit, generator_kwargs=None, forward_kwargs=None, model_pred=None, i=0, t=0, o=0, pred_x0=None, guidance_idx=None):
    vlm, vlm_processor = edit["vlm"], edit["vlm_processor"]
    device, dtype = vlm.device, vlm.dtype
    # Prepare and run forward
    if pred_x0 is None:
        pred_x0 = get_x0(pipe, forward_kwargs["hidden_states"], model_pred, i, t, generator_kwargs)
    input_ids, pixel_values, vlm_kwargs, optimize_mask = get_vlm_args(pipe, edit, pred_x0, guidance_idx=guidance_idx)
    input_ids = input_ids.to(device)
    pixel_values = pixel_values.to(device, dtype)
    outputs = vlm.forward(input_ids=input_ids, pixel_values=pixel_values, **vlm_kwargs)
    # Compute loss
    logits = outputs.logits
    labels = torch.ones_like(input_ids) * IGNORE_INDEX
    labels[:, optimize_mask] = input_ids[:, optimize_mask].detach().clone()
    loss = loss_logits(logits, labels)
    # Convert cross entropy loss to interpretable probabilities, using the same technique as Lin et. al., ECCV 2024
    # https://github.com/linzhiqiu/t2v_metrics/blob/d7e2a0c85c62a315a5c0dca7015c327e5b752889/t2v_metrics/models/vqascore_models/clip_t5_model.py#L280
    probs = (-loss).exp()
    # Visualize greedy next token prediction
    preds = logits[:, :-1][:, optimize_mask[1:]].argmax(dim=-1)
    preds = vlm_processor.tokenizer.batch_decode(preds, skip_special_tokens=True)[0]
    return loss, {"preds": preds, "probs": probs}

def loss_vlm_multiqa(edit, max_questions=5, **kwargs):
    # Average loss_vlm across multiple questions
    num_questions = len(edit["guidance_prompts"])
    assert num_questions <= max_questions
    loss, preds, probs = [], [], []
    for guidance_idx in range(num_questions):
        loss_j, meta_j = loss_vlm(edit=edit, guidance_idx=guidance_idx, **kwargs)
        loss.append(loss_j)
        preds.append(meta_j["preds"])
        probs.append(meta_j["probs"])
    loss = torch.stack(loss).mean()
    probs = torch.stack(probs)
    return loss, {"preds": preds, "probs": probs}