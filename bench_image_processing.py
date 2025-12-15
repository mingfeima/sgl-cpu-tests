from PIL import Image
from transformers import AutoImageProcessor
from typing import Union, Sequence, Dict, Any
import torch
import torch.nn.functional as F
import numpy as np

import argparse
from time import time

import sgl_kernel
image_preprocess_cpu = torch.ops.sgl_kernel.image_preprocess_cpu

# set seed
np.random.seed(1234)
torch.manual_seed(1234)

# precision
pres = {
    torch.bfloat16 : 1e-2,
    torch.float32: 1e-5,
}

def _prepare_image_like_inputs(images, do_convert_rgb):
    def _process_image(image, do_convert_rgb):
        if do_convert_rgb:
            image = image.convert("RGB")

        image = torch.from_numpy(np.array(image))

        # If the image is 2D, we need to unsqueeze it to add a channel dimension for processing
        if image.ndim == 2:
            image = image.unsqueeze(0)

        # put it from HWC to CHW format
        image = image.permute((2, 0, 1))

        return image

    return [_process_image(img, do_convert_rgb) for img in images]


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


# https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py
def normalize(tensor: torch.Tensor, mean: list[float], std: list[float], inplace: bool = False) -> torch.Tensor:
    if not tensor.is_floating_point():
        raise TypeError(f"Input tensor should be a float tensor. Got {tensor.dtype}.")

    if tensor.ndim < 3:
        raise ValueError(
            f"Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = {tensor.size()}"
        )

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    return tensor.sub_(mean).div_(std)

def qwen2vl_preprocess_torch(
    images: Union[torch.Tensor, Sequence[torch.Tensor]],
    do_convert_rgb: bool = True,
    do_resize: bool = True,
    size: Dict[str, int] = {"shortest_edge": 65536, "longest_edge": 16777216},
    interpolation: str = "bicubic",
    do_rescale: bool = True,
    rescale_factor: float = 1/255,
    do_normalize: bool = True,
    image_mean: Sequence[float] = [0.5, 0.5, 0.5],
    image_std: Sequence[float] = [0.5, 0.5, 0.5],
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    disable_grouping: bool = True,
    out_dtype: torch.dtype = torch.float,
) -> Dict[str, torch.Tensor]:
    """
    Preprocess images for Qwen2VL model in PyTorch.
    """

    if not isinstance(images, list):
        images = [images]

    images = _prepare_image_like_inputs(images, do_convert_rgb)

    processed_images = []
    processed_grids = []
    for image in images:
        height, width = image.shape[-2:]
        if do_resize:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=patch_size * merge_size,
                min_pixels=size["shortest_edge"],
                max_pixels=size["longest_edge"],
            )
            resized_image = F.interpolate(
                image.unsqueeze(0),
                size=(resized_height, resized_width),
                mode=interpolation,
                align_corners=False,
                antialias=True,
            )

        if do_rescale and do_normalize:
            # Fused rescale and normalize
            image_mean = torch.tensor(image_mean) * (1.0 / rescale_factor)
            image_std = torch.tensor(image_std) * (1.0 / rescale_factor)
            do_rescale = False

        if do_normalize:
            patches = normalize(resized_image.to(dtype=torch.float32), image_mean, image_std)
        elif do_rescale:
            patches = resized_image * rescale_factor

        if patches.ndim == 4:
            # add a temporal dimension if we have images
            patches = patches.unsqueeze(1)
        if patches.shape[1] % temporal_patch_size != 0:
            repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
            patches = torch.cat([patches, repeats], dim=1)
        batch_size, grid_t, channel = patches.shape[:3]
        grid_t = grid_t // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

        patches = patches.view(
            batch_size,
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
        # Reorder dimensions to group grid and patch information for subsequent flattening.
        # (batch, grid_t, grid_h, grid_w, merge_h, merge_w, channel, temp_patch_size, patch_h, patch_w)
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        flatten_patches = patches.reshape(
            batch_size,
            grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )
        processed_images.append(flatten_patches.squeeze(0))
        processed_grids.append([grid_t, grid_h, grid_w] * batch_size)

    pixel_values = torch.cat(processed_images, dim=0)
    image_grid_thw = torch.tensor(processed_grids)

    return {"pixel_values": pixel_values.to(out_dtype), "image_grid_thw": image_grid_thw}

def qwen2vl_preprocess_cpu(
    images: Union[torch.Tensor, Sequence[torch.Tensor]],
    do_convert_rgb: bool = True,
    do_resize: bool = True,
    size: Dict[str, int] = {"shortest_edge": 65536, "longest_edge": 16777216},
    interpolation: str = "bicubic",
    do_rescale: bool = True,
    rescale_factor: float = 1/255,
    do_normalize: bool = True,
    image_mean: Sequence[float] = [0.5, 0.5, 0.5],
    image_std: Sequence[float] = [0.5, 0.5, 0.5],
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    disable_grouping: bool = True,
    out_dtype: torch.dtype = torch.float,
) -> Dict[str, torch.Tensor]:
    """
    Preprocess images for Qwen2VL model in optimized CPU kernels.
    """

    if not isinstance(images, list):
        images = [images]

    images = _prepare_image_like_inputs(images, do_convert_rgb)

    pixel_values, image_grid_thw = image_preprocess_cpu(
        images,
        do_convert_rgb,
        do_resize,
        size["shortest_edge"],
        size["longest_edge"],
        interpolation,
        do_rescale,
        rescale_factor,
        do_normalize,
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        patch_size,
        temporal_patch_size,
        merge_size,
        disable_grouping,
        out_dtype)

    return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}


def create_random_image(min_size=1365, max_size=2048):
    """ Randomly create image"""
    width = torch.randint(low=min_size, high=max_size+1, size=(1,)).item()
    height = torch.randint(low=min_size, high=max_size+1, size=(1,)).item()
    img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


def run_single_benchmark(processor, bfloat16, min_size, max_size, nwarmup, niters, batch_size, validate):

    out_dtype = torch.bfloat16 if bfloat16 else torch.float32
    print(f"\n### Run image preprocessor benchmark with {batch_size:d} images with out dtype",out_dtype)
    print("### Image Min Size ", min_size, " and Max Size ", max_size)

    nwarmp = max(nwarmup // batch_size, 5)
    niters = niters // batch_size
    assert niters > 0, f"niters too small for batch_size={batch_size}"

    images = []
    for i in range(batch_size):
        img = create_random_image(min_size, max_size)
        images.append(img)

    # do warmups
    for _ in range(nwarmup):
        image_processor = processor.preprocess(images=images, return_tensors="pt")
        if bfloat16 and isinstance(image_processor.get("pixel_values"), torch.Tensor):
            image_processor["pixel_values"] = image_processor["pixel_values"].to(torch.bfloat16)

    # do benchmark
    start = time()
    for _ in range(niters):
        image_processor = processor.preprocess(images=images, return_tensors="pt")
        if bfloat16 and isinstance(image_processor.get("pixel_values"), torch.Tensor):
            image_processor["pixel_values"] = image_processor["pixel_values"].to(torch.bfloat16)
    end = time()
    total_time = (end - start) * 1000 / niters # ms
    avg_time = total_time / batch_size

    # do warmups
    for _ in range(nwarmup):
        image_processor_cpu = qwen2vl_preprocess_cpu(images, out_dtype=out_dtype)

    # do benchmark
    start = time()
    for _ in range(niters):
        image_processor_cpu = qwen2vl_preprocess_cpu(images, out_dtype=out_dtype)
    end = time()
    total_time_cpu = (end - start) * 1000 / niters # ms
    avg_time_cpu = total_time_cpu / batch_size

    print("")
    print("  Output shape :", image_processor['pixel_values'].shape, image_processor['image_grid_thw'].shape)
    print("  Output dtype :", image_processor['pixel_values'].dtype, image_processor['image_grid_thw'].dtype)
    print("  Average time : {:.3f} ms v.s. {:.3f} ms".format(avg_time, avg_time_cpu))
    print("  Total time   : {:.3f} ms v.s. {:.3f} ms".format(total_time, total_time_cpu))

    if validate:
        image_processor_torch = qwen2vl_preprocess_torch(images, out_dtype=out_dtype)

        atol = rtol = pres[out_dtype]
        res = torch.allclose(image_processor['pixel_values'], image_processor_torch['pixel_values'], rtol=rtol, atol=atol)
        res2 = torch.equal(image_processor['image_grid_thw'], image_processor_torch['image_grid_thw'])
        print("\n### validating original preprocessor with pure torch preprocessor : ", res, res2)

        res = torch.allclose(image_processor['pixel_values'], image_processor_cpu['pixel_values'], rtol=rtol, atol=atol)
        res2 = torch.equal(image_processor['image_grid_thw'], image_processor_cpu['image_grid_thw'])
        print("### validating original preprocessor with cpu optimized preprocessor : ", res, res2)


def benchmark_image_preprocessor(processor, bfloat16, min_size, max_size, nwarmup, niters, batches, profile, validate):
    for batch_size in batches:
        with torch.autograd.profiler.profile(enabled=profile) as prof:
            run_single_benchmark(processor, bfloat16, min_size, max_size, nwarmup, niters, batch_size, validate)

        if profile:
            print(prof.key_averages().table(sort_by="cpu_time_total"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bfloat16", action='store_true', default=False)
    parser.add_argument("--min", type=int, default=1365)
    parser.add_argument("--max", type=int, default=2048)
    parser.add_argument("--nwarmup", type=int, default=20)
    parser.add_argument("--niters", type=int, default=200)
    parser.add_argument("--batches", type=int, nargs='+', default=[20])
    parser.add_argument("--model_path", type=str, default="./models/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--profile", action='store_true', default=False)
    parser.add_argument("--validate", action='store_true', default=False)
    args = parser.parse_args()

    processor = AutoImageProcessor.from_pretrained(args.model_path)
    print("\n### processor: ", processor)

    benchmark_image_preprocessor(
        processor,
        args.bfloat16,
        args.min,
        args.max,
        args.nwarmup,
        args.niters,
        args.batches,
        args.profile,
        args.validate
    )
