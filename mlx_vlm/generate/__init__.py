"""Public generation API."""

from .ar import (
    BatchGenerator,
    BatchResponse,
    BatchStats,
    PromptProcessingBatch,
    batch_generate,
    generate_step,
)
from .cli import main, parse_arguments
from .common import (  # Fork: the 5e9b9503 engine port's own helpers
    GenerationResult,
    PromptCacheState,
    _adjust_chunk_for_snapshot_landing,
    _anchor_within_loop_range,
    _capture_anchor_state,
    _capture_arrays_layers_for_snapshot,
    _capture_rotating_layers_for_snapshot,
    _classify_snapshot_action,
    _compute_anchor_before_latest_user_offset,
    _first_kv_offset,
    _get_generation_stream,
    _has_non_trimmable,
    _is_rotating_kv_layer,
    _kv_seq_axis,
    _restore_arrays_layers_from_snapshots,
    _restore_deltanet_state,
    _restore_rotating_layers_from_snapshots,
    _rotating_post_gen_trim_safe,
    _rotating_rewind_safe,
    _should_capture_anchor_pre_prefill,
    _trim_cache,
    maybe_preallocate_kv_cache,
    maybe_quantize_kv_cache,
    wired_limit,
)
from .dispatch import generate, stream_generate
from .edit_image import (
    ImageEditModel,
    ImageEditRequest,
    edit_image,
    image_edit_model_class,
    is_image_edit_model,
    load_image_edit_model,
)
from .image import (
    ImageGenerationModel,
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageTask,
    generate_image,
    image_generation_model_class,
    image_to_b64_json,
    image_to_png_bytes,
    is_image_generation_model,
    load_image_generation_model,
    load_image_model,
)
from .types import GenerateKwargs, ProcessorLike
from .video_generation import (
    VideoGenerationModel,
    VideoGenerationRequest,
    VideoGenerationResult,
    VideoProgressCallback,
    VideoReference,
    VideoReferenceKind,
    VideoWorkflow,
    generate_video,
    is_video_generation_model,
    load_video_generation_model,
    save_video,
    video_generation_model_class,
)

__all__ = [
    "BatchGenerator",
    "BatchResponse",
    "BatchStats",
    "GenerateKwargs",
    "GenerationResult",
    "ImageEditModel",
    "ImageEditRequest",
    "ImageGenerationModel",
    "ImageGenerationRequest",
    "ImageGenerationResult",
    "ImageTask",
    "PromptCacheState",
    "PromptProcessingBatch",
    "ProcessorLike",
    "VideoGenerationModel",
    "VideoGenerationRequest",
    "VideoGenerationResult",
    "VideoProgressCallback",
    "VideoReference",
    "VideoReferenceKind",
    "VideoWorkflow",
    "batch_generate",
    "edit_image",
    "generate",
    "generate_image",
    "generate_step",
    "generate_video",
    "generation_stream",
    "image_edit_model_class",
    "image_generation_model_class",
    "image_to_b64_json",
    "image_to_png_bytes",
    "is_image_edit_model",
    "is_image_generation_model",
    "is_video_generation_model",
    "load_image_edit_model",
    "load_image_generation_model",
    "load_image_model",
    "load_video_generation_model",
    "main",
    # Fork: 7de8f7f1's KV pre-allocation entry point.
    "maybe_preallocate_kv_cache",
    "maybe_quantize_kv_cache",
    "parse_arguments",
    "save_video",
    "stream_generate",
    "video_generation_model_class",
    "wired_limit",
]


def __getattr__(name):
    # Fork: `common` is in the lazy-import tuple because the fork's generation
    # engine lives there (5e9b9503); upstream's tuple has no `common`. The
    # `generation_stream` special case keeps that name working as an attribute
    # while deferring Metal stream creation to first use.
    import importlib

    from . import ar, common, dispatch, image, video_generation

    edit_image_module = importlib.import_module("mlx_vlm.generate.edit_image")

    if name == "generation_stream":
        return common._get_generation_stream()
    if hasattr(common, name):
        return getattr(common, name)
    if hasattr(dispatch, name):
        return getattr(dispatch, name)
    if hasattr(edit_image_module, name):
        return getattr(edit_image_module, name)
    if hasattr(image, name):
        return getattr(image, name)
    if hasattr(video_generation, name):
        return getattr(video_generation, name)
    return getattr(ar, name)


def __dir__():
    # Fork: `common` is in the lazy-import tuple because the fork's generation
    # engine lives there (5e9b9503); upstream's tuple has no `common`. The
    # `generation_stream` special case keeps that name working as an attribute
    # while deferring Metal stream creation to first use.
    import importlib

    from . import ar, common, dispatch, image, video_generation

    edit_image_module = importlib.import_module("mlx_vlm.generate.edit_image")

    return sorted(
        set(__all__)
        | set(dir(ar))
        | set(dir(common))
        | set(dir(dispatch))
        | set(dir(edit_image_module))
        | set(dir(image))
        | set(dir(video_generation))
    )
