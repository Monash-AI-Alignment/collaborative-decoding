"""
Core library for weak-to-strong research.
Self-contained, no external dependencies.
"""

from .decode_config import DecodeConfig

try:
    from .config import (
        RunConfig,
        create_run_arg_parser,
        BASELINE_EPOCHS,
    )
except ImportError:
    pass

try:
    from .data import (
        load_dataset,
        format_classification_as_causal,
        detect_aar_mode,
    )
except ImportError:
    pass

try:
    from .train import (
        train_model,
        find_latest_checkpoint,
        load_model_from_checkpoint,
        normalize_model_name_for_path,
        is_base_model,
    )
except ImportError:
    pass

try:
    from .eval import (
        evaluate_model,
        print_evaluation_results,
        generate_predictions,
        save_predictions,
        compute_metrics_from_predictions,
    )
except ImportError:
    pass

try:
    from .vllm_inference import (
        predict_batch_labels,
    )
except ImportError:
    pass

try:
    from .seed_utils import (
        set_seed,
    )
except ImportError:
    pass

__all__ = [
    # Decoding
    "DecodeConfig",
    # Config
    "RunConfig",
    "create_run_arg_parser",
    "BASELINE_EPOCHS",
    # Data
    "load_dataset",
    "format_classification_as_causal",
    "detect_aar_mode",
    # Training
    "train_model",
    "find_latest_checkpoint",
    "load_model_from_checkpoint",
    "normalize_model_name_for_path",
    "is_base_model",
    "evaluate_model",
    "print_evaluation_results",
    "generate_predictions",
    "save_predictions",
    "compute_metrics_from_predictions",
    # Inference
    "predict_batch_labels",
    # Seed utilities
    "set_seed",
]
