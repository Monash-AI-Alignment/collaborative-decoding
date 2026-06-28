"""
Core library for weak-to-strong research.
Self-contained, no external dependencies.
"""

from .decode_config import DecodeConfig

__all__ = ["DecodeConfig"]

# The training/inference modules below pull in heavy optional dependencies
# (torch, transformers, vllm, datasets). They are imported best-effort so the
# lightweight decoding path works on a CPU-only environment. Only a genuinely
# missing optional package (ModuleNotFoundError) is tolerated; any other
# ImportError (a real bug in the module) is allowed to propagate.
try:
    from .config import (
        RunConfig,
        create_run_arg_parser,
        BASELINE_EPOCHS,
    )
    __all__ += ["RunConfig", "create_run_arg_parser", "BASELINE_EPOCHS"]
except ModuleNotFoundError:
    pass

try:
    from .data import (
        load_dataset,
        format_classification_as_causal,
        detect_aar_mode,
    )
    __all__ += ["load_dataset", "format_classification_as_causal", "detect_aar_mode"]
except ModuleNotFoundError:
    pass

try:
    from .train import (
        train_model,
        find_latest_checkpoint,
        load_model_from_checkpoint,
        normalize_model_name_for_path,
        is_base_model,
    )
    __all__ += [
        "train_model",
        "find_latest_checkpoint",
        "load_model_from_checkpoint",
        "normalize_model_name_for_path",
        "is_base_model",
    ]
except ModuleNotFoundError:
    pass

try:
    from .eval import (
        evaluate_model,
        print_evaluation_results,
        generate_predictions,
        save_predictions,
        compute_metrics_from_predictions,
    )
    __all__ += [
        "evaluate_model",
        "print_evaluation_results",
        "generate_predictions",
        "save_predictions",
        "compute_metrics_from_predictions",
    ]
except ModuleNotFoundError:
    pass

try:
    from .vllm_inference import (
        predict_batch_labels,
    )
    __all__ += ["predict_batch_labels"]
except ModuleNotFoundError:
    pass

try:
    from .seed_utils import (
        set_seed,
    )
    __all__ += ["set_seed"]
except ModuleNotFoundError:
    pass
