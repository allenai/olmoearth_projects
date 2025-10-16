"""olmoearth_run pipeline."""

from .olmoearth_run import finetune, olmoearth_run, one_stage, prepare_labeled_windows

workflows = {
    "olmoearth_run": olmoearth_run,
    "one_stage": one_stage,
    "prepare_labeled_windows": prepare_labeled_windows,
    "finetune": finetune,
}
