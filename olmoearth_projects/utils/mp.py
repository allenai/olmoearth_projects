"""Multi-processing utilities."""

import multiprocessing


def init_mp() -> None:
    """Set start method to preload and configure forkserver preload."""
    multiprocessing.set_start_method("forkserver", force=True)
    multiprocessing.set_forkserver_preload(
        [
            "pickle",
            "fiona",
            "gcsfs",
            "jsonargparse",
            "numpy",
            "PIL",
            "torch",
            "torch.multiprocessing",
            "torchvision",
            "upath",
            "wandb",
            "rslearn.main",
            "rslearn.train.dataset",
            "rslearn.train.data_module",
        ]
    )
