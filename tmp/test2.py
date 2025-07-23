class Optimizer:
    algo: str
    lr: float

    def __init__(self, algo: str, lr: float) -> None:
        self.algo = algo
        self.lr = lr


import importlib
from typing import cast

from hydra_zen import instantiate
from omegaconf import OmegaConf


def get_class_from_target_path(path: str) -> type:
    module_path, _, class_name = path.rpartition(".")
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


REG: dict[str, type] = {__name__ + ".Optimizer": Optimizer}

path = __name__ + ".Optimizer"
cfg = OmegaConf.create({"optimizer": {"_target_": path, "algo": "SGD", "lr": 0.01}})

opt = instantiate(cfg.optimizer)
opt = cast(REG[path], opt)  # ✅ now statically known
print(opt.algo, opt.lr)  # Output: SGD 0.01
