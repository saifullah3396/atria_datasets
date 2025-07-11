from atria_registry import RegistryGroup
from atria_registry.module_registry import ModuleRegistry

from atria_datasets.registry.registry_groups import (
    DatasetRegistryGroup,
    DataTransformRegistryGroup,
)

_initialized = False


def init_registry():
    global _initialized
    if _initialized:
        return
    _initialized = True
    ModuleRegistry().add_registry_group(
        name="DATASET", registry_group=DatasetRegistryGroup(name="dataset")
    )
    ModuleRegistry().add_registry_group(
        name="DATA_TRANSFORM",
        registry_group=DataTransformRegistryGroup(name="data_transform"),
    )
    ModuleRegistry().add_registry_group(
        name="DATA_PIPELINE", registry_group=RegistryGroup(name="data_pipeline")
    )
    ModuleRegistry().add_registry_group(
        name="BATCH_SAMPLER", registry_group=RegistryGroup(name="batch_sampler")
    )
