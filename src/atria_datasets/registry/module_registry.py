from atria_registry import RegistryGroup
from atria_registry.module_registry import ModuleRegistry

from atria_datasets.registry.registry_groups import DatasetRegistryGroup

ModuleRegistry().add_registry_group(
    name="DATASET", registry_group=DatasetRegistryGroup(name="dataset")
)
ModuleRegistry().add_registry_group(
    name="DATA_PIPELINE", registry_group=RegistryGroup(name="data_pipeline")
)
ModuleRegistry().add_registry_group(
    name="BATCH_SAMPLER", registry_group=RegistryGroup(name="batch_sampler")
)
