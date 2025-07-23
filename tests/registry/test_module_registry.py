from atria_registry.module_registry import ModuleRegistry
from atria_registry.registry_group import RegistryGroup
from utilities import MockClass3, MockClass4, MockDatasetClass

from atria_datasets import BATCH_SAMPLER, DATA_PIPELINE, DATASET


def test_registry_imports():
    assert DATASET == ModuleRegistry().DATASET and isinstance(DATASET, RegistryGroup)
    assert DATA_PIPELINE == ModuleRegistry().DATA_PIPELINE and isinstance(
        DATA_PIPELINE, RegistryGroup
    )
    assert BATCH_SAMPLER == ModuleRegistry().BATCH_SAMPLER and isinstance(
        BATCH_SAMPLER, RegistryGroup
    )


def test_get_registry_group_existing():
    for group in ModuleRegistry()._registry_groups:
        registry_group = ModuleRegistry()[group]
        if registry_group._is_factory:
            assert registry_group._name == group.lower() + "_factory"
        else:
            assert registry_group._name == group.lower()


def test_get_registry_group_non_existing():
    registry_group = ModuleRegistry().get_registry_group("NON_EXISTING_GROUP")
    assert registry_group is None


def test_register_all_modules():
    ModuleRegistry()["DATASET"].register("test_module1")(MockDatasetClass)
    ModuleRegistry()["DATASET"].register("test_group/test_module2")(MockDatasetClass)
    ModuleRegistry()["BATCH_SAMPLER"].register("test_module3")(MockClass3)
    ModuleRegistry()["BATCH_SAMPLER"].register("test_module4")(MockClass4)
    ModuleRegistry().register_all_modules()

    def assert_module_in_registry(group, module_name, registry_group):
        module_found = False
        for x in ModuleRegistry()[registry_group].registered_modules:
            if x == (group, module_name):
                module_found = True
                break
        if not module_found:
            raise AssertionError(
                f"Module {module_name} in group {group} not found in registry group {registry_group}"
                f"Registered modules: {ModuleRegistry()[registry_group].registered_modules.keys()}"
            )

    assert_module_in_registry("dataset/test_module1", "mock_config", "DATASET")
    assert_module_in_registry(
        "dataset/test_group/test_module2", "mock_config", "DATASET"
    )
    assert_module_in_registry("batch_sampler", "test_module3", "BATCH_SAMPLER")
    assert_module_in_registry("batch_sampler", "test_module4", "BATCH_SAMPLER")
