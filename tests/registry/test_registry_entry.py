import omegaconf
import pytest
from atria_registry.registry_entry import RegistryEntry


@pytest.mark.parametrize(
    "group, module_name, module_path, package, registers_target, build_kwargs, throws_validation_error, zen_partial, hydra_defaults",
    [
        (
            "test_group_1",
            "test_module_1",
            "utilities.MockClass2",
            None,
            False,
            {"var1": 1, "var2": 2},
            False,
            False,
            None,
        ),
        (
            "test_group_2",
            "test_module_2",
            "utilities.MockClass3",
            None,
            False,
            {"var1": 1, "var2": 2},
            False,
            False,
            None,
        ),
        (
            "test_group_3",
            "test_module_3",
            "utilities.MockClass4",
            "__global__",
            True,
            {"var1": 1, "var2": 2},
            False,
            False,
            None,
        ),
        (
            "test_group_3",
            "test_module_3",
            "utilities.MockClass4",
            "__global__",
            True,
            {"var1": "wrong_type", "var2": "wrong_type"},
            True,
            False,
            None,
        ),
        (
            "test_group_3",
            "test_module_3",
            "utilities.MockClass4",
            "__global__",
            True,
            {"var1": 1, "var2": 2},
            False,
            True,
            None,
        ),
        (
            "test_group_3",
            "test_module_3",
            "utilities.MockClass4",
            "__global__",
            True,
            {"var1": 1, "var2": 2},
            False,
            True,
            ["_self_", {"/parent@child": "test"}],
        ),
    ],
)
def test_registry_entry_register_success_basic(
    group,
    module_name,
    module_path,
    package,
    registers_target,
    build_kwargs,
    throws_validation_error,
    zen_partial,
    hydra_defaults,
):
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    registry_entry = RegistryEntry(
        group=group,
        module_name=module_name,
        module_path=module_path,
        package=package,
        registers_target=registers_target,
        build_kwargs={
            **build_kwargs,
            "zen_partial": zen_partial,
            "hydra_defaults": hydra_defaults,
        },
    )
    try:
        registry_entry.register()
    except omegaconf.errors.ValidationError:
        assert throws_validation_error is True
        return
    assert throws_validation_error is False
    module_name = f"{module_name}.yaml"
    assert registry_entry.is_registered is True
    assert group in cs.repo
    assert module_name in cs.repo[group]
    if registers_target:
        assert cs.repo[group][module_name].node["_target_"] == module_path
        if zen_partial:
            assert cs.repo[group][module_name].node["_partial_"] is True
    else:
        assert "_target_" not in cs.repo[group][module_name].node
        assert "_partial_" not in cs.repo[group][module_name].node
    assert cs.repo[group][module_name].group == group
    assert cs.repo[group][module_name].package == package
    for key, value in build_kwargs.items():
        assert cs.repo[group][module_name].node[key] == value
    if hydra_defaults is not None:
        assert "defaults" in cs.repo[group][module_name].node
        assert cs.repo[group][module_name].node["defaults"] == hydra_defaults
