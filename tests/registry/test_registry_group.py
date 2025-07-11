from utilities import MockClass1


def test_add_module_to_registry_group(test_registry_group):
    test_registry_group.register("mock_module")(MockClass1)
    registered_modules = list(test_registry_group.registered_modules.values())
    lazy_registered_modules = list(test_registry_group.lazy_registered_modules.values())
    assert len(registered_modules) == 0
    assert len(lazy_registered_modules) == 1
    assert lazy_registered_modules[0].group == "mock_group"
    assert lazy_registered_modules[0].module_name == "mock_module"
    assert lazy_registered_modules[0].module_path == MockClass1


def test_register_modules_default_params(test_registry_group):
    module_path = "utilities.MockClass1"
    test_registry_group.register_modules(
        module_path, module_names="mock_class", lazy_build=False
    )
    registered_modules = list(test_registry_group.registered_modules.values())
    assert len(registered_modules) == 1
    assert registered_modules[0].module_name == "mock_class"
    assert registered_modules[0].module_path == module_path


def test_register_modules_multiple(test_registry_group):
    test_registry_group.register_modules(
        ["module_1_path", "module_2_path"],
        module_names=["module_1", "module_2"],
        lazy_build=True,
    )
    lazy_registered_modules = list(test_registry_group.lazy_registered_modules.values())
    assert len(lazy_registered_modules) == 2
    for module in lazy_registered_modules:
        assert module.group == "mock_group"
    assert lazy_registered_modules[0].module_name == "module_1"
    assert lazy_registered_modules[1].module_name == "module_2"


def test_register_all_modules(test_registry_group):
    lazy_module = "utilities.MockClass1"
    test_registry_group.register_modules(lazy_module, lazy_build=True)
    test_registry_group.register_all_modules()
    registered_modules = list(test_registry_group.registered_modules.values())
    assert len(test_registry_group.registered_modules) == 1
    assert registered_modules[0].group == "mock_group"
    assert registered_modules[0].module_name == "mock_class1"
    assert registered_modules[0].module_path == lazy_module
