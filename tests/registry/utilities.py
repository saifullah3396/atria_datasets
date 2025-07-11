from atria_core.types import AtriaDatasetConfig


class MockDatasetClass:
    _REGISTRY_CONFIGS = [AtriaDatasetConfig(config_name="mock_config")]


class MockClass1:
    pass


class MockClass2:
    def __init__(self, var1: int, var2: int):
        self.var1 = var1
        self.var2 = var2


class MockClass3:
    def __init__(self, var1: int, var2: int):
        self.var1 = var1
        self.var2 = var2


class MockClass4:
    def __init__(self, var1: int, var2: int):
        self.var1 = var1
        self.var2 = var2
