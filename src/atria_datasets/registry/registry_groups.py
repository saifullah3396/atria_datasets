from typing import TYPE_CHECKING

from atria_registry import RegistryGroup

if TYPE_CHECKING:
    pass


class DatasetRegistryGroup(RegistryGroup):
    """
    A specialized registry group for managing datasets.

    This class provides additional methods for registering and managing datasets
    within the registry system.
    """

    def register(self, dataset_name: str, **kwargs):  # type: ignore
        """
        Decorator for registering a module with configurations.

        Args:
            name (str): The name of the module.
            **kwargs: Additional keyword arguments for the registration.

        Returns:
            function: A decorator function for registering the module with configurations.
        """

        def decorator(decorated_class):
            from atria_core.types import AtriaDatasetConfig

            if hasattr(decorated_class, "_REGISTRY_CONFIGS"):
                configs = decorated_class._REGISTRY_CONFIGS
                assert isinstance(configs, list), (
                    f"Expected _REGISTRY_CONFIGS on {decorated_class.__name__} to be a list, "
                    f"but got {type(configs).__name__} instead."
                )
                assert configs, (
                    f"{decorated_class.__name__} must provide at least one configuration in _REGISTRY_CONFIGS."
                )
                for config in configs:
                    assert isinstance(config, AtriaDatasetConfig), (
                        f"Configuration {config} must be a subclass of AtriaDatasetConfig."
                    )
                    config.dataset_name = dataset_name
                    self.register_modules(
                        module_paths=decorated_class,
                        module_names=config.dataset_name + "/" + config.config_name,
                        **config.model_dump(),
                        **kwargs,
                    )
                return decorated_class
            else:
                module_name = dataset_name
                self.register_modules(
                    module_paths=decorated_class,
                    module_names=module_name,
                    dataset_name=dataset_name,
                    **kwargs,
                )
                return decorated_class

        return decorator


class DataTransformRegistryGroup(RegistryGroup):
    """
    A specialized registry group for managing data transformations.

    This class provides additional methods for registering and managing data
    transformations within the registry system.
    """

    def register_torchvision_transform(self, transform: str, **kwargs):
        """
        Register a torchvision transform.

        Args:
            transform (str): The name of the torchvision transform.
            **kwargs: Additional keyword arguments for the registration.
        """
        from atria_core.utilities.strings import _convert_to_snake_case

        from atria_datasets.transforms.torchvision import TorchvisionTransform

        self.register_modules(
            module_paths=TorchvisionTransform,
            module_names=_convert_to_snake_case(transform),
            transform=transform,
            **kwargs,
        )

    def load_from_registry(  # type: ignore
        self, name: str, provider: str | None = "atria", config: dict = None
    ):
        """
        Load a data transformation from the registry.

        Args:
            name (str): The name of the transformation.
            provider (str, optional): The provider name. Defaults to "atria".
            config (dict, optional): Additional configuration for the transformation. Defaults to None.

        Returns:
            Any: The instantiated transformation.
        """
        from atria_registry.utilities import _overrides_from_dict
        from hydra import compose, initialize
        from hydra_zen import instantiate
        from hydra_zen.third_party.pydantic import pydantic_parser

        config = config or {}
        config_path = f"pkg://{provider}/conf"
        with initialize(version_base=None, config_path=config_path):
            config_name = f"{self.name}/{name}.yaml"
            overrides = ["hydra.searchpath=[pkg://atria]"]
            module_path = ".".join(config_name.split("/")[:-1])
            overrides += _overrides_from_dict(config, module_path)
            cfg = compose(config_name=config_name, overrides=overrides)
            obj = instantiate(cfg, _convert_="object", _target_wrapper_=pydantic_parser)
            for path in module_path.split("."):
                obj = obj[path]
            return obj
