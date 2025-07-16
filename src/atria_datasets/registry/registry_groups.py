from atria_registry import RegistryGroup


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
        from atria_core.utilities.auto_config import auto_config

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
                return auto_config()(decorated_class)
            else:
                module_name = dataset_name
                self.register_modules(
                    module_paths=decorated_class,
                    module_names=module_name,
                    dataset_name=dataset_name,
                    **kwargs,
                )
                return auto_config()(decorated_class)

        return decorator
