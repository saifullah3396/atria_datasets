from atria_registry import RegistryGroup


class DatasetRegistryGroup(RegistryGroup):
    """
    A specialized registry group for managing datasets.

    This class provides additional methods for registering and managing datasets
    within the registry system.
    """

    # def register(self, dataset_name: str, **kwargs):  # type: ignore
    #     """
    #     Decorator for registering a module with configurations.

    #     Args:
    #         name (str): The name of the module.
    #         **kwargs: Additional keyword arguments for the registration.

    #     Returns:
    #         function: A decorator function for registering the module with configurations.
    #     """

    #     def decorator(decorated_class):
    #         if hasattr(decorated_class, "_REGISTRY_CONFIGS"):
    #             configs = decorated_class._REGISTRY_CONFIGS
    #             assert isinstance(configs, dict), (
    #                 f"Expected _REGISTRY_CONFIGS on {decorated_class.__name__} to be a dict, "
    #                 f"but got {type(configs).__name__} instead."
    #             )
    #             assert configs, (
    #                 f"{decorated_class.__name__} must provide at least one configuration in _REGISTRY_CONFIGS."
    #             )
    #             for key, config in configs.items():
    #                 assert isinstance(config, dict), (
    #                     f"Configuration {config} must be a dict."
    #                 )
    #                 module_name = dataset_name
    #                 self.register_modules(
    #                     module_paths=decorated_class,
    #                     module_names=module_name + "/" + key,
    #                     exclude_fields=["dataset_name", "config_name"],
    #                     **config,
    #                     **kwargs,
    #                 )
    #             return auto_config(exclude=["dataset_name", "config_name"])(
    #                 decorated_class
    #             )
    #         else:
    #             self.register_modules(
    #                 module_paths=decorated_class,
    #                 module_names=dataset_name,
    #                 exclude_fields=["dataset_name", "config_name"],
    #                 **kwargs,
    #             )
    #             return auto_config(exclude=["dataset_name", "config_name"])(
    #                 decorated_class
    #             )

    #     return decorator
