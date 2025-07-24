from __future__ import annotations

from typing import TYPE_CHECKING

from atria_registry import RegistryGroup

if TYPE_CHECKING:
    from atria_datasets.core.dataset.atria_dataset import AtriaDatasetConfig


class DatasetRegistryGroup(RegistryGroup):
    """
    A specialized registry group for managing datasets.

    This class provides additional methods for registering and managing datasets
    within the registry system.
    """

    def register(
        self, name: str, configs: list[AtriaDatasetConfig] | None = None, **kwargs
    ):
        """
        Decorator for registering a module with configurations.

        Args:
            name (str): The name of the module.
            **kwargs: Additional keyword arguments for the registration.

        Returns:
            function: A decorator function for registering the module with configurations.
        """

        from atria_datasets.core.dataset.atria_dataset import AtriaDatasetConfig
        from atria_datasets.core.dataset.atria_huggingface_dataset import (
            AtriaHuggingfaceDataset,
        )

        configs = configs or []

        def decorator(decorated_class):
            if not issubclass(decorated_class, AtriaHuggingfaceDataset):
                configs.append(
                    decorated_class.__config_cls__(
                        config_name="default", dataset_name=name, **kwargs
                    )
                )
            assert isinstance(configs, list) and all(
                isinstance(config, AtriaDatasetConfig) for config in configs
            ), (
                f"Expected configs to be a list of AtriaDatasetConfig, got {type(configs)} instead."
            )
            assert len(configs) > 0, (
                f"{decorated_class.__name__} must provide at least one config."
            )
            for config in configs:
                config.dataset_name = name
                self.register_modules(
                    module_paths=decorated_class,
                    module_names=config.dataset_name + "/" + config.config_name,
                    **{k: getattr(config, k) for k in config.__class__.model_fields},
                    **kwargs,
                )
            return decorated_class

        return decorator
