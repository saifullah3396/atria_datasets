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
        self,
        name: str,
        configs: list[AtriaDatasetConfig] | None = None,
        builds_to_file_store: bool = True,
        **kwargs,
    ):
        """
        Decorator for registering a module with configurations.

        Args:
            name (str): The name of the module.
            **kwargs: Additional keyword arguments for the registration.

        Returns:
            function: A decorator function for registering the module with configurations.
        """
        from atria_datasets.core.dataset.atria_dataset import (
            AtriaDataset,
            AtriaDatasetConfig,
        )
        from atria_datasets.core.dataset.atria_huggingface_dataset import (
            AtriaHuggingfaceDataset,
        )

        if builds_to_file_store and not self._file_store_build_enabled:

            def noop_(module):
                return module

            return noop_

        # get spec params
        configs = configs or []
        provider = kwargs.pop("provider", None)
        is_global_package = kwargs.pop("is_global_package", False)
        registers_target = kwargs.pop("registers_target", True)
        defaults = kwargs.pop("defaults", None)
        assert defaults is None, "Dataset registry does not support defaults."

        def decorator(module):
            from atria_registry.module_spec import ModuleSpec

            assert issubclass(module, AtriaDataset), (
                f"Expected {module.__name__} to be a subclass of AtriaDataset, got {type(module)} instead."
            )
            if not issubclass(module, AtriaHuggingfaceDataset):
                configs.append(
                    module.__config_cls__(
                        config_name="default", dataset_name=name, **kwargs
                    )
                )
            assert isinstance(configs, list) and all(
                isinstance(config, AtriaDatasetConfig) for config in configs
            ), (
                f"Expected configs to be a list of AtriaDatasetConfig, got {type(configs)} instead."
            )
            assert len(configs) > 0, (
                f"{module.__name__} must provide at least one config."
            )

            # build the module spec
            module_spec = ModuleSpec(
                module=module,
                name=name,
                group=self.name,
                provider=provider or self._default_provider,
                is_global_package=is_global_package,
                registers_target=registers_target,
                defaults=defaults,
            )

            import copy

            for config in configs:
                config.dataset_name = name
                config_module_spec = copy.deepcopy(module_spec)
                config_module_spec.name = config.dataset_name + "/" + config.config_name
                config_module_spec.model_extra.update(
                    {k: getattr(config, k) for k in config.__class__.model_fields}
                )
                config_module_spec.model_extra.update({**kwargs})
                self.register_module(config_module_spec)
            return module

        return decorator
