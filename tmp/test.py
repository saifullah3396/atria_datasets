from atria_core.types import ImageInstance
from atria_core.utilities.imports import _get_package_base_path

from atria_datasets import AtriaImageDataset, SplitIterator

package_path = _get_package_base_path("atria")
dataset = AtriaImageDataset.load_from_registry(
    name="cifar10",
    provider="atria_datasets",
    build_kwargs={
        "max_train_samples": 1000,
        "max_test_samples": 1000,
        "max_validation_samples": 1000,
    },
)

x: SplitIterator[ImageInstance] = dataset.train
y = x[0]
print(y.to_tensor())
print(type(x))  # Should print the type of the data instance, e.g., BaseDataInstance
