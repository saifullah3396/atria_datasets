from pathlib import Path

from atria_registry.utilities import write_registry_to_yaml

from atria_datasets.image_classification.cifar10 import *  # noqa

if __name__ == "__main__":
    write_registry_to_yaml(str(Path(__file__).parent / "conf"))
