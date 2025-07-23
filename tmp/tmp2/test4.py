from typing import Generic, TypeVar

T_BaseDataInstance = TypeVar("T_BaseDataInstance")


class AtriaDataset(Generic[T_BaseDataInstance]):
    """
    Generic base class for datasets with subclass validation.
    """

    # Default values
    __abstract__ = True
    __iterator__ = None
    __default_config_path__ = "conf/dataset/config.yaml"
    __default_metadata_path__ = "metadata.yaml"
    __repr_fields__ = ["data_model", "data_dir", "train", "validation", "test"]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Skip validation only if THIS class explicitly sets __abstract__ = True
        # Don't inherit the abstract flag from parent classes
        if "__abstract__" in cls.__dict__ and cls.__dict__["__abstract__"]:
            return

        # Define required class variables
        required_vars = {"__data_model__": "Data model class for type validation"}

        # Check for required class variables
        missing = []
        for var_name, description in required_vars.items():
            if not hasattr(cls, var_name):
                missing.append(f"{var_name} ({description})")

        if missing:
            raise TypeError(
                f"Class '{cls.__name__}' must define the following class variables:\n"
                + "\n".join(f"  - {var}" for var in missing)
                + f"\n\nExample:\nclass {cls.__name__}(AtriaDataset[YourDataType]):\n"
                + "    __data_model__ = YourDataType"
            )

        # Validate __data_model__ is a type
        if hasattr(cls, "__data_model__"):
            data_model = getattr(cls, "__data_model__")
            if not isinstance(data_model, type):
                raise TypeError(
                    f"Class '{cls.__name__}.__data_model__' must be a type, "
                    f"got {type(data_model).__name__}: {data_model}"
                )

    def __init__(self, **kwargs):
        super().__init__()


# Example implementations
class DocumentDataset(AtriaDataset[str]):
    __data_model__ = str


class ImageDataset(AtriaDataset[bytes]):
    __data_model__ = bytes
    __iterator__ = "custom_iterator"  # Override default


# Abstract intermediate class (won't be validated)
class AbstractTextDataset(AtriaDataset[str]):
    __abstract__ = True  # Skip validation for this class

    def preprocess_text(self, text: str) -> str:
        return text.lower()


class ConcreteTextDataset(AbstractTextDataset):
    __data_model__ = str  # Required even for subclasses of abstract classes


# These work fine
doc_dataset = DocumentDataset()
img_dataset = ImageDataset()
text_dataset = ConcreteTextDataset()

# This would fail at class definition:
# class BadDataset(AtriaDataset[str]):
#     pass
# TypeError: Class 'BadDataset' must define the following class variables
