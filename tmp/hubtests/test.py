from atria_datasets import AtriaImageDataset

# package_path = _get_package_base_path("atria")
# dataset = AtriaImageDataset.load_from_registry(
#     name="cifar10",
#     provider="atria_datasets",
#     build_kwargs={
#         "max_train_samples": 1000,
#         "max_test_samples": 1000,
#         "max_validation_samples": 1000,
#     },
# )
# dataset.train.dataframe()
# dataset.upload_to_hub(name="cifar10-2", branch="test2")
dataset = AtriaImageDataset.load_from_hub(
    name="cifar10-2", branch="test2", streaming=True
)
