import os

import lakefs
from deltalake import DeltaTable

os.environ["LAKECTL_SERVER_ENDPOINT_URL"] = "http://127.0.0.1:8090"
os.environ["LAKECTL_CREDENTIALS_ACCESS_KEY_ID"] = "AKIAJ2NNAE7KY6KJDFVQ"
os.environ["LAKECTL_CREDENTIALS_SECRET_ACCESS_KEY"] = (
    "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"
)

# objects = lakefs.Repository("p-testuser12345-cifar10-2").branch("main").objects()
# for obj in objects:
#     print(obj)

print("Verifying lakeFS credentials…")
try:
    v = lakefs.client.Client().version
except:
    print("🛑 failed to get lakeFS version")
else:
    print(f"…✅lakeFS credentials verified\n\nℹ️lakeFS version {v}")

storage_options = {
    "AWS_ACCESS_KEY_ID": "AKIAJ2NNAE7KY6KJDFVQ",
    "AWS_SECRET_ACCESS_KEY": "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",
    "AWS_ENDPOINT": "http://127.0.0.1:8090",
    "AWS_REGION": "us-east-1",
    "AWS_ALLOW_HTTP": "true",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}
path = "lakefs://testuser12345-cifar10/main/delta/train"

print("path", path)
delta_table = DeltaTable(path, storage_options=storage_options)

parrow = delta_table.to_pyarrow_dataset()
# Convert to pandas DataFrame for easier data inspection
# Stream data based on index

# Method 1: Stream specific rows by index range
start_idx = 0
batch_size = 1000
scanner = parrow.scanner(columns=None)

print(f"Streaming data in batches of {batch_size}")
for i, batch in enumerate(scanner.to_batches()):
    df_batch = batch.to_pandas()
    print(
        f"Batch {i}: rows {start_idx + i * batch_size} to {start_idx + (i + 1) * batch_size - 1}"
    )
    print(f"Shape: {df_batch.shape}")

    # Process your batch here
    if i == 0:  # Show first batch details
        print(f"Columns: {df_batch.columns.tolist()}")
        print("\nFirst few rows of batch:")
        print(df_batch.head())

    # Break after a few batches for demo
    if i >= 2:
        break

# Method 2: Stream specific row indices
specific_indices = [0, 100, 500, 1000]  # Rows you want
for idx in specific_indices:
    try:
        # Get single row by taking slice
        single_row = parrow.take([idx]).to_table().to_pandas()
        print(f"\nRow {idx}:")
        print(single_row.iloc[0])
    except:
        print(f"Row {idx} not found")
