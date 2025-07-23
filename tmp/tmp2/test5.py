import io
import tarfile

# Your tar file and internal file path
base_path = "/media/aletheia/a865e985-032a-4793-9899-2063093eac27/home/ataraxia/.atria/datasets/rvlcdip/storage/image_with_ocr_1k/"
tar_path = "shards/000000-000000.tar"
internal_path = "80035521_tif.image_content"

# Open tar and extract the specific file
with tarfile.open(f"{base_path}/{tar_path}", "r") as tar:
    member = tar.getmember(internal_path)
    file_obj = tar.extractfile(member)
    data = file_obj.read()  # This is a bytes object
    from PIL import Image

    image = Image.open(io.BytesIO(data))
    image = image.convert("RGB")  # Convert to RGB if needed
    image.save("test.jpg")  # Save the image
