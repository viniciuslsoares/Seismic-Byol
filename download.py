import os
import urllib.request
import zipfile

def download_and_extract(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    filename = os.path.join(dest_folder, url.split('/')[-1])

    if not os.path.exists(filename):
        print(f'Downloading {url}...')
        urllib.request.urlretrieve(url, filename)
        print(f'Successfully downloaded {filename}')
    else:
        print(f'{filename} already exists, skipping download.')

    # Extract if it's a zip file
    if filename.endswith(".zip"):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)
        print(f'Extracted {filename}')

# Paths where the data will be downloaded and extracted
coco_data_dir = '../shared_data/data_vinicius/coco'

# COCO 2017 URLs
urls = [
    'http://images.cocodataset.org/zips/train2017.zip',
    'http://images.cocodataset.org/zips/val2017.zip',
    'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
]

# Download and extract COCO 2017 images and annotations
for url in urls:
    download_and_extract(url, coco_data_dir)

print("COCO dataset downloaded and extracted.")
