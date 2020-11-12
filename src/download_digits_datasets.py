"""Downloads digits datasets from the URL in the resources folder."""

import os
import requests, zipfile, io
from urllib.request import urlopen
import wget

DIGITS_DATASETS_URL = "https://www.dropbox.com/s/awr6ocrtfo00ffn/digits_datasets.zip"

def download_dataset(model_dir, zip_file_url):
  zip_file_name = zip_file_url.split('/')[-1]
  zip_file_path = os.path.join(os.getcwd(), zip_file_name)
  if not os.path.exists(zip_file_path):
    os.system('wget ' + zip_file_url)

  with zipfile.ZipFile(zip_file_path) as zfile:
    zfile.extractall(model_dir)


if __name__ == '__main__':

  # Create a resources directory for all datasets
  datasets_path = os.path.join(os.getcwd(), 'resources')
  if not os.path.exists(datasets_path):
    os.makedirs(datasets_path)

  download_dataset(datasets_path, DIGITS_DATASETS_URL)

  print('Done downloading the dataset.')
