#
# (c) All rights reserved. ECOLE POLYTECHNIQUE FÉDÉRALE DE LAUSANNE,
# Switzerland, Laboratory of Prof. Mackenzie W. Mathis (UPMWMATHIS) and
# original authors: Steffen Schneider, Jin H Lee, Mackenzie W Mathis. 2023.
#
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/LICENSE.md
#

import requests
import os
from tqdm import tqdm
import re

def download_file_with_progress_bar(url, location):
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code != 200:
        print(f"Error occurred while downloading the file. Response code: {response.status_code}")
        return None

    # Check if the response headers contain the 'Content-Disposition' header
    if 'Content-Disposition' not in response.headers:
        print("Unable to determine the filename. 'Content-Disposition' header not found.")
        return None

    # Extract the filename from the 'Content-Disposition' header
    filename_match = re.search(r'filename="(.+)"', response.headers.get("Content-Disposition"))
    if not filename_match:
        print("Unable to determine the filename from the 'Content-Disposition' header.")
        return None

    # Create the directory and any necessary parent directories
    os.makedirs(location, exist_ok=True)
    
    filename = filename_match.group(1)
    file_path = os.path.join(location,filename)

    total_size = int(response.headers.get("Content-Length", 0))

    with open(file_path, "wb") as file:
        with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                progress_bar.update(len(data))

    print(f"Download complete. Dataset saved in '{file_path}'")
    return response