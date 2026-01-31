#
# CEBRA: Consistent EmBeddings of high-dimensional Recordings using Auxiliary variables
# © Mackenzie W. Mathis & Steffen Schneider (v0.4.0+)
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/blob/main/LICENSE.md
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import gzip
import hashlib
import re
import warnings
from pathlib import Path
from typing import Optional

import requests
import tqdm

_MAX_RETRY_COUNT = 2


def download_file_with_progress_bar(url: str,
                                    expected_checksum: str,
                                    location: str,
                                    file_name: str,
                                    retry_count: int = 0) -> Optional[str]:
    """Download a file from the given URL.

    During download, progress is reported using a progress bar. The downloaded
    file's checksum is compared to the provided ``expected_checksum``.

    Args:
        url: The URL to download the file from.
        expected_checksum: The expected checksum value of the downloaded file.
        location: The directory where the file will be saved.
        file_name: The name of the file.
        retry_count: The number of retry attempts (default: 0).

    Returns:
        The path of the downloaded file if the download is successful, None otherwise.

    Raises:
        RuntimeError: If the maximum ``retry count`` is exceeded.
    """

    # Check if the file already exists in the location
    location_path = Path(location)
    file_path = location_path / file_name

    if file_path.exists():
        existing_checksum = calculate_checksum(file_path)
        if existing_checksum == expected_checksum:
            return file_path

    if retry_count >= _MAX_RETRY_COUNT:
        raise RuntimeError(
            f"Exceeded maximum retry count ({_MAX_RETRY_COUNT}). "
            f"Unable to download the file from {url}")

    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code != 200:
        raise requests.HTTPError(
            f"Error occurred while downloading the file. Response code: {response.status_code}"
        )

    # Check if the response headers contain the 'Content-Disposition' header
    if 'Content-Disposition' not in response.headers:
        raise ValueError(
            "Unable to determine the filename. 'Content-Disposition' header not found."
        )

    # Extract the filename from the 'Content-Disposition' header
    filename_match = re.search(r'filename="(.+)"',
                               response.headers.get("Content-Disposition"))
    if not filename_match:
        raise ValueError(
            "Unable to determine the filename from the 'Content-Disposition' header."
        )

    # Create the directory and any necessary parent directories
    location_path.mkdir(parents=True, exist_ok=True)

    filename = filename_match.group(1)
    file_path = location_path / filename

    total_size = int(response.headers.get("Content-Length", 0))
    checksum = hashlib.md5()  # create checksum

    with open(file_path, "wb") as file:
        with tqdm.tqdm(total=total_size, unit="B",
                       unit_scale=True) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                checksum.update(
                    data)  # Update the checksum with the downloaded data
                progress_bar.update(len(data))

    downloaded_checksum = checksum.hexdigest()  # Get the checksum value
    if downloaded_checksum != expected_checksum:
        warnings.warn(f"Checksum verification failed. Deleting '{file_path}'.")
        file_path.unlink()
        warnings.warn("File deleted. Retrying download...")

        # Retry download using a for loop
        for _ in range(retry_count + 1, _MAX_RETRY_COUNT + 1):
            return download_file_with_progress_bar(url, expected_checksum,
                                                   location, file_name,
                                                   retry_count + 1)
    else:
        print(f"Download complete. Dataset saved in '{file_path}'")
        return url


def calculate_checksum(file_path: str) -> str:
    """Calculate the MD5 checksum of a file.

    Args:
        file_path: The path to the file.

    Returns:
        The MD5 checksum of the file.
    """
    checksum = hashlib.md5()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def download_and_extract_gzipped_file(url: str,
                                      expected_checksum: str,
                                      gzipped_checksum: str,
                                      location: str,
                                      file_name: str,
                                      retry_count: int = 0) -> Optional[str]:
    """Download a gzipped file from the given URL, verify checksums, and extract.

    This function downloads a gzipped file, verifies the checksum of the gzipped
    file, extracts it, and then verifies the checksum of the extracted file.

    Args:
        url: The URL to download the gzipped file from.
        expected_checksum: The expected MD5 checksum of the extracted file.
        gzipped_checksum: The expected MD5 checksum of the gzipped file.
        location: The directory where the file will be saved.
        file_name: The name of the final extracted file (without .gz extension).
        retry_count: The number of retry attempts (default: 0).

    Returns:
        The path of the extracted file if successful, None otherwise.

    Raises:
        RuntimeError: If the maximum retry count is exceeded.
        requests.HTTPError: If the download fails.
    """

    # Check if the final extracted file already exists with correct checksum
    location_path = Path(location)
    final_file_path = location_path / file_name

    if final_file_path.exists():
        existing_checksum = calculate_checksum(final_file_path)
        if existing_checksum == expected_checksum:
            return final_file_path

    if retry_count >= _MAX_RETRY_COUNT:
        raise RuntimeError(
            f"Exceeded maximum retry count ({_MAX_RETRY_COUNT}). "
            f"Unable to download the file from {url}")

    # Create the directory and any necessary parent directories
    location_path.mkdir(parents=True, exist_ok=True)

    # Download the gzipped file
    gz_file_path = location_path / f"{file_name}.gz"

    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code != 200:
        raise requests.HTTPError(
            f"Error occurred while downloading the file. Response code: {response.status_code}"
        )

    total_size = int(response.headers.get("Content-Length", 0))
    checksum = hashlib.md5()  # create checksum for gzipped file

    # Download the gzipped file
    with open(gz_file_path, "wb") as file:
        with tqdm.tqdm(total=total_size,
                       unit="B",
                       unit_scale=True,
                       desc="Downloading") as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                checksum.update(data)
                progress_bar.update(len(data))

    downloaded_gz_checksum = checksum.hexdigest()

    # Verify gzipped file checksum
    if downloaded_gz_checksum != gzipped_checksum:
        warnings.warn(
            f"Gzipped file checksum verification failed. Deleting '{gz_file_path}'."
        )
        gz_file_path.unlink()
        warnings.warn("Gzipped file deleted. Retrying download...")
        return download_and_extract_gzipped_file(url, expected_checksum,
                                                 gzipped_checksum, location,
                                                 file_name, retry_count + 1)

    print("Gzipped file checksum verified. Extracting...")

    # Extract the gzipped file
    try:
        with gzip.open(gz_file_path, 'rb') as f_in:
            with open(final_file_path, 'wb') as f_out:
                # Extract with progress (estimate based on typical compression ratio)
                extracted_size = 0
                while True:
                    chunk = f_in.read(8192)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    extracted_size += len(chunk)
    except Exception as e:
        warnings.warn(f"Extraction failed: {e}. Deleting files and retrying...")
        if gz_file_path.exists():
            gz_file_path.unlink()
        if final_file_path.exists():
            final_file_path.unlink()
        return download_and_extract_gzipped_file(url, expected_checksum,
                                                 gzipped_checksum, location,
                                                 file_name, retry_count + 1)

    # Verify extracted file checksum
    extracted_checksum = calculate_checksum(final_file_path)
    if extracted_checksum != expected_checksum:
        warnings.warn(
            "Extracted file checksum verification failed. Deleting files.")
        gz_file_path.unlink()
        final_file_path.unlink()
        warnings.warn("Files deleted. Retrying download...")
        return download_and_extract_gzipped_file(url, expected_checksum,
                                                 gzipped_checksum, location,
                                                 file_name, retry_count + 1)

    # Clean up the gzipped file after successful extraction
    gz_file_path.unlink()

    print(f"Extraction complete. Dataset saved in '{final_file_path}'")
    return final_file_path
