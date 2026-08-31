from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError


BUCKET_NAME = "amazon-last-mile-challenges"
AWS_REGION = "us-west-2"

PROJECT_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_DIRECTORY / "data"

FILES_TO_DOWNLOAD = {
    "route_data.json": (
        "almrrc2021/"
        "almrrc2021-data-training/"
        "model_build_inputs/"
        "route_data.json"
    ),
    "package_data.json": (
        "almrrc2021/"
        "almrrc2021-data-training/"
        "model_build_inputs/"
        "package_data.json"
    ),
}


def create_public_s3_client():
    """
    Create an anonymous connection to the public S3 bucket.

    No AWS account, access key, secret key, or AWS CLI is required.
    """
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        config=Config(
            signature_version=UNSIGNED,
            connect_timeout=30,
            read_timeout=120,
            retries={"max_attempts": 5},
        ),
    )


def download_file(s3_client, filename, object_key):
    """
    Download one file from Amazon S3 into the project's data folder.
    """
    destination = DATA_DIRECTORY / filename

    print(f"\nPreparing to download: {filename}")
    print(f"S3 location: s3://{BUCKET_NAME}/{object_key}")

    try:
        file_information = s3_client.head_object(
            Bucket=BUCKET_NAME,
            Key=object_key,
        )

        expected_size = file_information["ContentLength"]

    except ClientError as error:
        error_code = error.response.get(
            "Error",
            {},
        ).get(
            "Code",
            "Unknown",
        )

        raise RuntimeError(
            f"Amazon S3 could not access {filename}. "
            f"AWS error code: {error_code}"
        ) from error

    if (
        destination.exists()
        and destination.stat().st_size == expected_size
    ):
        print(
            f"Skipping {filename}. "
            "A complete copy already exists."
        )
        return

    if destination.exists():
        print(
            f"An incomplete copy of {filename} was found. "
            "It will be replaced."
        )
        destination.unlink()

    print(
        "Downloading now. This file is large, so please wait "
        "and do not close PyCharm."
    )

    try:
        s3_client.download_file(
            Bucket=BUCKET_NAME,
            Key=object_key,
            Filename=str(destination),
        )

    except ClientError as error:
        if destination.exists():
            destination.unlink()

        error_code = error.response.get(
            "Error",
            {},
        ).get(
            "Code",
            "Unknown",
        )

        raise RuntimeError(
            f"The download of {filename} failed. "
            f"AWS error code: {error_code}"
        ) from error

    except Exception:
        if destination.exists():
            destination.unlink()

        raise

    downloaded_size = destination.stat().st_size

    if downloaded_size != expected_size:
        destination.unlink()

        raise IOError(
            f"The downloaded copy of {filename} is incomplete. "
            f"Expected {expected_size} bytes but received "
            f"{downloaded_size} bytes."
        )

    print(f"Download completed: {filename}")
    print(f"Saved to: {destination}")


def download_required_files():
    """
    Download the two files required by the clustering application.
    """
    DATA_DIRECTORY.mkdir(exist_ok=True)

    print("Amazon Last-Mile Dataset Downloader")
    print("-----------------------------------")
    print("Connecting anonymously to Amazon S3...")

    s3_client = create_public_s3_client()

    for filename, object_key in FILES_TO_DOWNLOAD.items():
        download_file(
            s3_client=s3_client,
            filename=filename,
            object_key=object_key,
        )

    print("\nAll required files were downloaded successfully.")
    print(f"Data folder: {DATA_DIRECTORY}")

    print("\nFiles ready for the K-Means application:")

    for filename in FILES_TO_DOWNLOAD:
        file_path = DATA_DIRECTORY / filename
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        print(
            f"  {filename}: {file_size_mb:.2f} MB"
        )


if __name__ == "__main__":
    try:
        download_required_files()

    except Exception as error:
        print("\nThe download did not complete.")
        print(f"Reason: {error}")

        print(
            "\nRead the reason shown above. "
            "A 404 error means the S3 file path was not found. "
            "A connection error may indicate an internet, firewall, "
            "VPN, or antivirus problem."
        )

        raise