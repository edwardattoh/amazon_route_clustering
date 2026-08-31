import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError


BUCKET_NAME = "amazon-last-mile-challenges"
AWS_REGION = "us-west-2"
SEARCH_PREFIX = "almrrc2021/"

WANTED_FILENAMES = {
    "route_data.json",
    "package_data.json",
}


def create_public_s3_client():
    """
    Create an anonymous connection to Amazon S3.

    No AWS account, access keys, or AWS CLI are required.
    """
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        config=Config(
            signature_version=UNSIGNED,
            s3={
                "addressing_style": "path",
            },
            connect_timeout=30,
            read_timeout=120,
            retries={
                "max_attempts": 5,
                "mode": "standard",
            },
        ),
    )


def find_files():
    """
    Locate route_data.json and package_data.json in the dataset.
    """
    print("Amazon Dataset File Finder")
    print("--------------------------")
    print("Connecting anonymously to Amazon S3...")
    print("No AWS account or AWS CLI is being used.")
    print(f"Searching under: {SEARCH_PREFIX}\n")

    s3_client = create_public_s3_client()
    found_files = []

    try:
        paginator = s3_client.get_paginator("list_objects_v2")

        pages = paginator.paginate(
            Bucket=BUCKET_NAME,
            Prefix=SEARCH_PREFIX,
            PaginationConfig={
                "PageSize": 1000,
            },
        )

        number_examined = 0

        for page in pages:
            objects = page.get("Contents", [])

            for item in objects:
                number_examined += 1

                object_key = item["Key"]
                filename = object_key.rsplit("/", 1)[-1]

                if filename in WANTED_FILENAMES:
                    found_files.append(
                        {
                            "filename": filename,
                            "key": object_key,
                            "size": item["Size"],
                        }
                    )

                    print("FOUND:")
                    print(f"  Filename: {filename}")
                    print(f"  Complete key: {object_key}")
                    print(
                        "  Size: "
                        f"{item['Size'] / (1024 * 1024):.2f} MB"
                    )
                    print()

        print(f"Number of S3 objects examined: {number_examined}")

    except ClientError as error:
        details = error.response.get("Error", {})
        error_code = details.get("Code", "Unknown")
        error_message = details.get(
            "Message",
            "No message was provided.",
        )

        print("\nThe S3 search did not complete.")
        print(f"AWS error code: {error_code}")
        print(f"AWS message: {error_message}")

        print(
            "\nThis result does not mean that Python or "
            "PyCharm is broken."
        )

        return

    print("\nSearch finished.")

    if not found_files:
        print(
            "\nThe listing request worked, but the two filenames "
            "were not found under the selected prefix."
        )
        return

    print("\nExact paths found:\n")

    for item in found_files:
        print(f"{item['filename']}:")
        print(item["key"])
        print()


if __name__ == "__main__":
    find_files()