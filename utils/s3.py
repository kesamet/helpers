"""
S3 utility functions
"""

from io import BytesIO
from pathlib import Path
from typing import List, Union

import boto3
import botocore
import numpy as np


def list_blobs(bucket_name: str, prefix: str, delimiter: str = "/") -> List[str]:
    """List blob paths in prefix folder."""
    s3 = boto3.resource("s3")
    s3_bucket = s3.Bucket(bucket_name)
    return [f.key for f in s3_bucket.objects.filter(Prefix=prefix, Delimiter=delimiter)]


def list_dirs(bucket_name: str, prefix: str, delimiter: str = "/") -> List[str]:
    """List dirs in prefix folder."""
    s3_client = boto3.client("s3")
    result = s3_client.list_objects(Bucket=bucket_name, Prefix=prefix, Delimiter=delimiter)
    if result.get("CommonPrefixes") is None:
        return []
    return [r.get("Prefix") for r in result.get("CommonPrefixes")]


def download_file(
    bucket_name: str,
    origin_blob_path: Union[str, Path],
    dest_filename: Union[str, Path],
) -> None:
    """Download blob from S3 bucket."""
    s3 = boto3.resource("s3")

    try:
        s3.Bucket(bucket_name).download_file(str(origin_blob_path), str(dest_filename))
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print("The object does not exist.")
        else:
            raise


def upload_file(
    origin_file_path: Union[str, Path],
    bucket_name: str,
    dest_key: Union[str, Path],
) -> None:
    """Upload file to S3 bucket."""
    s3 = boto3.resource("s3")

    try:
        s3.Object(bucket_name, str(dest_key)).upload_file(str(origin_file_path))
    except botocore.exceptions.ClientError:
        raise


def isfile(bucket_name: str, blob_path: str) -> bool:
    """Check if file exists in S3 bucket."""
    s3_client = boto3.client("s3")
    try:
        s3_client.head_object(Bucket=bucket_name, Key=blob_path)
        return True
    except botocore.exceptions.ClientError:
        return False


def _get_blob(bucket_name: str, blob_path: str) -> bytes:
    """Get blob."""
    s3 = boto3.resource("s3")
    obj = s3.Object(bucket_name, blob_path)
    return obj.get()["Body"].read()


def load_blob(bucket_name: str, blob_path: str, as_buffer: bool = False) -> str | bytes:
    """Load file from S3 bucket."""
    body = _get_blob(bucket_name, blob_path)
    if as_buffer:
        return BytesIO(body)
    return body.decode("utf-8")


def save_blob(data: bytes, bucket_name: str, blob_path: str) -> None:
    """Save data in S3."""
    s3_client = boto3.client("s3")
    try:
        s3_client.put_object(Body=data, Bucket=bucket_name, Key=blob_path)
    except Exception as e:
        print(f"Error uploading file: {e}")


def delete_blob(bucket_name: str, blob_path: str) -> None:
    """Delete blob from S3."""
    s3 = boto3.resource("s3")
    s3.Object(bucket_name, blob_path).delete()


def copy_blob(
    source_bucket: str,
    source_blobpath: str,
    dest_bucket: str,
    dest_blobpath: str,
) -> None:
    """Copy blob from source_bucket to dest_bucket."""
    s3 = boto3.resource("s3")

    try:
        copy_source = {
            "Bucket": source_bucket,
            "Key": source_blobpath,
        }
        s3.Bucket(dest_bucket).copy(copy_source, dest_blobpath)
    except Exception as e:
        print(f"Error copying file: {e}")


def copy_dir(
    source_bucket: str,
    source_prefix: str,
    dest_bucket: str,
    dest_prefix: str,
) -> None:
    """Copy all blobs from source_prefix to dest_prefix."""
    s3_client = boto3.client("s3")

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=source_bucket, Prefix=source_prefix)

        for page in pages:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                source_key = obj["Key"]
                dest_key = source_key.replace(source_prefix, dest_prefix, 1) if source_prefix else source_key

                s3_client.copy_object(
                    CopySource={
                        "Bucket": source_bucket,
                        "Key": source_key,
                    },
                    Bucket=dest_bucket,
                    Key=dest_key,
                )
    except Exception as e:
        print(f"Error uploading files: {e}")


def load_pil(bucket_name: str, blob_path: str):
    """Load image from S3 as PIL."""
    from PIL import Image

    pil_img = Image.open(load_blob(bucket_name, blob_path, as_buffer=True))
    return pil_img


def imread(bucket_name: str, blob_path: str) -> np.array:
    """Load image from S3 as array in BGR."""
    import cv2

    body = _get_blob(bucket_name, blob_path)
    np_img = cv2.imdecode(np.asarray(bytearray(body), dtype=np.uint8), cv2.IMREAD_ANYCOLOR)  # BGR
    return np_img


def save_image(np_img: np.ndarray, bucket_name: str, blob_path: str, quality: int = 95) -> None:
    """Save image in S3. Image is assumed to be in BGR."""
    import cv2

    success, buffer = cv2.imencode(".jpg", np_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise ValueError("Unable to encode frame as JPEG")
    
    image_bytes = BytesIO(buffer)

    s3_client = boto3.client("s3")
    try:
        s3_client.upload_fileobj(image_bytes, bucket_name, blob_path)
    except Exception as e:
        print(f"Error uploading image: {e}")
