"""
S3 utility functions
"""
from io import BytesIO
from pathlib import Path
from typing import List, Union

import boto3
import botocore
import numpy as np


def s3_list_blobs(bucket_name: str, prefix: str) -> List[str]:
    """List blob paths in prefix folder."""
    s3 = boto3.resource("s3")
    s3_bucket = s3.Bucket(bucket_name)
    return [f.key for f in s3_bucket.objects.filter(Prefix=prefix).all()]


def s3_download_file(
    bucket_name: str,
    origin_blob_path: Union[str, Path],
    dest_filename: Union[str, Path],
) -> None:
    """Download blob from S3 bucket."""
    s3 = boto3.resource("s3")

    try:
        s3.Bucket(bucket_name).download_file(
            str(origin_blob_path), str(dest_filename)
        )
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print("The object does not exist.")
        else:
            raise


def s3_upload_file(
    origin_file_path: Union[str, Path],
    bucket_name: str,
    dest_key: Union[str, Path],
) -> None:
    """Upload file to S3 bucket."""
    s3 = boto3.resource("s3")

    try:
        s3.Object(bucket_name, str(dest_key)).upload_file(
            str(origin_file_path)
        )
    except botocore.exceptions.ClientError:
        raise


def s3_get_blob(bucket_name: str, blob_path: str) -> bytes:
    """Get blob."""
    s3 = boto3.resource("s3")
    obj = s3.Object(bucket_name, blob_path)
    return obj.get()["Body"].read()


def s3_read(bucket_name: str, blob_path: str) -> str:
    """Load file from S3 bucket as str."""
    body = s3_get_blob(bucket_name, blob_path)
    return body.decode("utf-8")


def s3_imread(bucket_name: str, blob_path: str) -> np.array:
    """Load image from S3 as array in BGR."""
    import cv2

    body = s3_get_blob(bucket_name, blob_path)
    img_arr = np.asarray(bytearray(body), dtype=np.uint8)
    image = cv2.imdecode(img_arr, cv2.IMREAD_ANYCOLOR)  # BGR
    return image


def s3_buffer_read(bucket_name: str, blob_path: str) -> BytesIO:
    """Load file from S3 bucket as buffer."""
    body = s3_get_blob(bucket_name, blob_path)
    buffered = BytesIO(body)
    return buffered


def s3_delete_blob(bucket_name: str, blob_path: str) -> None:
    """Delete blob from S3."""
    s3 = boto3.resource("s3")
    s3.Object(bucket_name, blob_path).delete()


def s3_copy_blob(
    source_bucket: str,
    source_blobpath: str,
    dest_bucket: str,
    dest_blobpath: str,
) -> None:
    """Copy blob from source_bucket to dest_bucket."""
    s3 = boto3.resource("s3")
    copy_source = {
      "Bucket": source_bucket,
      "Key": source_blobpath,
    }
    bucket = s3.Bucket(dest_bucket)
    bucket.copy(copy_source, dest_blobpath)


def s3_isfile(bucket_name: str, blob_path: str) -> bool:
    """Check if file exists in S3 bucket."""
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=bucket_name, Key=blob_path)
        return True
    except botocore.exceptions.ClientError:
        return False
"""
S3 utility functions
"""
from io import BytesIO
from pathlib import Path
from typing import List, Union

import boto3
import botocore
import numpy as np


def s3_list_blobs(bucket_name: str, prefix: str) -> List[str]:
    """List blob paths in prefix folder."""
    s3 = boto3.resource("s3")
    s3_bucket = s3.Bucket(bucket_name)
    return [f.key for f in s3_bucket.objects.filter(Prefix=prefix).all()]


def s3_download_file(
    bucket_name: str,
    origin_blob_path: Union[str, Path],
    dest_filename: Union[str, Path],
) -> None:
    """Download blob from S3 bucket."""
    s3 = boto3.resource("s3")

    try:
        s3.Bucket(bucket_name).download_file(
            str(origin_blob_path), str(dest_filename)
        )
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print("The object does not exist.")
        else:
            raise


def s3_upload_file(
    origin_file_path: Union[str, Path],
    bucket_name: str,
    dest_key: Union[str, Path],
) -> None:
    """Upload file to S3 bucket."""
    s3 = boto3.resource("s3")

    try:
        s3.Object(bucket_name, str(dest_key)).upload_file(
            str(origin_file_path)
        )
    except botocore.exceptions.ClientError:
        raise


def s3_get_blob(bucket_name: str, blob_path: str) -> bytes:
    """Get blob."""
    s3 = boto3.resource("s3")
    obj = s3.Object(bucket_name, blob_path)
    return obj.get()["Body"].read()


def s3_read(bucket_name: str, blob_path: str) -> str:
    """Load file from S3 bucket as str."""
    body = s3_get_blob(bucket_name, blob_path)
    return body.decode("utf-8")


def s3_imread(bucket_name: str, blob_path: str) -> np.array:
    """Load image from S3 as array in BGR."""
    import cv2

    body = s3_get_blob(bucket_name, blob_path)
    img_arr = np.asarray(bytearray(body), dtype=np.uint8)
    image = cv2.imdecode(img_arr, cv2.IMREAD_ANYCOLOR)  # BGR
    return image


def s3_buffer_read(bucket_name: str, blob_path: str) -> BytesIO:
    """Load file from S3 bucket as buffer."""
    body = s3_get_blob(bucket_name, blob_path)
    buffered = BytesIO(body)
    return buffered


def s3_delete_blob(bucket_name: str, blob_path: str) -> None:
    """Delete blob from S3."""
    s3 = boto3.resource("s3")
    s3.Object(bucket_name, blob_path).delete()


def s3_copy_blob(
    source_bucket: str,
    source_blobpath: str,
    dest_bucket: str,
    dest_blobpath: str,
) -> None:
    """Copy blob from source_bucket to dest_bucket."""
    s3 = boto3.resource("s3")
    copy_source = {
      "Bucket": source_bucket,
      "Key": source_blobpath,
    }
    bucket = s3.Bucket(dest_bucket)
    bucket.copy(copy_source, dest_blobpath)


def s3_isfile(bucket_name: str, blob_path: str) -> bool:
    """Check if file exists in S3 bucket."""
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=bucket_name, Key=blob_path)
        return True
    except botocore.exceptions.ClientError:
        return False

