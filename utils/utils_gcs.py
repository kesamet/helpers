"""
Data loading helper functions for GCS
"""
from io import BytesIO
from typing import List

import numpy as np
from google.cloud import storage


def gcs_list_blobs(
    bucket_name: str,
    prefix: str,
    project_name: str = None,
) -> List[str]:
    """List blob paths in prefix folder.

    project_name (str, optional): the project which the client
        acts on behalf of. Will be passed when creating a topic.
        If not passed, falls back to the default inferred from
        the environment.
    """
    client = storage.Client(project_name)
    return [
        blob.name for blob in client.list_blobs(bucket_name, prefix=prefix)
    ]


def gcs_get_blob(
    bucket_name: str,
    blob_path: str,
    project_name: str = None,
) -> str:
    """Get blob.

    project_name (str, optional): the project which the client
        acts on behalf of. Will be passed when creating a topic.
        If not passed, falls back to the default inferred from
        the environment.
    """
    client = storage.Client(project_name)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_path)
    return blob.download_as_string()


def gcs_txt_read(
    bucket_name: str,
    blob_path: str,
    project_name: str = None,
) -> str:
    """Load text file from GCS bucket.

    project_name (str, optional): the project which the client
        acts on behalf of. Will be passed when creating a topic.
        If not passed, falls back to the default inferred from
        the environment.
    """
    body = gcs_get_blob(bucket_name, blob_path, project_name=project_name)
    txt = body.decode("utf-8")
    return txt


def gcs_imread(
    bucket_name: str,
    blob_path: str,
    project_name: str = None,
) -> np.array:
    """Load image from GCS bucket as array in RGB.

    project_name (str, optional): the project which the client
        acts on behalf of. Will be passed when creating a topic.
        If not passed, falls back to the default inferred from
        the environment.
    """
    import cv2

    body = gcs_get_blob(bucket_name, blob_path, project_name=project_name)
    img_arr = np.asarray(bytearray(body), dtype=np.uint8)
    image = cv2.imdecode(img_arr, cv2.IMREAD_ANYCOLOR)  # BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB
    return image


def gcs_buffer_read(
    bucket_name: str,
    blob_path: str,
    project_name: str = None,
) -> BytesIO:
    """Load file from GCS bucket as buffer.

    project_name (str, optional): the project which the client
        acts on behalf of. Will be passed when creating a topic.
        If not passed, falls back to the default inferred from
        the environment.
    """
    body = gcs_get_blob(bucket_name, blob_path, project_name=project_name)
    buffered = BytesIO(body)
    return buffered


def gcs_write(
    origin_filename: str,
    bucket_name: str,
    dest_blob_path: str,
    project_name: str = None
) -> None:
    """Save file to GCS bucket.

    project_name (str, optional): the project which the client
        acts on behalf of. Will be passed when creating a topic.
        If not passed, falls back to the default inferred from
        the environment.
    """
    client = storage.Client(project_name)
    bucket = client.get_bucket(bucket_name)
    bucket.blob(dest_blob_path).upload_from_filename(origin_filename)
