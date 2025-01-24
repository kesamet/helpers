import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel, huggingface_llm_image_uri


def delete_endpoint_and_config(endpoint_name: str):
    """Delete existing endpoint and config if they exist."""
    sagemaker_client = boto3.client("sagemaker")

    # Delete endpoint
    try:
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"Deleting endpoint {endpoint_name}")
        waiter = sagemaker_client.get_waiter("endpoint_deleted")
        waiter.wait(EndpointName=endpoint_name)
    except sagemaker_client.exceptions.ClientError as e:
        if "Could not find endpoint" in str(e):
            print(f"Endpoint {endpoint_name} does not exist")
        else:
            raise e

    # Delete endpoint config
    try:
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print(f"Deleting endpoint configuration {endpoint_name}")
    except sagemaker_client.exceptions.ClientError as e:
        if "Could not find endpoint configuration" in str(e):
            print(f"Endpoint configuration {endpoint_name} does not exist")
        else:
            raise e


region_name = "..."

runtime_client = boto3.client("sagemaker-runtime", region_name=region_name)
role = sagemaker.get_execution_role()

# LLM
endpoint_name = "..."
instance_type = "ml.g5.2xlarge"

env = {
    "HF_MODEL_ID": "Qwen/Qwen2.5-3B-instruct",
    "OPTION_ROLLING_BATCH": "vllm",
    "TENSOR_PARALLEL_DEGREE": "max",
    "OPTION_MAX_ROLLING_BATCH_SIZE": "2",
    "OPTION_DTYPE": "fp16",
}

# available frameworks: "djl-lmi" (for vllm, lmi-dist), "djl-tensorrtllm" (for tensorrt-llm),
# "djl-neuronx" (for transformers neuronx)
container_uri = sagemaker.image_uris.retrieve(
    framework="djl-lmi", version="0.30.0", region=region_name
)
model = sagemaker.Model(
    image_uri=container_uri,
    role=role,
    env=env,
)
predictor = model.deploy(
    instance_type=instance_type,
    initial_instance_count=1,
    endpoint_name=endpoint_name,
)

# embeddings
endpoint_name = "..."
instance_type = "ml.m5.xlarge"

env = {
    "HF_MODEL_ID": "BAAI/bge-m3",
    "HF_TASK": "feature-extraction",
    "SAGEMAKER_MODEL_SERVER_TIMEOUT": "3600",
    "TS_MAX_RESPONSE_SIZE": "2000000000",
    "TS_MAX_REQUEST_SIZE": "2000000000",
    "MMS_MAX_RESPONSE_SIZE": "2000000000",
    "MMS_MAX_REQUEST_SIZE": "2000000000",
}
hf_model = HuggingFaceModel(
    env=env,
    role=role,
    transformers_version="4.37.0",
    pytorch_version="2.1.0",
    py_version="py310",
)
predictor = model.deploy(
    instance_type=instance_type,
    initial_instance_count=1,
    endpoint_name=endpoint_name,
)

# use TEI
endpoint_name = "..."
instance_type = "ml.m5.xlarge"

env = {
    "HF_MODEL_ID": "BAAI/bge-reranker-v2-m3",
    "MAX_BATCH_TOKENS": "8192",
    "MAX_CONCURRENT_REQUESTS": "1",
    "MAX_CLIENT_BATCH_SIZE": "16",
}
hf_model = HuggingFaceModel(
    image_uri=huggingface_llm_image_uri("huggingface-tei", version="1.2.3"),
    env=env,
    role=role,
)
predictor = model.deploy(
    instance_type=instance_type,
    initial_instance_count=1,
    endpoint_name=endpoint_name,
)
