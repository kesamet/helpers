from utils.serve.kinesis_camera import CameraStream, camera_uuid

devices = [CameraStream(c) for c in camera_uuid]
