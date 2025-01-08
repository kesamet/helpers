import os

from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.ops as ops
import nvidia.dali.types as types


class VideoReaderPipeline(Pipeline):
    def __init__(self, batch_size, sequence_length, num_threads, device_id, files, crop_size):
        super(VideoReaderPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.reader = ops.VideoReaderResize(
            device="gpu",
            filenames=files,
            sequence_length=sequence_length,
            normalized=False,
            random_shuffle=False,
            image_type=types.RGB,
            dtype=types.FLOAT,
            initial_fill=16,
            pad_last_batch=True,
            skip_vfr_check=True,
            size=crop_size,
        )
        # self.crop = ops.Crop(device="gpu", crop=crop_size, dtype=types.FLOAT)
        # self.uniform = ops.Uniform(range=(0.0, 1.0))
        self.transpose = ops.Transpose(device="gpu", perm=[3, 0, 1, 2])

    def define_graph(self):
        frame = self.reader(name="Reader")
        # cropped = self.crop(frame, crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
        output = self.transpose(frame)
        return output


class DALILoader():
    def __init__(self, batch_size, file_root, sequence_length, crop_size):
        container_files = os.listdir(file_root)
        container_files = [file_root + '/' + f for f in container_files]
        self.pipeline = VideoReaderPipeline(
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_threads=2,
            device_id=0,
            files=container_files,
            crop_size=crop_size,
        )
        self.pipeline.build()
        self.epoch_size = self.pipeline.epoch_size("Reader")
        self.dali_iterator = DALIGenericIterator(
            self.pipeline,
            ["data"],
            reader_name="Reader",
            auto_reset=True,
        )

    def __len__(self):
        return int(self.epoch_size)

    def __iter__(self):
        return self.dali_iterator.__iter__()
