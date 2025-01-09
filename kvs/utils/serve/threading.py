"""
Detection utility functions.
"""

import time
from collections import deque
from datetime import timedelta
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

import cv2
import pytz
import numpy as np
from imutils.video import FPS

from config import CFG
from utils.serve.detecting import process_img, obj_detection
from utils.serve.kinesis_camera import CameraStream, camera_uuid
from utils.serve.general import get_execution_time


def process_all_streams(period=50, duration=200):
    streams = [ThreadedVideoStream(c, period, duration) for c in range(len(camera_uuid))]
    fps_estimator = FPS().start()
    n_frames = 3600
    n_processed = 0
    result = []
    while any(s.thread.is_alive() or len(s.frames) > 0 for s in streams):
        data = []
        for s in streams:
            take_n = min(16, len(s.frames))
            for _ in range(take_n):
                frame = s.frames.popleft()
                data.append(frame + (s.camera_id,))
            s.n_processed_frames += take_n
        if len(data) == 0:
            continue
        # Update stats
        if len(result) == 0:
            start = time.time()
        take_n = len(data)
        frames, pos_msec, round_msec, camera_ids = zip(*data)
        # Run detection
        imgs = process_img(frames)
        rects = obj_detection(imgs)
        if take_n == 1:
            rects = [rects]
        # Combine results
        for i, r in enumerate(rects):
            if len(r) == 0:
                continue
            pos_msec_arr = np.array([pos_msec[i]] * len(r)).reshape(-1, 1)
            round_msec_arr = np.array([round_msec[i]] * len(r)).reshape(-1, 1)
            camera_id_arr = np.array([camera_ids[i]] * len(r)).reshape(-1, 1)
            result.append(np.hstack((r, pos_msec_arr, round_msec_arr, camera_id_arr)))
        # 2 min video at 30 fps has 3600 frames
        if n_processed % n_frames + take_n >= n_frames:
            print(f"Scoring {n_frames} frames took: {time.time() - start:.3f}s")
            start = time.time()
        # Update stats
        n_processed += take_n
        fps_estimator._numFrames += take_n
    fps_estimator.stop()
    for s in streams:
        s.thread.join()
    info = {
        "total cameras": len(camera_uuid),
        "total processed frames": sum(s.n_frames for s in streams),
        "total frames": sum(s.n_processed_frames for s in streams),
        "elapsed time": fps_estimator.elapsed(),
        "approx FPS": fps_estimator.fps() * max(1, CFG.fps * period / 1000),
        "num_rows": len(result),
    }
    return info, np.concatenate(result) if len(result) > 0 else []


def process_threaded_stream(camera_id, period=50, duration=200, stream=False):
    """
    Process threaded stream.

    :param src: video stream
    :param camera_id:
    :param period: msecs
    :param duration: secs
    """
    if stream:
        stream = ThreadedVideoStream(camera_id, period, duration)
    else:
        stream = ThreadedVideoClip(camera_id, period, duration)

    stream.process_frame()
    if len(stream.all_rects) > 0:
        result = np.concatenate(stream.all_rects)
    else:
        result = []

    info = {
        "camera_id": camera_id,
        "total processed frames": stream.n_frames,
        "total frames": stream.n_processed_frames,
        "elapsed time": stream.fps_estimator.elapsed(),
        "approx FPS": stream.fps_estimator.fps() * max(1, CFG.fps * period / 1000),
        "num_rows": len(result),
    }
    return info, result


class ThreadedVideoClip:
    def __init__(self, src=0, period=50, duration=200):
        self.period = period
        self.duration = duration
        self.camera_id = src
        self.camera_stream = CameraStream(camera_uuid[src])
        self.frames = deque()
        self.n_frames = 0
        self.n_processed_frames = 0
        self.all_rects = []
        self.fps_estimator = FPS().start()
        self.clip_length = timedelta(minutes=2)
        # Start separate thread to read frames
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.futures = [f for f in self.create_worker()]

    def create_worker(self):
        exec_time = get_execution_time().astimezone(pytz.utc)
        end = exec_time + timedelta(seconds=self.duration)
        clip_start = exec_time
        while clip_start < end:
            clip_end = clip_start + self.clip_length
            yield self.executor.submit(self.decode_frame, exec_time, clip_start, clip_end)
            clip_start = clip_end

    def decode_frame(self, exec_time, clip_start, clip_end):
        # Download video
        start = time.time()
        payload = self.camera_stream.get_clip(clip_start, clip_end)
        fp = f"./{clip_start.isoformat()}.mp4"
        with open(fp, "wb") as f:
            chunk = payload.read()
            while chunk:
                f.write(chunk)
                chunk = payload.read()
        print(f"Downloading {fp} took: {time.time() - start:.3f}s")
        start = time.time()
        # Decode video
        n_frames = 0
        offset = (clip_start - exec_time).total_seconds() * 1000
        capture = cv2.VideoCapture(fp)
        while True:
            success, frame = capture.read()
            if not success:
                break
            # Resize frame
            frame = cv2.resize(frame, CFG.new_fsize, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pos_msec = capture.get(cv2.CAP_PROP_POS_MSEC)
            if CFG.fps * self.period < 1000:
                # Use native fps
                self.frames.append((frame, offset + pos_msec, offset + pos_msec))
                n_frames += 1
            else:
                # Sample frames every 'period' msec
                n = pos_msec // self.period
                if n > n_frames - 1:
                    self.frames.append((frame, offset + pos_msec, n * self.period))
                    n_frames = n + 1
        capture.release()
        self.n_frames += n_frames
        print(f"Decoding {n_frames} frames took: {time.time() - start:.3f}s")

    def process_frame(self):
        start = time.time()
        n_frames = self.clip_length.total_seconds() * CFG.fps
        while not all(f.done() for f in self.futures) or len(self.frames) > 0:
            take_n = min(64, len(self.frames))
            if take_n == 0:
                continue
            frames, pos_msec, round_msec = zip(*[self.frames.popleft() for _ in range(take_n)])
            # Run detection
            imgs = process_img(frames)
            rects = obj_detection(imgs)
            if take_n == 1:
                rects = [rects]
            # Combine results
            for i, r in enumerate(rects):
                if len(r) > 0:
                    pos_msec_arr = np.array([pos_msec[i]] * len(r)).reshape(-1, 1)
                    round_msec_arr = np.array([round_msec[i]] * len(r)).reshape(-1, 1)
                    camera_id_arr = np.array([self.camera_id] * len(r)).reshape(-1, 1)
                    self.all_rects.append(
                        np.hstack((r, pos_msec_arr, round_msec_arr, camera_id_arr))
                    )
            # 2 min video at 30 fps has 3600 frames
            if self.n_processed_frames % n_frames + take_n >= n_frames:
                print(f"Scoring {n_frames} frames took: {time.time() - start:.3f}s")
                start = time.time()
            # Update stats
            self.n_processed_frames += take_n
            self.fps_estimator._numFrames += take_n
        self.fps_estimator.stop()
        for f in self.futures:
            f.result()
        self.executor.shutdown()
        print(f"Scoring {n_frames} frames took: {time.time() - start:.3f}s")


class ThreadedVideoStream:
    def __init__(self, src=0, period=50, duration=200):
        self.period = period
        self.duration = duration
        self.camera_id = src
        self.camera_stream = CameraStream(camera_uuid[src])
        self.frames = deque()
        self.n_frames = 0
        self.n_processed_frames = 0
        self.pos_msec = 0
        self.all_rects = []
        self.trackers = []
        self.fps_estimator = FPS().start()
        # Start separate thread to read frames
        self.thread = Thread(target=self.decode_frame, args=())
        self.thread.daemon = True
        self.thread.start()

    def decode_frame(self):
        exec_time = get_execution_time().astimezone(pytz.utc)
        capture = cv2.VideoCapture(self.camera_stream.get_stream_url(exec_time))
        start = time.time()
        n_frames = 3600
        while self.pos_msec < self.duration * 1000:
            if not capture.isOpened():
                break
            success, frame = capture.read()
            if not success:
                end = exec_time + timedelta(milliseconds=self.pos_msec)
                print(f"End of stream: {end.isoformat()}")
                break
            self.pos_msec = capture.get(cv2.CAP_PROP_POS_MSEC)
            # Resize frame
            frame = cv2.resize(frame, CFG.new_fsize, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if CFG.fps * self.period < 1000:
                # Use native fps
                self.frames.append((frame, self.pos_msec, self.pos_msec))
                self.n_frames += 1
            else:
                # Sample frames every 'period' msec
                n = self.pos_msec // self.period
                if n > self.n_frames - 1:
                    self.frames.append((frame, self.pos_msec, n * self.period))
                    self.n_frames = n + 1
            if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) % n_frames == 0:
                print(f"Decoding {n_frames} frames took: {time.time() - start:.3f}s")
                start = time.time()
        capture.release()

    def process_frame(self):
        start = time.time()
        n_frames = 3600
        while self.thread.is_alive() or len(self.frames) > 0:
            take_n = min(64, len(self.frames))
            if take_n == 0:
                continue
            frames, pos_msec, round_msec = zip(*[self.frames.popleft() for _ in range(take_n)])
            # Run detection
            imgs = process_img(frames)
            rects = obj_detection(imgs)
            if take_n == 1:
                rects = [rects]
            # Combine results
            for i, r in enumerate(rects):
                if len(r) > 0:
                    pos_msec_arr = np.array([pos_msec[i]] * len(r)).reshape(-1, 1)
                    round_msec_arr = np.array([round_msec[i]] * len(r)).reshape(-1, 1)
                    camera_id_arr = np.array([self.camera_id] * len(r)).reshape(-1, 1)
                    self.all_rects.append(
                        np.hstack((r, pos_msec_arr, round_msec_arr, camera_id_arr))
                    )
            # 2 min video at 30 fps has 3600 frames
            if self.n_processed_frames % n_frames + take_n >= n_frames:
                print(f"Scoring {n_frames} frames took: {time.time() - start:.3f}s")
                start = time.time()
            # Update stats
            self.n_processed_frames += take_n
            self.fps_estimator._numFrames += take_n
        self.fps_estimator.stop()
        self.thread.join()
        print(f"Scoring {n_frames} frames took: {time.time() - start:.3f}s")
