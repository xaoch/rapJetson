
import grpc
import random
from FeatureExtractionApi_pb2 import Image
import FeatureExtractionApi_pb2_grpc
import io, os
import threading
import time
import logging
from Queue import Queue, Empty
from picamera import PiCamera
from picamera.exc import PiCameraMMALError, PiCameraError
from configs import CONFIGS, CAMERA
from datetime import datetime

logger = logging.getLogger("Camera")
logger.setLevel(logging.DEBUG)
class VideoRecorder:

    def __init__(self):
        "docstring"
        try:
            self.camera = PiCamera()
            self.set_camera_params()
            # self.camera.start_preview()
            #self.camera.start_recording(self.my_stream, format="h264")
            channel = grpc.insecure_channel('200.126.23.95:50052')
            self.grpc_stub = FeatureExtractionApi_pb2_grpc.FeatureExtractionStub(channel)
            self.recording_stop = True
            self.stream = io.BytesIO()
            logger.debug("Camera and grpc started")
        except (PiCameraMMALError, PiCameraError) as error:
            raise ConnectionError("Camera not available")
            print (error)

    def set_camera_params(self):
        self.camera.resolution = CAMERA['resolution']
        self.camera.framerate = CAMERA['framerate']
        self.camera.brightness = CAMERA['brightness']
        self.camera.saturation = CAMERA['saturation']
        self.camera.contrast = CAMERA['contrast']
        self.camera.hflip = True
        self.camera.vflip = True
        time.sleep(5)

    def generate_videos_iterator(self, filename):
        try:
            count = 1
            # Use the video-port for captures...
            for foo in self.camera.capture_continuous(self.stream, 'jpeg',use_video_port=True):
                print("New frame: #",count)
                self.stream.seek(0)
                yield Image(source=self.stream.read(), file_name=filename, timestamp = str(datetime.now()))
                if self.recording_stop:
                     break
                self.stream.seek(0)
                self.stream.truncate()
                count += 1
        finally:
            self.stream.seek(0)
            self.stream.truncate()

    def start_recording(self, filename):
        videos_iterator = self.generate_videos_iterator(filename)
        response = self.grpc_stub.processVideo(videos_iterator)
        print (response)

    def record(self):
        filename=CONFIGS["session"]
        self.recording_stop = False
        threading.Thread(target=self.start_recording, args=(filename, )).start()

    def stop_record(self):
        self.recording_stop = True

    def clean(self):
        self.camera.close()

    def convert_to_mp4(self):
        filename_mp4 =  self.filename.split(".")[0]+".mp4"
        print("file .h264 saved.. Transforming to mp4...")
        os.system("MP4Box -fps 30 -add "+ self.filename + " " + filename_mp4)
        print("File converted to mp4")

if __name__ == "__main__":
    vid_recorder = VideoRecorder()
    print ("Set vid recorder")
    vid_recorder.camera.wait_recording(5)
    vid_recorder.record()
    vid_recorder.camera.wait_recording(2)
    vid_recorder.camera.capture("foo.jpg", use_video_port=True)
    print ("Pic taken")
    vid_recorder.camera.wait_recording(5)
    vid_recorder.stop_record()
    vid_recorder.clean()
