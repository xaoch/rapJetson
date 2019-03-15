
import grpc
import random
from FeatureExtractionApi_pb2 import Image
import FeatureExtractionApi_pb2_grpc
import io, os
import threading
import time
import logging
try: #Depende de version de python
    from Queue import Queue, Empty
except:
    from queue import Queue, Empty
from picamera import PiCamera
from picamera.exc import PiCameraMMALError, PiCameraError
from configs import CONFIGS, CAMERA, SERVER_URL
from datetime import datetime

logger = logging.getLogger("Camera")
logger.setLevel(logging.DEBUG)
class VideoRecorder:

    def __init__(self, on_error):
        "docstring"
        try:
            self.camera = PiCamera()
            self.set_camera_params()
            #self.camera.start_recording(self.my_stream, format="h264")
            self.recording_stop = True
            self.stream = io.BytesIO()
            self.image_queue = Queue()
            self.count = 0
            self.sent_count = 0
            self.grabbing = False
            self.on_error = on_error
            self.channel = None
            self.record_channel = None
            logger.debug("Camera and grpc started")
        except (PiCameraMMALError, PiCameraError) as error:
            self.on_error()
            raise ConnectionError("Camera not available")

    def set_camera_params(self):
        self.camera.resolution = CAMERA['resolution']
        self.camera.framerate = CAMERA['framerate']
        self.camera.brightness = CAMERA['brightness']
        self.camera.saturation = CAMERA['saturation']
        self.camera.contrast = CAMERA['contrast']
        self.camera.hflip = True
        self.camera.vflip = True
        time.sleep(5)

    def capture_continuous(self, filename):
        """
        Lee frames de camara y encola en image_queue
        """
        logger.debug("capture continuous")
        try:
            self.count = 1
            self.grabbing = True
            # Use the video-port for captures...
            for foo in self.camera.capture_continuous(self.stream, 'jpeg',use_video_port=True):
                self.stream.seek(0)
                self.image_queue.put(Image(source=self.stream.read(), file_name=filename, timestamp = str(datetime.now())))
                if self.recording_stop:
                     break
                self.stream.seek(0)
                self.stream.truncate()
                self.count += 1
        finally:
            self.stream.seek(0)
            self.stream.truncate()
            self.grabbing = False

    def generate_videos_iterator(self):
        """
        Iterator. Lee frames de cola image_queue
        """
        logger.debug("generate video iterator")
        self.sent_count = 0
        while not self.recording_stop or not self.image_queue.empty() or self.grabbing:
            try:
                yield self.image_queue.get(block=True, timeout=1)
                self.image_queue.task_done()
                # print ("sent",self.sent_count, "of", self.count, "captured")
                self.sent_count += 1
            except Empty as ex:
                logger.error("No data in image queue")
        logger.debug("Done generating images")

    def start_recording(self, filename):
        """
        Empieza captura de frames en hilo aparte y envia frames con grpc channel
        """
        logger.info("Start recording")
        try:
            self.record_channel = grpc.insecure_channel(SERVER_URL)
            if not self.ping():
                raise
            self.grpc_stub = FeatureExtractionApi_pb2_grpc.FeatureExtractionStub(self.record_channel)
            threading.Thread(target=self.capture_continuous, args=(filename, )).start()
            videos_iterator = self.generate_videos_iterator()
            response = self.grpc_stub.processVideo(videos_iterator)
            logger.debug(response)

            self.record_channel.close()
            self.record_channel = None
        except:
            logger.exception("start_recording")
            logger.error("Murio grpc")
            self.on_error()

    def ping(self):
        """
        Comprueba que exista conexion con canal grpc
        """
        if self.channel is None:
            self.channel = grpc.insecure_channel(SERVER_URL)
        try:
            grpc.channel_ready_future(self.channel).result(timeout=1)
            logger.info("Ping")
            return True
        except grpc.FutureTimeoutError as e:
            logger.error("Couldnt connect to GRPC SERVER")
            self.channel.close()
            self.channel = None
            return False

    def record(self):
        """
        Inicia hilo de grabacion
        """
        filename=CONFIGS["session"]
        self.recording_stop = False
        self.image_queue = Queue()
        threading.Thread(target=self.start_recording, args=(filename, )).start()

    def stop_record(self, callback=None):
        self.recording_stop = True
        time.sleep(5)
        self.image_queue.join()
        CONFIGS["session"] = '0'
        if callback:
            callback()

    def get_progress(self):
        try:
            return "{} %".format(int(self.sent_count * 100.0 / self.count))
        except:
            return "0 %"

    def clean(self):
        self.camera.close()
        logger.debug("Camera released")

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
