
import random
import csv
import io
import os
import threading
import time
import logging
import sys
from sys import platform
from datetime import datetime
try: #Depende de la version de python
   from Queue import Queue, Empty
except:
   from queue import Queue, Empty

import cv2
from configs import CONFIGS, CAMERA, SERVER_URL
from datetime import datetime

#Antropometric constant values of the human head. 
#Found on wikipedia and on:
# "Head-and-Face Anthropometric Survey of U.S. Respirator Users"
#
#X-Y-Z with X pointing forward and Y on the left.
#The X-Y-Z coordinates used are like the standard
# coordinates of ROS (robotic operative system)
P3D_RIGHT_SIDE = np.float32([-100.0, -77.5, -5.0]) #0
P3D_GONION_RIGHT = np.float32([-110.0, -77.5, -85.0]) #4
P3D_MENTON = np.float32([0.0, 0.0, -122.7]) #8
P3D_GONION_LEFT = np.float32([-110.0, 77.5, -85.0]) #12
P3D_LEFT_SIDE = np.float32([-100.0, 77.5, -5.0]) #16
P3D_FRONTAL_BREADTH_RIGHT = np.float32([-20.0, -56.1, 10.0]) #17
P3D_FRONTAL_BREADTH_LEFT = np.float32([-20.0, 56.1, 10.0]) #26
P3D_SELLION = np.float32([0.0, 0.0, 0.0]) #27
P3D_NOSE = np.float32([21.1, 0.0, -48.0]) #30
P3D_SUB_NOSE = np.float32([5.0, 0.0, -52.0]) #33
P3D_RIGHT_EYE = np.float32([-20.0, -65.5,-5.0]) #36
P3D_RIGHT_TEAR = np.float32([-10.0, -40.5,-5.0]) #39
P3D_LEFT_TEAR = np.float32([-10.0, 40.5,-5.0]) #42
P3D_LEFT_EYE = np.float32([-20.0, 65.5,-5.0]) #45
#P3D_LIP_RIGHT = numpy.float32([-20.0, 65.5,-5.0]) #48
#P3D_LIP_LEFT = numpy.float32([-20.0, 65.5,-5.0]) #54
P3D_STOMION = np.float32([10.0, 0.0, -75.0]) #62
TRACKED_POINTS = (0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62)

FFMPEG_BIN="/usr/bin/ffmpeg"

def yawpitchrolldecomposition(R):
     sin_x = math.sqrt(R[2,0] * R[2,0] +  R[2,1] * R[2,1])    
     validity  = sin_x < 1e-6
     if not validity:
         z1 = math.atan2(R[2,0], R[2,1])     # around z1-axis
         x = math.atan2(sin_x,  R[2,2])     # around x-axis
         z2 = math.atan2(R[0,2], -R[1,2])    # around z2-axis
     else: # gimbal lock
         z1 = 0                                         # around z1-axis
         x = math.atan2(sin_x,  R[2,2])     # around x-axis
         z2 = 0                                         # around z2-axis
     return np.array([[z1], [x], [z2]])

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.) 
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
        import openpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.) 
        sys.path.append('/home/nvidia/openpose/buildPython/python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e


FPS = CAMERA['framerate']

logger = logging.getLogger("Camera")
logger.setLevel(logging.DEBUG)


class VideoRecorder:
    """
    VideoRecorder que utiliza opencv para leer datos de camara usb.
    """

    def __init__(self, on_error):
        """
        on_error: callback
        """
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise IOError("Error al reconocer la camara USB")
        # self.set_camera_params()
        # print self.camera.get(cv2.CAP_PROP_FRAME_WIDTH), self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #self.channel = grpc.insecure_channel(SERVER_URL)
        self.record_channel = None
        # self.grpc_stub = FeatureExtractionApi_pb2_grpc.FeatureExtractionStub(channel)
        self.recording_stop = True
        self.image_queue = Queue()
        self.count = 0
        self.sent_count = 0
        self.grabbing = False
        self.on_error = on_error
        # Starting OpenPose
        params = dict()
        params["model_folder"] = "../../../models/"
        params["face"] = True
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        self.videofile = cv2.VideoWriter('video.avi',fourcc, 5.0, (800,600))
        


        logger.debug("Camera started")

    def set_camera_params(self):
        self.camera.set(3,1296)#width
        self.camera.set(4,972)#Height

    def capture_continuous(self, filename):
        """
        Captura frames en un loop y los encola en image_queue
        """
        logger.debug("capture continuous")
        self.count = 1
        self.grabbing = True
        self.filename = filename
 	csv_file=open(str(filename)+'_result.csv', mode='w')        
	self.resultfile = csv.writer(csv_file, delimiter=',')
        while True:
            start = time.time()
            ret, frame = self.camera.read()
            #frame=cv2.flip(frame,0)
            #bytesImg= cv2.imencode(".jpg",frame)[1].tostring()
            self.image_queue.put(frame)
            
            if self.recording_stop:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            self.count += 1
            time.sleep(max(1./FPS - (time.time() - start), 0))
        self.grabbing = False


    def generate_videos_iterator(self):
        """
        Iterator. lee frames de cola image_queue
        """
        logger.debug("generate video iterator")
        self.sent_count = 0
        while not self.recording_stop or not self.image_queue.empty() or self.grabbing:
            try:
                frame= self.image_queue.get(block=True, timeout=1)
                datum = op.Datum()		
                datum.cvInputData = frame
                resolution=frame.size()
                resolution_flags =  str(resolution.width)+"x"+ str(resolution.height)
                print("ESTUDIANTE: "+self.filename)
                self.opWrapper.emplaceAndPop([datum])

                rectangles = datum.faceRectangles
                keypoints = datum.poseKeypoints
                fk=datum.faceKeypoints
                bodyId=select_biggest_skeleton(keypoints)
                hp=headPosture(fk,bodyId)
                


                self.videofile.write(datum.cvOutputData)
                self.image_queue.task_done()
                print ("sent",self.sent_count, "of", self.count, "captured")
                self.sent_count += 1
            except Empty as ex:
                logger.error("No data in image queue")

        logger.debug("Done generating images")
        self.videofile.release()


    def select_biggest_skeleton(keypoints):
        max_id = 0;
        max_size = 0;
        for i in range(0,keypoints.shape[0]):
             rhip_y = keypoints[i, 8, 1]
             lhip_y = keypoints[i, 11, 1]
             neck_y = keypoints[i, 1, 1]
             size = 0
             if (neck_y != 0 and (rhip_y != 0 or lhip_y != 0)):
                  size = (rhip_y + lhip_y) / 2 - neck_y
                  if (size > max_size):
                      max_size = size
                      max_id = i		
        return max_id

    def headPosture(fk,bodyId):
        landmarks_2D = np.zeros((len(TRACKED_POINTS),2), dtype=np.float32)
        counter = 0
        for point in TRACKED_POINTS:
            landmarks_2D[counter] = [fk[bodyId][point][0], fk[bodyId][point][1]]
            counter += 1               
        retval, rvec, tvec = cv2.solvePnP(landmarks_3D, 
                                          landmarks_2D, 
                                          camera_matrix, camera_distortion)
        rmat = cv2.Rodrigues(rvec)[0]
        ypr = -180*yawpitchrolldecomposition(rmat)/math.pi
        ypr[1,0] = ypr[1,0]+90
        if ypr[0,0]>75 and ypr[0,0]<105:
            if ypr[1,0]>-5 and ypr[1,0]<5:
                return "center"
            else:
                if ypr[1,0]>=5:
                    return "up"
            else:
                return "down"                 
        else:
           if ypr[0,0]>=105:
              return "right"
           else:
              return "left" 

    def processVideo(videos_iterator):
        while True:
           for frame in videos_iterator:
               pass

    def start_recording(self, filename):
        """
        Empieza grabacion. Crea hilo para captura y canal grpc para envio
        """
        logger.info("Start recording")
        try:
            #self.record_channel = grpc.insecure_channel(SERVER_URL)
            #if not self.ping():
            #    raise
            #self.grpc_stub = FeatureExtractionApi_pb2_grpc.FeatureExtractionStub(self.record_channel)
            threading.Thread(target=self.capture_continuous, args=(filename, )).start()
            videos_iterator = self.generate_videos_iterator()
            worker = threading.Thread(target=self.processVideo, args=(videos_iterator))
            worker.setDaemon(True)
            worker.start()
          
            #logger.debug(response)

            #self.record_channel.close()
            self.record_channel = None
        except Exception as e:
            logger.exception("start_recording")
            logger.error("Murio grpc")
            self.on_error()

    

    def ping(self):
        """
        Verifica que exista conexion con servidor grpc
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
        Crea hilo para que inicie grabacion
        """
        filename=CONFIGS["session"]
        self.recording_stop = False
        self.image_queue = Queue()
        threading.Thread(target=self.start_recording, args=(filename, )).start()

    def stop_record(self, callback=None):
        """
        Detiene grabacion de video
        callback: se ejecuta una vez ha finalizado el envio de todos los frames a servidor grpc
        """
        
        self.recording_stop = True
        time.sleep(5)
        self.image_queue.join()
        self.channel = None
        
        CONFIGS["session"] = '0'
        if callback:
            callback()

    def get_progress(self):
        try:
            return "{} %".format(int(self.sent_count * 100.0 / self.count))
        except:
            return "0 %"
        # return "{}/{}".format(self.sent_count, self.count)

    def clean(self):
        self.camera.release()
        logger.debug("Camera released")
        # self.camera.close()

    def convert_to_mp4(self):
        filename_mp4 = self.filename.split(".")[0]+".mp4"
        logger.info("file .h264 saved.. Transforming to mp4...")
        os.system("MP4Box -fps 30 -add "+ self.filename + " " + filename_mp4)
        logger.info("File converted to mp4")

    def select_biggest_skeleton(keypoints) {
	max_id = 0;
	max_size = 0;
	for i in range(0,keypoints.getSize(0); i++) {
			float rhip_y = keypoints.at(std::vector<int>{i, 8, 1});
			float lhip_y = keypoints.at(std::vector<int>{i, 11, 1});
			float neck_y = keypoints.at(std::vector<int>{i, 1, 1});
			int size = 0;
			if (neck_y != 0 && (rhip_y != 0 || lhip_y != 0)) {
				size = (rhip_y + lhip_y) / 2 - neck_y;
				if (size > max_size) {
					max_size = size;
					max_id = i;
				}
			}			
		}
		return max_id;
	}


if __name__ == "__main__":
    vid_recorder = VideoRecorder()
    print ("Set vid recorder")
    # vid_recorder.camera.wait_recording(5)
    time.sleep(2)
    start = datetime.now()
    print("Start" , start)
    print(start)
    vid_recorder.record()
    # vid_recorder.camera.wait_recording(2)
    # vid_recorder.camera.capture("foo.jpg", use_video_port=True)
    # print ("Pic taken")
    # vid_recorder.camera.wait_recording(30)
    time.sleep(30)
    vid_recorder.stop_record()

    vid_recorder.clean()
