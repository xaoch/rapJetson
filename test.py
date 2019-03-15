import cv2
import math
import os
import sys
import numpy as np
from sys import platform
import os
from sklearn.externals import joblib
from glob import glob
import csv
import shutil
import requests
from requests import ConnectionError
import random
import subprocess


SAVE_CSV_URL = "http://10.17.255.108/resultados/save_csv/"
SAVE_IMAGES_URL = "http://10.17.255.108/resultados/save_images/"
FFMPEG_BIN="/usr/bin/ffmpeg"


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
HAND_MID_SPINE_THRESHOLD=100
HAND_DISTANCE_THRESHOLD=80





def createFolders(path):
     try:
         shutil.rmtree(path)
     except:
         print("Cannot delete")

     try:
          os.mkdir(path)
          os.mkdir(path + "/center")
          os.mkdir(path + "/up")
          os.mkdir(path + "/down")
          os.mkdir(path + "/right")
          os.mkdir(path + "/left")
          os.mkdir(path + "/tv")
          os.mkdir(path + "/good")
          os.mkdir(path + "/bad")
     except:
          print("Directories already created")


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

def bodyPosture(keypoints, person_index, face_orientation, head_height):
    rwrist_y = keypoints[person_index][4][1]
    rwrist_x = keypoints[person_index][4][0]
    lwrist_y = keypoints[person_index][7][1]
    lwrist_x = keypoints[person_index][7][0]
    mhip_y = keypoints[person_index][8][1]
    lhip_y = keypoints[person_index][11][1]
    neck_y = keypoints[person_index][1][1]
    nose_y = keypoints[person_index][0][1]
    rshoulder_x = keypoints[person_index][2][0]
    lshoulder_x = keypoints[person_index][5][0]
    
    if rshoulder_x != 0 and lshoulder_x != 0 and lshoulder_x < rshoulder_x: #Persona de espaldas
        return "bad"
    if mhip_y == 0:
        return "NOT_DETECTED"
    if lwrist_y == 0:
        lwrist_y = rwrist_y
    if rwrist_y == 0:
        rwrist_y = lwrist_y
    if rwrist_y == 0:
        return "NOT_DETECTED"

    hand_distance_threshold = neck_y - nose_y
    spinebase = mhip_y
    spinemid = ((3*spinebase) + neck_y)/4
    normalizer = 0
    if head_height > 0:
       normalizer= head_height
    else:
       normalizer=HAND_MID_SPINE_THRESHOLD
    if lwrist_y < (spinemid - (HAND_MID_SPINE_THRESHOLD/head_height)) or rwrist_y < (spinemid - (HAND_MID_SPINE_THRESHOLD/head_height)):
        if rwrist_x != 0 and lwrist_x != 0 and abs(rwrist_x - lwrist_x) < hand_distance_threshold:
            return "bad"
        return "good"			
    return "bad"

def writeToRapCsv(csvwriter, frame, posture, face):
    if posture != "good":
        postureValue = 0
    else:
        postureValue = 1
    csvwriter.writerow([frame, face, postureValue])


def captureFacePoseImages(directory, imgDictionary, img, actualOrientation, lastOrientation, mode, x, y, width, height):
    #Mode 0 Face, Mode 1 Pose
    if (mode != 0 and mode != 1):
       return
    if actualOrientation==lastOrientation:
        return
    if actualOrientation in imgDictionary.keys():
        countImage=imgDictionary[actualOrientation]
    else:
        countImage=0
    countImage=countImage+1
    
    imgDictionary[actualOrientation]=countImage
    if (mode == 0):
         # Condiciones que debe cumplir el ROI
         # box within the image plane
         img=img[int(y):int(y)+int(height),int(x):int(x)+int(width)]
         img=cv2.resize(img, (200, 200))
    cv2.imwrite(directory+"/"+ actualOrientation+"/img" + str(countImage)+".jpg", img)



def send_results(directory,student_id):
    try:
        id = student_id
    except:
        print("student ID error")
        return
    csv_path = os.path.join(directory+"/results.csv")
    csv_file = open(csv_path,"rb");
    #print csv_path

    values = {"resultado_id":id}
    files = {"csvfile":csv_file}

    try:
        response = requests.post(SAVE_CSV_URL, data=values, files=files)
        #print "Send results:",response.status_code
        return response.status_code
    except ConnectionError as e:
        print(e)
        #print "---------Error al conectarse con el servidor--------------- "
        #print "Sent results:",400
        return 400

def send_images(directory,student_id):
    try:
        id = student_id
    except:
        print("Student Id error",student_id)
        return
    imagesPath = directory
    values = {"resultado_id":id}
    files = {}

    image_type = "img_type_"
    classifier = "classifier_"
    filename = "filename_"
    fileString = "img_"
    count = 0

    for actualfile in glob(imagesPath+"/*.jpg"):

        files["{}{:d}".format(fileString,count)] = open(actualfile, "rb")

        actualfile = actualfile.split('\\')
        actualFileName = actualfile[-1]

        actual_classifier = actualFileName.split('_')[-1].split('.')[0]
        if(actual_classifier=="good" or actual_classifier=="bad" ):
            values["{}{:d}".format(image_type,count)] = "p"
        else:
            values["{}{:d}".format(image_type,count)] = "m"

        values["{}{:d}".format(classifier,count)] = actual_classifier
        values["{}{:d}".format(filename,count)] = actualFileName

        count += 1

    values["num_images"] = count
    try:
        response = requests.post(SAVE_IMAGES_URL, data=values, files=files)
        print("Sent images:",response.status_code)
    except ConnectionError as e:
        #print "---------Error al conectarse con el servidor--------------- "
        #print "Sent images:",400
        print(e)
        return 400

    for key,value in  files.items(): value.close()
    #for actualfile in glob(imagesPath+"/*.jpg"):os.remove(actualfile)
    #print os.path.join(imagesPath,"video.avi")
    # Transforma y envia el video en mp4
    command = [FFMPEG_BIN,'-i',os.path.join(imagesPath,"video.avi"),os.path.join(imagesPath,"video.mp4")]
    FNULL = open(os.devnull, 'w')
    print ("Converting to mp4..",)
    join_process = subprocess.Popen(command, stdout=FNULL, stderr=subprocess.STDOUT, bufsize=10**8)    
    join_process.communicate()
    FNULL.close()
    print("Done")
    #subprocess.call(["pscp", "-pw", "Uh71og0saM", os.path.join(imagesPath,"video.mp4"),"root@200.10.150.110:/opt/feedback-estudiantes/media/estudiantes/"+str(id)+"/video"])
    subprocess.call(["scp", "-pw", "H8bmbnear", os.path.join(imagesPath,"video.mp4"),"root@10.17.255.108:/home/rap/RAP/rap_v2/static/resultados/"+str(id)+"/video"])
    response2 = requests.get("http://10.17.255.108/resultados/process_video/?resultado_id="+str(id))
    print ("Process media: ", response2.status_code)
        #print video_status
    #os.remove(os.path.join(imagesPath,"video.mp4"))
    return response.status_code


def selectRandomImages(directory,imgDictionary, maxImages):		
    path = directory
    for pose in imgDictionary.keys():
        value = imgDictionary[pose]
        if value>maxImages:
             randomValues=random.sample(range(1, value+1), maxImages)
        else:
             randomValues=range(1,value+1)
        for number in randomValues:
             img = cv2.imread(directory+"/"+pose+"/img"+ str(number) + ".jpg")
             cv2.imwrite(directory+ "/img" +str(number) +"_"+pose+ ".jpg", img)




params = dict()
params["model_folder"] = "../../../models/"
params["face"] = True
#params["net_resolution"]="96x96"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

frameCounter=0
directory="results" 
createFolders(directory)
csvfilename=directory+"/results.csv"
csvfile=open(csvfilename, mode='w')
csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
csvwriter.writerow(["frame","looks","positions"])


imgDictionary={}
lastBodyPosture="none"
lastHeadPosture="none"



camera = cv2.VideoCapture(0)
if not camera.isOpened():
     raise IOError("Error al reconocer la camara USB")
fourcc = cv2.VideoWriter_fourcc(*'H264')
videoFile = cv2.VideoWriter(directory+'/video.avi',fourcc, 5.0, (int(camera.get(3)),int(camera.get(4))))

while(True):
     ret, frame = camera.read()
     frameCounter=frameCounter+1
     cam_w = int(camera.get(3))
     cam_h = int(camera.get(4))
     c_x = cam_w / 2
     c_y = cam_h / 2
     f_x = c_x / np.tan(60/2 * np.pi / 180)
     f_y = f_x

     camera_matrix = np.float32([[f_x, 0.0, c_x],
                                   [0.0, f_y, c_y], 
                                   [0.0, 0.0, 1.0] ])

     camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

     landmarks_3D = np.float32([P3D_RIGHT_SIDE,
                                  P3D_GONION_RIGHT,
                                  P3D_MENTON,
                                  P3D_GONION_LEFT,
                                  P3D_LEFT_SIDE,
                                  P3D_FRONTAL_BREADTH_RIGHT,
                                  P3D_FRONTAL_BREADTH_LEFT,
                                  P3D_SELLION,
                                  P3D_NOSE,
                                  P3D_SUB_NOSE,
                                  P3D_RIGHT_EYE,
                                  P3D_RIGHT_TEAR,
                                  P3D_LEFT_TEAR,
                                  P3D_LEFT_EYE,
                                  P3D_STOMION])


     datum = op.Datum()		
     datum.cvInputData = frame
     #resolution=frame.size
     #resolution_flags =  str(resolution.width)+"x"+ str(resolution.height)
     #print("ESTUDIANTE: "+str(1))
     opWrapper.emplaceAndPop([datum])

     rectangles = datum.faceRectangles
     fk = datum.faceKeypoints
     keypoints = datum.poseKeypoints
     if not(keypoints is None) and len(rectangles)>0 and keypoints.shape[0]>0:
          bodyId=select_biggest_skeleton(keypoints)
          head_height=rectangles[bodyId].height
          if (len(rectangles) > 0 and rectangles[bodyId].y>0):
               hp=headPosture(fk,bodyId)
               bp=bodyPosture(keypoints,bodyId,hp,head_height)
               writeToRapCsv(csvwriter, frameCounter, bp, hp)
               captureFacePoseImages(directory,imgDictionary, frame, hp, lastHeadPosture, 0, rectangles[bodyId].x, rectangles[bodyId].y, rectangles[bodyId].width, rectangles[bodyId].height)
               captureFacePoseImages(directory,imgDictionary, frame, bp, lastBodyPosture, 1, rectangles[bodyId].x, rectangles[bodyId].y, rectangles[bodyId].width, rectangles[bodyId].height)
               lastHeadPosture=hp
               lastBodyPosture=bp
               print(hp,bp)
     videoFile.write(frame)
     cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", datum.cvOutputData)
     key = cv2.waitKey(15)
     if key == 27: break
     

csvfile.close()
videoFile.release()
selectRandomImages(directory,imgDictionary,3)
send_results(directory,1)
send_images(directory,1)









def calcAngle(p1_x, p1_y, p2_x, p2_y):
     if (p1_x == 0 or p1_y == 0 or p2_x == 0 or p2_y == 0):
          return 1000
     return np.rad2deg(np.arctan2(p1_y - p2_y, p1_x - p2_x))

def distance(p1_x, p1_y, p2_x, p2_y, dist):
     if (p1_x == 0 or p2_x == 0 or dist == 0):
          return -1
     return (pow(p1_x - p2_x, 2) + pow(p1_y - p2_y, 2)) / dist

#classifier = None#joblib.load("D:/RAP_Openpose/tree_model_full.pkl")
#scaler = None#joblib.load("D:/RAP_OPenpose/tree_scaler_full.pkl")
#class_names = ['center', 'down', 'left', 'right', 'tv', 'up']
#LocalPath = os.path.dirname(os.path.abspath(__file__))

#load_predictor("face_scaler.pkl","face_model.pkl")

def load_predictor(scaler_pkl,classifier_pkl):
    global classifier, scaler
    print("loading classifier ",classifier_pkl,"...")
    classifier = joblib.load(classifier_pkl)
    print("Classifier loaded")
    print("loading scaler ",scaler_pkl,"...")
    scaler = joblib.load(scaler_pkl)
    print("Scaler loaded")
    return True

def predict(features):
    features_to_scale = np.array(features[:-2], dtype=np.float64)
    try:
        scaled = scaler.transform(features_to_scale.reshape(1,-1))
        scaled_features = np.append(scaled,features[-2:])
        pred = classifier.predict(scaled_features.reshape(1,-1))
        return class_names[pred[0]]
    except ValueError as e:
        print(features)
        return "none"

#nose_x = keypoints[bodyId][0][0]
#               nose_y = keypoints[bodyId][0][1]
#               leye_x = keypoints[bodyId][15][0]
#               leye_y = keypoints[bodyId][15][1]
#               reye_x = keypoints[bodyId][14][0]
#               reye_y = keypoints[bodyId][14][1]
#               lear_x = keypoints[bodyId][17][0]
#               lear_y = keypoints[bodyId][17][1]
#               rear_x = keypoints[bodyId][16][0]
#               rear_y = keypoints[bodyId][16][1]
#               neck_x = keypoints[bodyId][1][0]
#               neck_y = keypoints[bodyId][1][1]
#               head_height=rectangles[bodyId].y

#               v1=calcAngle(leye_x, leye_y, lear_x, lear_y)
#               v2=calcAngle(reye_x, reye_y, rear_x, rear_y)
#               v3=calcAngle(leye_x, leye_y, nose_x, nose_y)
#               v4=calcAngle(reye_x, reye_y, nose_x, nose_y)
#               v5=calcAngle(lear_x, lear_y, nose_x, nose_y)
#               v6=calcAngle(rear_x, rear_y, nose_x, nose_y)
#               v7=distance(leye_x, leye_y, lear_x, lear_y, head_height)
#               v8=distance(leye_x, leye_y, reye_x, reye_y, head_height)
#               v9=distance(leye_x, leye_y, nose_x, nose_y, head_height)
#               v10=distance(reye_x, reye_y, rear_x, rear_y, head_height)
 #              v11=distance(reye_x, reye_y, nose_x, nose_y, head_height)
#               v12=(nose_y - neck_y) / head_height
 #              v13=neck_x
#               if lear_x == 0:
#                    v14 = 0
#               else:
#                    v14 = 1
#               if rear_x == 0:
#                    v15=0
#               else:
#                    v15=1

#               vector=np.array([v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15], dtype=np.float64)
