
import os
from sklearn.externals import joblib
from glob import glob
import numpy as np
import requests
import sys
import subprocess
import thread
import win32api
import imp
import ctypes

# Load the DLL manually to ensure its handler gets
# set before our handler.
# basepath = imp.find_module('numpy')[1]
# print "numpy base path", basepath
# ctypes.CDLL(os.path.join(basepath, 'core', 'libmmd.dll'))
# ctypes.CDLL(os.path.join(basepath, 'core', 'libifcoremd.dll'))

# def handler(dwCtrlType, hook_sigint=thread.interrupt_main):
#     if dwCtrlType == 0: # CTRL_C_EVENT
#         hook_sigint()
#         return 1 # don't chain to the next handler
#     return 0 # chain to the next handler

#win32api.SetConsoleCtrlHandler(handler, 1)
"""
Reads argumnents to predict the face orientation
Arg_order:
1. angle_leye_lear
2. angle_reye_rear
3. angle_leye_nose
4. angle_reye_nose
5. angle_lear_nose
6. angle_rear_nose
7. d_leye_lear
8. d_leye_reye
9. d_leye_nose
10. d_reye_rear
11. d_reye_nose
12. d_nose_neck
13. neck_x
14. lear: 1 if present 0 otherwise
15. rear: 1 if present 0 otherwise

"""


classifier = None#joblib.load("D:/RAP_Openpose/tree_model_full.pkl")
scaler = None#joblib.load("D:/RAP_OPenpose/tree_scaler_full.pkl")
class_names = ['center', 'down', 'left', 'right', 'tv', 'up']
LocalPath = os.path.dirname(os.path.abspath(__file__))

SAVE_CSV_URL = "http://200.10.150.110/resultados/save_csv/"
SAVE_IMAGES_URL = "http://200.10.150.110/resultados/save_images/"
#SAVE_IMAGES_GRABACIONES_URL = "http://200.10.150.206/student/save_images/"
#SAVE_CSV_GRABACIONES_URL = "http://200.10.150.206/student/save_csv/"
FFMPEG_BIN = "ffmpeg.exe"
#SAVE_IMAGES_URL2 = "http://200.126.23.221/student/save_images/"
#SAVE_CSV_URL2 = "http://200.126.23.221:8000/student/save_csv/"
def load_predictor(scaler_pkl,classifier_pkl):
    global classifier, scaler
    print "loading classifier ",classifier_pkl,"...",
    classifier = joblib.load(classifier_pkl)
    print "Classifier loaded"
    print "loading scaler ",scaler_pkl,"...",
    scaler = joblib.load(scaler_pkl)
    print "Scaler loaded"
    return True

def predict(*features):
    features_to_scale = np.array(features[:-2], dtype=np.float64)
    try:
        scaled = scaler.transform(features_to_scale.reshape(1,-1))
        scaled_features = np.append(scaled,features[-2:])
        pred = classifier.predict(scaled_features.reshape(1,-1))
        return class_names[pred[0]]
    except ValueError as e:
        print features
        return "none"


def predict_proba(*features):
    features_to_scale = np.array(features[:-2], dtype=np.float64)
    try:
        scaled = scaler.transform(features_to_scale.reshape(1,-1))
        scaled_features = np.append(scaled,features[-2:])
        pred = classifier.predict(scaled_features.reshape(1,-1))[0]
        probabilities = classifier.predict_proba(scaled_features.reshape(1,-1))[0]
        prob_center = probabilities[0]
        prob_prediction = probabilities[pred]
        # if abs(prob_center - prob_prediction) < threshold:
        #     pred = 0
        #return class_names[pred[0]]
        return list(probabilities)
    except ValueError as e:
        print features
        return []

def send_images(main_folder,student_folder):
    try:
        id = int(student_folder)
    except:
        print "Error con el ID del estudiante",student_folder
        return
    imagesPath = os.path.join(LocalPath,main_folder,student_folder)
    values = {"student_id":id}
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
        print "Sent images:",response.status_code
    except ConnectionError as e:
        print "Error al conectarse al servidor"
        print "Sent images:",400

    for key,value in  files.iteritems(): value.close()
    for actualfile in glob(imagesPath+"/*.jpg"):os.remove(actualfile)
    #print os.path.join(imagesPath,"video.avi")

    # Transforma y envia el video en mp4
    #command = [FFMPEG_BIN,'-i',os.path.join(imagesPath,"video.avi"),os.path.join(imagesPath,"video.mp4")]
    #join_process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)
    #join_process.communicate()
    #subprocess.call(["pscp", "-pw", "mQAoHRGi", os.path.join(imagesPath,"video.avi"),"root@200.10.150.206:/opt/feedback-estudiantes/media/estudiantes/"+str(id)+"/video"])
    #os.remove(os.path.join(imagesPath,"video.mp4"))

def send_results(main_folder,student_folder,csv_name):
    try:
        id = int(student_folder)
    except:
        print "Error con el ID del estudiante"
        return
    csv_path = os.path.join(LocalPath,main_folder,student_folder,csv_name+".csv")
    csv_file = open(csv_path,"rb");
    #print csv_path

    values = {"student_id":id}
    files = {"csvfile":csv_file}

    response = requests.post(SAVE_CSV_URL, data=values, files=files)
    print "Send resultados:",response.status_code
