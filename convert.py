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

FFMPEG_BIN="/usr/bin/ffmpeg"

imagesPath="results"

command = [FFMPEG_BIN,'-i',os.path.join(imagesPath,"video.avi"),os.path.join(imagesPath,"video.mp4")]
FNULL = open(os.devnull, 'w')
print ("Converting to mp4..",)
join_process = subprocess.Popen(command, stdout=FNULL, stderr=subprocess.STDOUT, bufsize=10**8)    
join_process.communicate()
FNULL.close()
