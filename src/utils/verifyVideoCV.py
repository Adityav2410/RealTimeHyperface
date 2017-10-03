import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle

from keras import backend as K
from keras.optimizers import Adam,Nadam
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
from keras_frcnn import resnet_hyper as nn
import keras_frcnn.roi_helpers as roi_helpers
from keras.applications.vgg16 import VGG16
import os
from pdb import set_trace as bp
from datetime import datetime

import cv2
from pdb import set_trace as bp



bp()
cap = cv2.VideoCapture('../videos/vaibhav.MP4')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



###################################################load video#######################################################
vid_name = 'vaibhav.MP4'
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
vid_out = cv2.VideoWriter('out.mp4',fourcc, 30.0, (1920,1080), False)
rgbVideo = cv2.VideoCapture('../videos/'+vid_name)



while True:
	try:
		ret, img = rgbVideo.read()
		if ret==True:
			start_time = time.time()

			print('Elapsed time = {}'.format(time.time() - start_time))
			vid_out.write(img)
			cv2.imshow('frame',img)
			cv2.waitKey(1)
			# filename = 'image71306.jpg'
			
	except:
		print("Error in processing the image")
		continue

