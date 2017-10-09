#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import cv2
sys.path.append('/home/nvidia/softwares/caffe3.3/python')

import numpy as np


import dlib
import openface
import caffe

###
import detection
import dlibapi



### Loading Detection Models
def loadcaffemodel():
	caffe.set_mode_gpu()
	PNet = caffe.Net("detmodels/det1.prototxt", "detmodels/det1.caffemodel", caffe.TEST)
	RNet = caffe.Net("detmodels/det2.prototxt", "detmodels/det2.caffemodel", caffe.TEST)
	ONet = caffe.Net("detmodels/det3.prototxt", "detmodels/det3.caffemodel", caffe.TEST)
	return PNet, RNet, ONet

### load detection model - action
PNet, RNet, ONet = loadcaffemodel()



### Loading Landmark & Recognition Models
pose_predictor = dlib.shape_predictor("recmodels/shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("recmodels/dlib_face_recognition_resnet_model_v1.dat")
face_aligner = openface.AlignDlib("recmodels/shape_predictor_68_face_landmarks.dat")


#### Camera configuration

mirror = False
equalize = False
resize = False
video_capture = cv2.VideoCapture(0)
print video_capture.get(3)
print video_capture.get(4)
video_capture.set(3,1920)
video_capture.set(4,1080)
print video_capture.get(3)
print video_capture.get(4)

ID = 'Majid'#FROM USER TODO


## Face Detection Configuration
minsize = 40
threshold = [0.7, 0.8, 0.8]
factor = 0.509
tolerance=0.6


################## Train
# Initialize some variables


face_locations = []
face_encodings = []
total_face_encodings = []
face_names = []
index_sample = 0
index_aligned_sample = 0

while True:
    # Grab a single frame of video
    ret, frame= video_capture.read()
    
    # Preprocessing 
    if resize:
    	# Resize frame of video to 1/4 size for faster face recognition processing
    	small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    if mirror: 
    	img = cv2.flip(img, 1) 
    if equalize:
        cv2.equalizeHist(img, img)
    	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    

    # Face Detection TODO
    face_locs,scores  = detection.detect_face(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB), minsize, PNet, RNet, ONet, threshold, False, factor)

    # TODO decide whether it is a fake face or no


    # If the face it not fake, proceed to recognize it
    if len(face_locs):
        # Extracting encoding of faces
	face_encodings, raw_landmarks, alignedFaces = dlibapi.face_encodings_known_location(frame, pose_predictor, face_encoder, face_aligner, np.roll(face_locs,-1,axis=1))
	total_face_encodings.append(face_encodings)

  	for alignedFace in alignedFaces:

    		cv2.imwrite("./database/" + ID + "/aligned/" + "%05d.jpg"%index_aligned_sample, frame) #(top, bottom), (left, right)
		index_aligned_sample += 1


    	for face_loc in face_locs:
		# make sure face locations are integer values
            	face_loc=face_loc.astype(int)

            	# Draw a box around the face with specific color
		color = (255, 0, 0)
    		cv2.imwrite("./database/" + ID + "/raw/" + "%05d.jpg"%index_sample , frame[face_loc[1]:face_loc[3], 
      			                                                        face_loc[0]:face_loc[2]]) #(top, bottom), (left, right)
		index_sample += 1
		cv2.rectangle(frame, (face_loc[0], face_loc[1]), 
      			             (face_loc[2], face_loc[3]), color, 2) #(left, top), (right, bottom)

    # Display the resulting image
    out= cv2.resize(frame,(640,480))
    cv2.imshow('Video', out)


    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()




#if __init__='__main__':

