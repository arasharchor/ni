#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import cv2
sys.path.append('/home/nvidia/softwares/caffe3.3/python')
import caffe
import numpy as np

###
import detection
import testfaces
import dlibapi
import dlib
import openface
from glob import glob
import os
####
from deepgaze.face_landmark_detection import faceLandmarkDetection


####
from imutils import face_utils
from scipy.spatial import distance as dist

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 5

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]	
	
COUNTER = 0
	
### Loading Detection Models
def loadcaffemodel():
	caffe.set_mode_gpu()
	PNet = caffe.Net("detmodels/d1.prototxt", "detmodels/d1.caffemodel", caffe.TEST)
	RNet = caffe.Net("detmodels/d2.prototxt", "detmodels/d2.caffemodel", caffe.TEST)
	ONet = caffe.Net("detmodels/d3.prototxt", "detmodels/d3.caffemodel", caffe.TEST)
	return PNet, RNet, ONet


####

#### Loading Recognition Models

predictor_model = "recmodels/sh_pre.dat"

pose_predictor = dlib.shape_predictor(predictor_model)

face_encoder = dlib.face_recognition_model_v1("recmodels/fac_rec_model.dat")

face_aligner = openface.AlignDlib(predictor_model)

	

#### Camera configuration

mirror = False
equalize = False
resize= False
video_capture = cv2.VideoCapture(0)
print video_capture.get(3)
print video_capture.get(4)
video_capture.set(3,640)
video_capture.set(4,480)
print video_capture.get(3)
print video_capture.get(4)


## Compute Speed

from time import time
_tstart_stack = []
def tic():
    _tstart_stack.append(time())
def toc(fmt="Time Elapsed: %s"):
    print fmt % (time()-_tstart_stack.pop())
    
## Face Detection Configuration
minsize = 40
threshold = [0.7, 0.8, 0.8]
factor = 0.509
tolerance=0.32


### load detection model - action
PNet, RNet, ONet = loadcaffemodel()

# Test   - TODO give only path to the main folder. Let the network see the folder and their names and detect faces inside folders and assign folder names to encodings.
paths = []
known_encodings=[]
ids = []
_ImagesNames = glob('./samples/**/**.jpg')

for _ImageName in _ImagesNames:
	head, basename = os.path.split(_ImageName)
	ID = os.path.split(head)[-1]
	paths.append([ID,_ImageName])
print paths	
for path in paths: #TODO 
	#img_test = scipy.misc.imread(path, mode='RGB')
	img = cv2.resize(cv2.imread(path[1]),(640,480))
	ref_face_encoding = testfaces.encodeTestImage(img, PNet, RNet, ONet, threshold ,factor ,pose_predictor, face_encoder, face_aligner, minsize)
	#for alignedFace in alignedFaces:
		# Save the aligned image to a file
		#cv2.imwrite("aligned_face_{}.jpg".format(i), alignedFace)
	known_encodings.append(ref_face_encoding)
	ids.append(paths[0]) 
	


print('known {}'.format(known_encodings))
print(known_encodings[0])
print('ids {}'.format(ids))
################## Train
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
SPOOF_CHECK = False
HEAD_POSE = False
BLINK = False
while True:
    # Grab a single frame of video
    ret, frame= video_capture.read()
    '''
    if(video_capture.isOpened() == False):
        print("Error: the resource is busy or unvailable")
    else:
        print("The video source has been opened correctly...")
    '''
    height, width, channels = frame.shape
    '''
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
    '''

    # Process begin
    if process_this_frame:
        #tic()
        # Face Detection TODO
        face_locs,scores  = detection.detect_face(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB), minsize, PNet, RNet, ONet, threshold, False, factor)
	# If the face it not fake, proceed to recognize it
	# Recognition
	# Extracting encoding of faces
	face_encodings = dlibapi.face_encodings(frame, np.roll(face_locs,-1,axis=1))
	face_names = []
        #toc()


	if len(face_locs):
	
		# Spoof detection 
		# TODO decide whether it is a fake face or no
		# SPOOF_CHECK = False	
		# 5 frames to confirm the reality of face
		check_counter = 0
		for face_loc in face_locs:
		
			if SPOOF_CHECK:
			
				## Check EyE Blink
				# loop over the face detections
			
				# determine the facial landmarks for the face region, then
				# convert the facial landmark (x, y)-coordinates to a NumPy
				# array
				frame_check = frame
				gray = cv2.cvtColor(frame_check, cv2.COLOR_BGR2GRAY)
				shape = pose_predictor(gray, dlib.rectangle(face_loc[0], face_loc[1], face_loc[2], face_loc[3]))
				shape = face_utils.shape_to_np(shape)

				# extract the left and right eye coordinates, then use the
				# coordinates to compute the eye aspect ratio for both eyes
				leftEye = shape[lStart:lEnd]
				rightEye = shape[rStart:rEnd]
				leftEAR = eye_aspect_ratio(leftEye)
				rightEAR = eye_aspect_ratio(rightEye)

				# average the eye aspect ratio together for both eyes
				ear = (leftEAR + rightEAR) / 2.0

				# compute the convex hull for the left and right eye, then
				# visualize each of the eyes
				leftEyeHull = cv2.convexHull(leftEye)
				rightEyeHull = cv2.convexHull(rightEye)
				cv2.drawContours(frame_check, [leftEyeHull], -1, (0, 255, 0), 1)
				cv2.drawContours(frame_check, [rightEyeHull], -1, (0, 255, 0), 1)

				# check to see if the eye aspect ratio is below the blink
				# threshold, and if so, increment the blink frame counter
				print ear
				if ear < EYE_AR_THRESH:
					COUNTER += 1
					print COUNTER
					# if the eyes were closed for a sufficient number of
					# then sound the alarm
					if COUNTER >= EYE_AR_CONSEC_FRAMES:
						BLINK = True
						COUNTER = 0


    			
    				#cv2.putText(frame, "FACE " + str(i+1), (text_x1,text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1);
            			#cv2.rectangle(frame, 
                       		# 		(face_x1, face_y1), 
                        	# 		(face_x2, face_y2), 
                       		# 		(0, 255, 0), 
                        	#  		2)  
    				cv2.rectangle(frame_check, (face_loc[0], face_loc[1]), 
              			             	(face_loc[2], face_loc[3]), (255,255,255), 2) #(left, top), (right, bottom)
              			             
             
    				
    				if BLINK == True:
    					SPOOF_CHECK = False




			if not SPOOF_CHECK:


                		# Comparing the extracted encoding with known encodings.
				for face_encoding in face_encodings: #TODO use classifiers for comparing
    					distance = dlibapi.compare_faces(known_encodings, face_encoding, tolerance)
					print "distance is:  {} ".format(distance)
                			match=list(distance <= tolerance)
    					name = "unknown"
    					print(match==True)
					if match[0]:#distance[0] < 0.3:
						name = "recognized"
						#print "distance recognized: %f" % distance[0]
					else:
						pass #print "distance of Unknown: %f" % distance[0]
			
					face_names.append(name)
			
				nrof_faces = face_locs.shape[0]#number of faces
				#print('{} Face Detected'.format(nrof_faces))


    				for face_loc, name in zip(face_locs, face_names):
					# make sure face locations are integer values
            				face_loc=face_loc.astype(int)

            				# Draw a box around the face with specific color
					if name == "unknown":
						color = (0, 0, 255)
					else:
						color = (255, 0, 00)
            				cv2.rectangle(frame, (face_loc[0], face_loc[1]), 
              				             (face_loc[2], face_loc[3]), color, 2) #(left, top), (right, bottom)


            				# Draw a label with a name below the face
            				cv2.rectangle(frame, (face_loc[0], face_loc[3]), 
             				    	 (face_loc[2], face_loc[3]+35), color, -1)


            				cv2.putText(frame, name, (face_loc[0] + 6 , 
                     					face_loc[3] + 25 ), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)


    	# Display the resulting image
    	out= cv2.resize(frame,(640,480))
    	cv2.imshow('Video', out)

    #process_this_frame = not process_this_frame

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

