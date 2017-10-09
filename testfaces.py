import cv2
import detection
import dlibapi
import scipy
import numpy as np
import dlib

'''
landmarkMap = {
'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
}
'''

def encodeTestImage(img,PNet,RNet,ONet,threshold,factor, pose_predictor, face_encoder, face_aligner, minsize=20):
	# Load a sample picture and learn how to recognize it. 
	face_locs,scores = detection.detect_face(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), minsize, PNet, RNet, ONet, threshold, False, factor)
	if len(face_locs):
		#alignedFaces = [face_aligner.align(96, img, face_loc,  landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE, skipMulti = 'store_true') for face_loc in face_loc] #534
		pose_landmarks = [pose_predictor(img, dlib.rectangle(face_loc[1], face_loc[0], face_loc[3], face_loc[2])) for face_loc in face_locs]
		#[np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]
		face_encodings = dlibapi.face_encodings(img, np.roll(face_locs,-1), 1)[0]
		return face_encodings
	else:
	 	return {},{}
	 	
	 	
