#!/usr/bin/env python2

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import cv2
sys.path.append('/home/nvidia/softwares/caffe3.3/python')
import caffe
import numpy as np

###
import detection
import dlibapi
import dlib
####
import argparse
import os
import pickle
###

import numpy as np
np.set_printoptions(precision=2)
from sklearn.mixture import GMM
import openface


fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join('/usr/src/tensorrt/data/')
dlibModelDir = os.path.join(modelDir, 'recmodels')
openfaceModelDir = os.path.join(modelDir, 'recmodels')



### Loading Detection Models
def loadcaffemodel():
	caffe.set_mode_gpu()
	PNet = caffe.Net("/usr/src/tensorrt/data/detmodels/d1.prototxt", "/usr/src/tensorrt/data/detmodels/d1.caffemodel", caffe.TEST)
	RNet = caffe.Net("/usr/src/tensorrt/data/detmodels/d2.prototxt", "/usr/src/tensorrt/data/detmodels/d2.caffemodel", caffe.TEST)
	ONet = caffe.Net("/usr/src/tensorrt/data/detmodels/d3.prototxt", "/usr/src/tensorrt/data/detmodels/d3.caffemodel", caffe.TEST)
	return PNet, RNet, ONet

#### Camera configuration

## Compute Speed

from time import time
import time
start = time.time()
_tstart_stack = []
def tic():
    _tstart_stack.append(time())
def toc(fmt="Time Elapsed: %s"):
    print fmt % (time()-_tstart_stack.pop())
    
## Face Detection Configuration
minsize = 40
threshold = [0.7, 0.8, 0.8]
factor = 0.509
tolerance=0.6


### load detection model - action
PNet, RNet, ONet = loadcaffemodel()



def getRep(bgrImg):
    start = time.time()
    if bgrImg is None:
        raise Exception("Unable to load image/frame")

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))
    if args.verbose:
        print("Loading the image took {} seconds.".format(time.time() - start))

    start = time.time()

    # Get the largest face bounding box
    # bb = align.getLargestFaceBoundingBox(rgbImg) #Bounding box

    # Get all bounding boxes
    #bb = align.getAllFaceBoundingBoxes(rgbImg)
    face_locs, _ = detection.detect_face(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB), minsize, PNet, RNet, ONet, threshold, False, factor)
    face_locs =np.roll(face_locs,-1,axis=1)
    bb = dlibapi.to_dlib_rect_format(face_locs)
    print(type(bb))
    if bb is None:
        # raise Exception("Unable to find a face: {}".format(imgPath))
        return None
    if args.verbose:
        print("Face detection took {} seconds.".format(time.time() - start))

    start = time.time()

    alignedFaces = []
    for box in bb:
        alignedFaces.append(
            align.align(
                args.imgDim,
                rgbImg,
                box,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))

    if alignedFaces is None:
        raise Exception("Unable to align the frame")
    if args.verbose:
        print("Alignment took {} seconds.".format(time.time() - start))

    start = time.time()

    reps = []
    for alignedFace in alignedFaces:
        reps.append(net.forward(alignedFace))

    if args.verbose:
        print("Neural network forward pass took {} seconds.".format(
            time.time() - start))

    # print (reps)
    return reps


def infer(img, args):
    with open(args.classifierModel, 'r') as f:
        if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)  # le - label and clf - classifer
        else:
                (le, clf) = pickle.load(f, encoding='latin1')  # le - label and clf - classifer

    reps = getRep(img)
    persons = []
    confidences = []
    for rep in reps:
        try:
            rep = rep.reshape(1, -1)
        except:
            print ("No Face detected")
            return (None, None)
        start = time.time()
        predictions = clf.predict_proba(rep).ravel()
        # print (predictions)
        maxI = np.argmax(predictions)
        # max2 = np.argsort(predictions)[-3:][::-1][1]
        persons.append(le.inverse_transform(maxI))
        # print (str(le.inverse_transform(max2)) + ": "+str( predictions [max2]))
        # ^ prints the second prediction
        confidences.append(predictions[maxI])
        if args.verbose:
            print("Prediction took {} seconds.".format(time.time() - start))
            pass
        # print("Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence))
        if isinstance(clf, GMM):
            dist = np.linalg.norm(rep - clf.means_[maxI])
            print("  + Distance from the mean: {}".format(dist))
            pass
    return (persons, confidences)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(
            dlibModelDir,
            "sh_pre.dat"))
    parser.add_argument(
        '--networkModel',
        type=str,
        help="Path to Torch network model.",
        default=os.path.join(
            openfaceModelDir,
            'n4.s2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument(
        '--captureDevice',
        type=int,
        default=0,
        help='Capture device. 0 for latop webcam and 1 for usb webcam')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument(
        'classifierModel',
        type=str,
        help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')

    args = parser.parse_args()

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(
        args.networkModel,
        imgDim=args.imgDim,
        cuda=args.cuda)

    # Capture device. Usually 0 will be webcam and 1 will be usb cam.
    video_capture = cv2.VideoCapture(args.captureDevice)
    video_capture.set(3, args.width)
    video_capture.set(4, args.height)

    confidenceList = []
    while True:
        ret, frame = video_capture.read()
        persons, confidences = infer(frame, args)
        print ("P: " + str(persons) + " C: " + str(confidences))
        try:
            # append with two floating point precision
            confidenceList.append('%.2f' % confidences[0])
        except:
            # If there is no face detected, confidences matrix will be empty.
            # We can simply ignore it.
            pass

        for i, c in enumerate(confidences):
            if c <= args.threshold:  # 0.5 is kept as threshold for known face.
                persons[i] = "_unknown"

                # Print the person name and conf value on the frame
        cv2.putText(frame, "P: {} C: {}".format(persons, confidences),
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('', frame)
        # quit the program on the press of key 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()





