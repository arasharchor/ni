#!/bin/bash

source ~/.profile 
workon cv3.3-py2.7-facerec


./align-dlib.py ~/NovinIlia/ni/training-images/ align outerEyesAndNose ~/NovinIlia/ni/rec-morethanoneimage/aligned-images/ --size 96

./lua/main.lua -outDir ~/NovinIlia/ni/rec-morethanoneimage/generated-embeddings/ -data ~/NovinIlia/ni/rec-morethanoneimage/aligned-images/

./classifier.py train ~/NovinIlia/ni/rec-morethanoneimage/generated-embeddings/


./classifier_webcam_mtcnn.py ~/NovinIlia/ni/rec-morethanoneimage/generated-embeddings/classifier.pkl
