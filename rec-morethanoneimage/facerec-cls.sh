#!/bin/bash

source ~/.profile 
workon cv3.3-py2.7-facerec


python /home/nvidia/ni/rec-morethanoneimage/align-dlib.py /home/nvidia/NovinIlia/ni/training-images/ align outerEyesAndNose /home/nvidia/NovinIlia/ni/aligned-images/ --size 96

#cd /home/nvidia/ni/rec-morethanoneimage/lua/
../../ni/rec-morethanoneimage/lua/main.lua -outDir /home/nvidia/NovinIlia/ni/generated-embeddings/ -data  /home/nvidia/NovinIlia/ni/aligned-images/

python /home/nvidia/ni/rec-morethanoneimage/classifier.py train /home/nvidia/NovinIlia/ni/generated-embeddings/

python /home/nvidia/ni/rec-morethanoneimage/classifier_webcam_mtcnn.py /home/nvidia/NovinIlia/ni/generated-embeddings/classifier.pkl
