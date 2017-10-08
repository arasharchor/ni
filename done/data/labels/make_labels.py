import os

l = ["face"]

for word in l:
    os.system("convert -fill black -background white -bordercolor white -border 4 -font Courier -pointsize 18 label:\"%s\" \"%s.png\""%(word, word))
