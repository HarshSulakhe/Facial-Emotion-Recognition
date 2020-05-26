import os
import numpy as np
import pandas as pd
from PIL import Image

path = '/home/harsh/Downloads/FER2013/'
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
csv = pd.read_csv(path+'fer2013.csv')
counts = [0,0,0]
csv = csv.to_numpy()
pointer = 0
for row in csv:

    if row[2] == "Training":
        dir = 'train/'
        counts[0]+=1
        pointer = 0
        ###
    elif row[2] == "PublicTest":
        dir = 'valid/'
        counts[1]+=1
        pointer = 1
        ###
    elif row[2] == "PrivateTest":
        dir = 'test/'
        counts[2]+=1
        pointer = 2

    label = label_map[row[0]] + '/'
    if os.path.exists(path+dir+label):
        pass
    else:
        os.mkdir(path+dir+label)

    pixels = np.fromstring(row[1], dtype=int, sep=" ").reshape((48,48))
    image = Image.fromarray(np.uint8(pixels),'L')

    filename = path + dir + label + '{}'.format(counts[pointer]) + '.jpg'

    image.save(filename)
    print("saved:",filename)
