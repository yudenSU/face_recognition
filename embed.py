import cv2
from cv2 import IMREAD_GRAYSCALE
from cv2 import IMREAD_COLOR
import numpy as np

# takes in an image as a numpy array 
def embedder(image):
	img = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
	embedder = cv2.dnn.readNetFromTorch("src/nn4.small2.v1.t7")
	faceBlob = cv2.dnn.blobFromImage(img, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)

	embedder.setInput(faceBlob)
	vec = embedder.forward()

	return vec



'''
print("\n"+str(vec)+"\n")

dist = np.linalg.norm(vec) 

print("\n"+str(dist)+"\n")

#########################

img2 = cv2.imread("roi_color_copy.png", IMREAD_COLOR)


faceBlob2 = cv2.dnn.blobFromImage(img2, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)

embedder.setInput(faceBlob2)
vec2 = embedder.forward()

print("\n vec 2 - vec ="+str(vec2)+"\n")


print("\n vec 2 - vec ="+str(vec - vec2)+"\n")

dist2 = np.linalg.norm(vec - vec2) 

print("\n dist 2 = "+str(dist2)+"\n")

'''