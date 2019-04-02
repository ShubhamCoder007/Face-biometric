import cv2
import os

filenames = os.listdir()

for name in filenames:
	try:

		img = cv2.imread(name,1)
				
		cascade_obj = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
		face = cascade_obj.detectMultiScale(img, scaleFactor = 1.05, minNeighbors = 5)
		face = face.tolist()[0]
		x,y,w,h = face

		crop_img = img[y:y+h, x:x+w]
		cv2.imwrite(name, crop_img)
		cv2.waitKey(0)

	except ValueError:
		print("Was unable to bound for face!")