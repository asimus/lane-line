# lane-line
udacity project1

Pyhton and OpenCV are used in finding lane line project. 

Following steps are followed to the finding lines;
* The Grayscale transform
* The Canny transform
* Image mask for interested region
* Video clip samples are used to find lane lines.


Test images are taken with following code;

import cv2

print (cv2.__version__)

vidcap = cv2.VideoCapture('solidWhiteRight.mp4')

success,image = vidcap.read()

count = 0

success = True

while success:

  success,image = vidcap.read()
	
  print 'Read a new frame: ', success
	
  cv2.imwrite("frame%d.jpg" % count, image)   # save frame as JPEG file
	
  count += 1



