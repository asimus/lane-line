# lane-line
udacity project1
	Overview

The vehicle direction determined according to lane lines on the road and steering inputs are acted on the steering wheel by 
the driver. Finding lane lines are first step to determine vehicle direction. 
In this project, Pyhton and OpenCV are used in finding lane line.  

The steps of this project are the following;
Lane lines on the road are found by pipeline
Reflection of the work in a written report

	Finding lane lines on the road;

###1.Description of the pipeline and draw_lines() function;

The image is masked and filters applied on the sections of highway to identify lane lines. The images are integral parts of video,
each image processed through pixel information in 3 layers red, green. Following steps are followed to the finding lines;

* The Grayscale transform
* Apply a slight Gaussian blur
* The Canny transform
* Image mask for interested region
* Retrieve Hough lines
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

###2.Potential shortcomings
In a corner cases like curvy roads or faint lines are causing rupture on pipelines. They could be improved by GPS support on 
vehicle location between lanes. 


###3.Possible improvements

~Color contrast could be improved between line markings and road.
~Parameters could be adapt the environment in terms of light level and perimeter etc.



 




