# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 21:49:37 2018

@author: mozlen
"""

#packages for lane keeping
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

#reading images for getting plots
image = mpimg.imread('frame0.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image) 

# In[1]: this part is depicted from riddlex on github

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

# In[2]
    
import os
os.listdir("test_images/")

from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
   
    result = grayscale(image)
    result = gaussian_blur(result, 5)
    result = canny(result, 50, 150)
    regions = np.array([[
        [0, image.shape[0]],
        [image.shape[1] * 0.45, image.shape[0] * 0.65],
        [image.shape[1] * 0.55, image.shape[0] * 0.65],
        [image.shape[1], image.shape[0]] 
    ]], dtype=np.int32)
    result = region_of_interest(result, regions)
    plt.imshow(result, cmap='gray')
    
    rho = 2
    theta = np.pi / 180 * 1
    threshold = 10
    min_line_len = 20
    max_line_gap = 10
    img = result
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    lines = process_lines(lines, image)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines, thickness = 10)
    result = weighted_img(line_img, image, β=0.5)
    
    return result

def process_lines(lines, image):
    '''
    Valid line should intersect the bottem edge, and slope should be in certain range
    Pick average of intersection and slope
    Only left 2 lines: left line has negtive slope, right line has positive slope
    '''
    HEIGHT_R = 0.65 
    results = [{"slope":0.00001, "x":0, "weight": 0} for i in range(2)];
    f_lines = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if x1 == x2 : continue
            slope = (y1 - y2) * 1.0/ (x1 - x2)
            intersect_x = (image.shape[0] - y2) / slope + x2
            if intersect_x < 0 or intersect_x >= image.shape[1]:
                continue
            if slope > -0.85 and slope < -0.5:
                index = 0
            elif slope > 0.5 and slope < 0.85:
                index = 1
            else:
                continue
            one_r = results[index]
            
            weight = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
            one_r["slope"] += slope * weight
            one_r["x"] += intersect_x * weight
            one_r["weight"] += weight
            
    for one_r in results:
       
        if one_r["weight"] == 0:
            f_lines.append([[0,0,0,0]])
            continue
        one_r["x"] /= float(one_r["weight"])
        one_r["slope"] /= float(one_r["weight"])

        point0 = (int(one_r["x"]), image.shape[0])       
        point1 = (int(image.shape[0] * (HEIGHT_R - 1) / one_r["slope"] + one_r["x"]), int(image.shape[0] * HEIGHT_R))
        f_lines.append([point0 + point1])
    return f_lines

# the solid white lane on the right 

white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) 

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))

yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))

challenge_output = 'extra.mp4'
clip2 = VideoFileClip('challenge.mp4')
challenge_clip = clip2.fl_image(process_image)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))
