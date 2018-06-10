import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse

#color and intensity conversions
def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    
def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def select_white_yellow(image):
    converted = convert_hls(image)
    # white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)

def convert_clahe(image):  
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(image)
    return cl1


###furthur image processing
def apply_smoothing(image, kernel_size=15):
     return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def detect_edges(image, low_threshold=60, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)

def adjust_gamma(image, gamma=1.0):

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
   
##identify ROI
def region_of_interest(img, vertices):
      # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    # Retrieve the number of color channels of the image.
    channel_count = img.shape[2]
    # Create a match color with the same color channel counts.
    match_mask_color = (255,) * channel_count
      
    # Fill the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)
    
    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#identify rectangles
def detect_shape(cnt):
     peri = cv2.arcLength(cnt,True)
     v= cv2.approxPolyDP(cnt, 0.04 * peri, True)
     area=cv2.contourArea(cnt)
        
     if(len(v)==4):
        x, y, width, height = cv2.boundingRect(v)
        if(height>width and area>2000):
            cv2.rectangle(road,(x,y),(x+width,y+height),(0,255,255),2)
            
#draw lane lines
def draw_lines_nm(img,lines):
    if lines is None:
            return
    else:
        flag_left=False
        flag_right=False
        flag_mid=False
        for line in lines:
            for rho,theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                if(flag_left==False):
                    if(x1<0 and y1>0):
                        flag_left=True
                        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2) 
                if(flag_right==False):
                    if(x1<0 and y1<0):
                        flag_right=True
                        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
                if(flag_mid==False):
                    if(x1>0 and y1>0):
                        flag_mid=True
                        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2) 
                if(flag_left==True and flag_right==True and flag_mid==True):
                    break
    return img 


def find_contns(org_img,bin_img):
    #finding and drawing contours
    _,contours,h=cv2.findContours(bin_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(org_img, contours, -1, (0,255,0), 2)
    
    #identify crossings
    for i in contours:
        detect_shape(i)
        
    #identify lane boundaries
    rho=1
    theta=np.pi/60
    line_thresh_val=80

    lines = cv2.HoughLines(bin_img,rho,theta,line_thresh_val)
    img=draw_lines_nm(road,lines=lines)
    
    return img
    
#applying the algorithm

#input the image
road=cv2.imread('/Users/Rajitha/AnacondaProjects/images/IMG01545.jpg',cv2.IMREAD_COLOR)

#resize the image
road=cv2.resize(road,(1280,1024))
h,w=road.shape[:2]
road=cv2.resize(road,(h/2,w/2))
cv2.imwrite('/Users/Rajitha/AnacondaProjects/result_images/img2_result_org_image.jpg',road)
h,w=road.shape[:2]

#identify ROI vertices
region_of_interest_vertices = [(0, h),(w,h),(w-100,h-350),(w-400,h-350)]

#crop the images
cropped_image = region_of_interest( road,np.array([region_of_interest_vertices]),)

#apply gamma transform
road_gamma=adjust_gamma(cropped_image,gamma=3.9)

#color conversions
road_color=select_white_yellow(road_gamma)

#convert to grayscale
road_gray=convert_gray_scale(road_color)

#histogram equalization
road_clahe=convert_clahe(road_gray)

#smoothing image using gaussian blur
road_blur=apply_smoothing(road_clahe,kernel_size=15)

#detect edges using Canny edges
road_edge=detect_edges(road_blur,low_threshold=150,high_threshold=250)

#identify road markings and detection
img=find_contns(road,road_edge)
    

#showing the output image
cv2.imshow('f1',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

##descriptions
# green color areas show all the road markins which are identified by the system. 
# yellow color rectangels show the pedestrian crssoings which are approximately identified by the system.



    
    