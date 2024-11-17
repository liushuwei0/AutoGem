#!/usr/bin/env python3

import time
import math
import numpy as np
import cv2
import rospy

from line_fit import line_fit, tune_fit, bird_fit, final_viz, viz1
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology

import matplotlib.pyplot as plt


class lanenet_detector():
    def __init__(self):

        self.bridge = CvBridge()
        # NOTE
        # Uncomment this line for lane detection of GEM car in Gazebo
        self.sub_image = rospy.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image, self.img_callback, queue_size=1)
        # Uncomment this line for lane detection of videos in rosbag
        # self.sub_image = rospy.Subscriber('camera/image_raw', Image, self.img_callback, queue_size=1)

        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)


        ### ===== Uncomment this block to see the result of the three filtered image ===== ###
        # self.pub_color_thresh = rospy.Publisher("lane_detection/color_thresh", Image, queue_size=1)
        # self.pub_grad_thresh = rospy.Publisher("lane_detection/grad_thresh", Image, queue_size=1)
        # self.pub_combine_thresh = rospy.Publisher("lane_detection/combine_thresh", Image, queue_size=1)
        ######################################################################################

        self.pub_waypoints = rospy.Publisher('lane_detection/waypoints', Float32MultiArray, queue_size=1)
        self.waypoints_msg = Float32MultiArray()
        self.waypoints_msg.data = []
    
        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True


    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()

        rows, cols, _ = raw_img.shape
        x_clip = int(cols//5)
        y_clip = int(rows//2)
        # y_clip = 3*int(rows//5)

        raw_img[:y_clip, :] = [60, 60, 60]
        raw_img[:, :x_clip] = [60, 60, 60]
        raw_img[:, 4*x_clip:] = [60, 60, 60]

        # cv2.imwrite("test.png", raw_img)
        # time.sleep(10000000)



        #####
        mask_image, bird_image, waypoints = self.detection(raw_img)

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)

            # Publish waypoints
            flat_coordinates = [item for sublist in waypoints for item in sublist]
            self.waypoints_msg.data = flat_coordinates
            self.pub_waypoints.publish(self.waypoints_msg)
            print("Published coordinates: ", self.waypoints_msg.data)
        #####


        ### ===== Uncomment this block to see the result of the three filtered image ===== ###
        ##### Test for each functoin
        # grad_image = self.gradient_thresh(raw_img)
        # colorTH_image = self.color_thresh(raw_img)

        # combine_image = self.combinedBinaryImage(raw_img)
        # combine_image = (combine_image* 255).astype(np.uint8)

        # if grad_image is not None:
        #     out_grad_img_msg = self.bridge.cv2_to_imgmsg(grad_image, 'mono8')
        #     self.pub_grad_thresh.publish(out_grad_img_msg)
        # if colorTH_image is not None:
        #     out_colorTH_img_msg = self.bridge.cv2_to_imgmsg(colorTH_image, 'mono8')
        #     self.pub_color_thresh.publish(out_colorTH_img_msg)

        # if combine_image is not None:
        #     out_combine_img_msg = self.bridge.cv2_to_imgmsg(combine_image, 'mono8')
        #     self.pub_combine_thresh.publish(out_combine_img_msg)
        ######################################################################################


    def gradient_thresh(self, img, thresh_min=120, thresh_max=300):
        """
        Apply sobel edge detection on input image in x, y direction
        #1. Convert the image to gray scale
        #2. Gaussian blur the image
        #3. Use cv2.Sobel() to find derievatives for both X and Y Axis
        #4. Use cv2.addWeighted() to combine the results
        #5. Convert each pixel to unint8, then apply threshold to get binary image
        """
        ## TODO
        # # ===== Sobel Filter =====
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(gray, (5,5),0)

        # sobel_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        # sobel_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        # sobel_x = cv2.convertScaleAbs(sobel_x)
        # sobel_y = cv2.convertScaleAbs(sobel_y)

        # grad = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

        # ret, binary_output = cv2.threshold(grad, 50, 255, cv2.THRESH_BINARY)
        
        # ===== Canny Edge Detection =====
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = self.increase_contrast(gray)
        # Apply Gaussian blur to reduce noise and improve edge detection
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use Canny edge detector with provided threshold values
        binary_output = cv2.Canny(blur, thresh_min, thresh_max)

        kernel = np.ones((5, 5), np.uint8)
        binary_output = cv2.dilate(binary_output, kernel, iterations=1)
        ####

        return binary_output


    def increase_contrast(self, gray_img, clip_limit=2, tile_grid_size=(10,10)):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        contrast_img = clahe.apply(gray_img)
        return contrast_img
    
    def color_thresh(self, img):

        ### OLD VERSION ###
        # HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        # # H(Hue):           0-180, 0-R-Y-G-90-B-P-R-180
        # # L(Lightness):     0-255, higher is white, lower is black
        # # S(Saturation):    0-255, higher is bright, lower is gray

        # mask = cv2.inRange(HLS, (100, 50, 0), (180, 255, 255)) # Highbay
        # # mask = cv2.inRange(HLS, (0, 0, 0), (70, 255, 255)) # sim
        # # mask = cv2.inRange(HLS, (0, 200, 0), (180, 255, 150)) # rosbag

        # Masked = cv2.bitwise_and(HLS, HLS, mask= mask)
        # GRY = cv2.cvtColor(Masked, cv2.COLOR_BGR2GRAY)

        # ret, binary_output = cv2.threshold(GRY, 15, 255, cv2.THRESH_BINARY)
        # # ret, binary_output = cv2.threshold(GRY, 75, 255, cv2.THRESH_BINARY)

        ### CURRENT VERSION ###
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        L = hls[:,:,1]
        contrast_L = self.increase_contrast(L)

        _, otsu_binary = cv2.threshold(contrast_L, 10, 255, cv2.THRESH_OTSU)

        hls[:,:,1] = contrast_L
        # hls[:,:,1] = otsu_binary

        h_thresh_y = (15,35)
        l_thresh_w = (175, 255)

        H = hls[:,:,0]
        L = hls[:,:,1]

        yellow_mask = cv2.inRange(H, h_thresh_y[0], h_thresh_y[1])
        white_mask = cv2.inRange(L, l_thresh_w[0], l_thresh_w[1])
        ####

        binary_output = cv2.bitwise_or(yellow_mask, white_mask)
        return binary_output


    def combinedBinaryImage(self, img):
        """
        Get combined binary image from color filter and sobel filter
        """
        #1. Apply sobel filter and color filter on input image
        #2. Combine the outputs
        ## Here you can use as many methods as you want.

        ## TODO

        SobelMono8 = self.gradient_thresh(img)
        # SobelMono8 = cv2.GaussianBlur(SobelMono8, (9,1),0)
        # _, SobelMono8 = cv2.threshold(SobelMono8, 100, 255, cv2.THRESH_BINARY)

        ColorMono8 = self.color_thresh(img)
        # ColorMono8 = cv2.GaussianBlur(ColorMono8, (9,1),0)
        # _, ColorMono8 = cv2.threshold(ColorMono8, 100, 255, cv2.THRESH_BINARY)
        
        SobelOutput = SobelMono8 > 0
        ColorOutput = ColorMono8 > 0

        ####

        binaryImage = np.zeros_like(SobelOutput)
        binaryImage[(ColorOutput==1) & (SobelOutput==1)] = 1
        # binaryImage[(SobelOutput==1)] = 1

        
        # Remove noise from binary image
        binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=200,connectivity=2)
        
        return binaryImage


    def perspective_transform(self, img, verbose=False):
        """
        Get bird's eye view from input image
        """
        #1. Visually determine 4 source points and 4 destination points
        #2. Get M, the transform matrix, and Minv, the inverse using cv2.getPerspectiveTransform()
        #3. Generate warped image in bird view using cv2.warpPerspective()

        ## TODO

        img = np.uint8(img * 255)

        rows, cols = img.shape
        rows_b = rows
        cols_b = cols

        # sim
        # src = np.float32([[cols/2 -63, rows/2 +29 ], [cols/2 +63, rows/2 +29],\
        #                 [cols/2 -355, rows/2 +169], [cols/2 +355, rows/2 +169]])

        # rosbag
        # src = np.float32([[cols/2 -95-55, rows/2 +45 ], [cols/2 +95-55, rows/2 +45],\
        #                 [cols/2 -355-55, rows/2 +169], [cols/2 +355-55, rows/2 +169]])
        src = np.float32([[510,416],[710,416],[200,717],[1056,717]])

        dst = np.float32([[0, 0], [cols_b, 0], [0, rows_b], [cols_b, rows_b]])

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = np.linalg.inv(M)

        warped_img = cv2.warpPerspective(img, M, (cols_b, rows_b)) > 0

        ####

        return warped_img, M, Minv


    def detection(self, img):

        binary_img = self.combinedBinaryImage(img)
        img_birdeye, M, Minv = self.perspective_transform(binary_img)

        if not self.hist:
            # Fit lane without previous result
            ret = line_fit(img_birdeye)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']
        else:
            # Fit lane with previous result
            if not self.detected:
                ret = line_fit(img_birdeye)

                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                    self.detected = True

            else:
                left_fit = self.left_line.get_fit()
                right_fit = self.right_line.get_fit()
                # ret = tune_fit(img_birdeye, left_fit, right_fit)
                ret = line_fit(img_birdeye)
                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    left_fit = self.left_line.add_fit(left_fit)
                    right_fit = self.right_line.add_fit(right_fit)

                else:
                    self.detected = False

            # Annotate original image
            bird_fit_img = None
            combine_fit_img = None
            waypoints = []
            if ret is not None:
                bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
                combine_fit_img, waypoints = final_viz(img, left_fit, right_fit, Minv)
                # viz1(img, ret)
            else:
                print("Unable to detect lanes")

            return combine_fit_img, bird_fit_img, waypoints


if __name__ == '__main__':
    # init args
    rospy.init_node('lanenet_node', anonymous=True)
    lanenet_detector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)
