#!/usr/bin/env python3

import time
import math
import numpy as np
import cv2
import rospy

from line_fit import line_fit, tune_fit, bird_fit, final_viz
from Line import Line
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from skimage import morphology

import matplotlib.pyplot as plt

from std_msgs.msg import Float32MultiArray


class lanenet_detector():
    def __init__(self):

        self.bridge = CvBridge()
        # NOTE
        # Uncomment this line for lane detection of GEM car in Gazebo
        self.sub_image = rospy.Subscriber('/front_single_camera/image_raw', Image, self.img_callback, queue_size=1)
        # Uncomment this line for lane detection of videos in rosbag
        # self.sub_image = rospy.Subscriber('camera/image_raw', Image, self.img_callback, queue_size=1)

        self.pub_image = rospy.Publisher("lane_detection/annotate", Image, queue_size=1)
        self.pub_bird = rospy.Publisher("lane_detection/birdseye", Image, queue_size=1)

        self.pub_waypoints = rospy.Publisher('lane_detection/waypoints', Float32MultiArray, queue_size=1)
        self.waypoints_msg = Float32MultiArray()
        # self.waypoints_msg.data = [[0,0],[0,0],[0,0],[0,0]]
        self.waypoints_msg.data = []

        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True


    def img_callback(self, data):

        try:
            # Convert a ROS image message into an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            # plt.imshow(cv_image)
            # plt.show()

        except CvBridgeError as e:
            print(e)

        raw_img = cv_image.copy()


        #####
        mask_image, bird_image, waypoints = self.detection(raw_img)

        if mask_image is not None and bird_image is not None:
            # Convert an OpenCV image into a ROS image message
            out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
            out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')

            # Publish image message in ROS
            self.pub_image.publish(out_img_msg)
            self.pub_bird.publish(out_bird_msg)

            flat_coordinates = [item for sublist in waypoints for item in sublist]
            self.waypoints_msg.data = flat_coordinates
            # self.waypoints_msg.data = waypoints
            self.pub_waypoints.publish(self.waypoints_msg)
            rospy.loginfo(f"Published coordinates: {self.waypoints_msg.data}")
        #####


        ##### Test for each functoin
        # # mask_image = self.gradient_thresh(raw_img)
        # # bird_image = self.color_thresh(raw_img)
        # mask_image = self.combinedBinaryImage(raw_img)
        # mask_image = np.uint8(mask_image * 255)
        # bird_image, M, Minv = self.perspective_transform(mask_image)
        # bird_image = np.uint8(bird_image * 255)
        # # _, bird_image = self.detection(raw_img)
        # if mask_image is not None and bird_image is not None:
        #     # Convert an OpenCV image into a ROS image message
        #     # out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'bgr8')
        #     out_img_msg = self.bridge.cv2_to_imgmsg(mask_image, 'mono8')
        #     # out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'bgr8')
        #     out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, 'mono8')
        #     # Publish image message in ROS
        #     self.pub_image.publish(out_img_msg)
        #     self.pub_bird.publish(out_bird_msg)
        #####




    def gradient_thresh(self, img, thresh_min=25, thresh_max=100):
        """
        Apply sobel edge detection on input image in x, y direction
        """
        #1. Convert the image to gray scale
        #2. Gaussian blur the image
        #3. Use cv2.Sobel() to find derievatives for both X and Y Axis
        #4. Use cv2.addWeighted() to combine the results
        #5. Convert each pixel to unint8, then apply threshold to get binary image

        ## TODO

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5),0)

        sobel_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        sobel_x = cv2.convertScaleAbs(sobel_x)
        sobel_y = cv2.convertScaleAbs(sobel_y)

        grad = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

        ret, binary_output = cv2.threshold(grad, 100, 255, cv2.THRESH_BINARY)

        ####

        return binary_output


    def color_thresh(self, img, thresh=(100, 255)):
        """
        Convert RGB to HSL and threshold to binary image using S channel
        """
        #1. Convert the image from RGB to HSL
        #2. Apply threshold on S channel to get binary image
        #Hint: threshold on H to remove green grass
        ## TODO

        # HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        # # H(Hue):           0-180, 0-R-Y-G-90-B-P-R-180
        # # L(Lightness):     0-255, higher is white, lower is black
        # # S(Saturation):    0-255, higher is bright, lower is gray

        # mask = cv2.inRange(HLS, (0, 0, 0), (70, 255, 255)) # sim
        # # mask = cv2.inRange(HLS, (0, 200, 0), (180, 255, 150)) # rosbag

        # Masked = cv2.bitwise_and(HLS, HLS, mask= mask)
        # GRY = cv2.cvtColor(Masked, cv2.COLOR_BGR2GRAY)

        # ret, binary_output = cv2.threshold(GRY, 100, 255, cv2.THRESH_BINARY)



        #1. Convert RGB image to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        h_thresh_y=(15, 35)

        l_thresh_w=(180, 255)  # in case some white lanes are kinda gray (increase for lighter)

        # extract  channels
        H = hls[:, :, 0]
        L = hls[:, :, 1]
        S = hls[:, :, 2]

        yellow_mask = cv2.inRange(H, h_thresh_y[0], h_thresh_y[1])

        white_mask = cv2.inRange(L, l_thresh_w[0], l_thresh_w[1])

        binary_output = cv2.bitwise_or(yellow_mask, white_mask)

        ####

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
        binaryImage[(ColorOutput==1)|(SobelOutput==1)] = 1
        # Remove noise from binary image
        binaryImage = morphology.remove_small_objects(binaryImage.astype('bool'),min_size=50,connectivity=2)

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

        # gem-gazebo
        # 5 m x 4 m
        # src = np.float32([[227, 312], [411, 312],\
        #                   [10, 480],  [634, 480]])
        # 10 m x 4 m
        src = np.float32([[267, 282], [371, 282],\
                          [10, 480],  [634, 480]])
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
                ret = tune_fit(img_birdeye, left_fit, right_fit)

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
            if ret is not None:
                bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
                combine_fit_img, waypoints = final_viz(img, left_fit, right_fit, Minv)
            else:
                print("Unable to detect lanes")

            return combine_fit_img, bird_fit_img, waypoints


if __name__ == '__main__':
    # init args
    rospy.init_node('lanenet_node', anonymous=True)
    lanenet_detector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)
