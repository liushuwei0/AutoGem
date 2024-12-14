#!/usr/bin/env python3

import time
import math
import numpy as np
import cv2
import rospy
import ros_numpy
import torch
from ultralytics import YOLO
from Line import Line
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from skimage import morphology
from std_msgs.msg import Header, Float32MultiArray, Bool

from line_fit import line_fit, tune_fit, bird_fit, final_viz, viz1
from std_msgs.msg import Header
from std_msgs.msg import Float32
import matplotlib.pyplot as plt

import rospkg

class CombinedDetector:
    def __init__(self):
        # Initialize ROS Node
        rospy.init_node("combined_detector_node", anonymous=True)

        # YOLOpv2 initialization
        rospy.loginfo("Loading YOLOpv2 model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"Using device: {self.device}")
        try:
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('gem_lane_detection')
            model_path = f"{package_path}/src/yolopv2.pt"
            self.segmentation_model = torch.jit.load(model_path, map_location=self.device).to(self.device)
            # self.segmentation_model = torch.jit.load("./yolopv2.pt", map_location=self.device).to(self.device)
            self.segmentation_model.eval()
            rospy.loginfo("YOLOpv2 model loaded successfully.")
        except RuntimeError as e:
            rospy.logerr(f"Error loading YOLOpv2 model: {e}")
            rospy.signal_shutdown("Model loading failed.")
            exit()


        self.pub_waypoints = rospy.Publisher('lane_detection/waypoints', Float32MultiArray, queue_size=1)
        self.waypoints_msg = Float32MultiArray()
        self.waypoints_msg.data = []

        self.pub_linefits  = rospy.Publisher('lane_detection/linefits', Float32MultiArray, queue_size=1)
        self.linefits_msg = Float32MultiArray()
        self.linefits_msg.data = []

        self.left_line = Line(n=5)
        self.right_line = Line(n=5)
        self.detected = False
        self.hist = True
        
        # Lane detection initialization
        self.bridge = CvBridge()

        # Publishers
        # green area
        self.pub_lane = rospy.Publisher("/lane_detection/annotate", Image, queue_size=1)
        
        # bird view - with blue and red line fit
        self.pub_bird = rospy.Publisher("/lane_detection/birdseye", Image, queue_size=1)
        # combined yolopv2 with sobel and color
        self.pub_combined_image = rospy.Publisher("/combined_detection/image", Image, queue_size=1)
        # yolo result
        self.pub_yolo = rospy.Publisher("/yolopv2/lane/image", Image, queue_size=1)  # Initialize pub_yolo
        
        # ROS Subscriber
        rospy.Subscriber('/zed2/zed_node/rgb/image_rect_color', Image, self.img_callback, queue_size=1)


    # Added: postprocess masks
    def postprocess_masks(self, masks, image):
        binary_masks = (masks > 0.5).astype(np.uint8) * 255
        image_resized = cv2.resize(image, (binary_masks.shape[1], binary_masks.shape[0]), interpolation=cv2.INTER_LINEAR)
        white_mask = cv2.inRange(image_resized, (140, 140, 140), (255, 255, 255))
        binary_masks = cv2.bitwise_and(binary_masks, binary_masks, mask=white_mask)
        
        return binary_masks


    def img_callback(self, data):
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            raw_img = cv_image.copy()

            # YOLO Segmentation Here
            resized_img = self.resize_image(raw_img)
            tensor_input = self.prepare_input(resized_img, self.device)
            with torch.no_grad():
                results = self.segmentation_model(tensor_input)
            seg_mask = self.extract_segmentation_masks(results)

            # Added: Postprocess the segmentation masks
            # seg_mask = self.postprocess_masks(seg_mask, raw_img)

            if seg_mask is None:
                rospy.logwarn("Segmentation mask is not available. Skipping processing.")
                return

            # Lane Detection and Bird's-eye View
            mask_image, bird_image, waypoints = self.detection(raw_img, seg_mask)

            if mask_image is not None:
                out_img_msg = self.bridge.cv2_to_imgmsg(np.array(mask_image), "bgr8")
                self.pub_lane.publish(out_img_msg)

            if mask_image is not None and bird_image is not None:
                # Publish waypoints
                flat_coordinates = [item for sublist in waypoints for item in sublist]
                self.waypoints_msg.data = flat_coordinates
                self.pub_waypoints.publish(self.waypoints_msg)
                # print("Published coordinates: ", self.waypoints_msg.data)


            if bird_image is not None:
                if bird_image.ndim == 2:  # If the image is single-channel
                    # Convert grayscale/binary to 3-channel BGR
                    bird_image = cv2.cvtColor((bird_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                elif bird_image.ndim == 3:
                    # If already multi-channel, assume it includes red/blue lines
                    bird_image = bird_image.astype(np.uint8)

               
                # Publish the bird_image as a color image
                out_bird_msg = self.bridge.cv2_to_imgmsg(bird_image, "bgr8")
                self.pub_bird.publish(out_bird_msg)
            else:
                rospy.logwarn("bird_image is None.")

            # Combined Binary Image
            combine_image = self.combinedBinaryImage(raw_img, seg_mask)
            if combine_image is not None:
                if combine_image.ndim == 3:  # Check if the image is multi-channel
                    combine_image = cv2.cvtColor(combine_image, cv2.COLOR_BGR2GRAY)  # Convert to single-channel
                combine_image = (combine_image * 255).astype(np.uint8)
                out_combine_img_msg = self.bridge.cv2_to_imgmsg(combine_image, "mono8")
                self.pub_combined_image.publish(out_combine_img_msg)


            # YOLO Visualization
            if seg_mask is not None:
                seg_visual = self.visualize_segmentation(raw_img, seg_mask)
                yolo_msg = self.bridge.cv2_to_imgmsg(seg_visual, "rgb8")
                self.pub_yolo.publish(yolo_msg)

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge error: {e}")
        except Exception as e:
            rospy.logerr(f"Error in img_callback: {e}")
    
    # YOLO Helper func
    def resize_image(self, image, size=(640, 384)):
        """Resize the image to the model's expected input size."""
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

    def prepare_input(self, image, device):
        """Convert a NumPy array to a PyTorch tensor."""
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(device)
    
    def extract_segmentation_masks(self, results):
        try:
            seg_logits = results[2].squeeze(0).squeeze(0)
            seg_masks = (torch.sigmoid(seg_logits) > 0.6).cpu().numpy()  # this para can be adjusted
            return seg_masks
        except Exception as e:
            rospy.logerr(f"Error extracting segmentation masks: {e}")
            return None

    def visualize_segmentation(self, original_image, masks):
        """Visualize segmentation masks on the original image."""
        resized_masks = cv2.resize(masks.astype(np.uint8),
                                   (original_image.shape[1], original_image.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        seg_visual = original_image.copy()
        seg_visual[resized_masks.astype(bool)] = [0, 255, 0]
        return seg_visual

    # Canny & Color
    def gradient_thresh(self, img, thresh_min=50, thresh_max=350):
        """Apply Sobel edge detection on the input image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = self.increase_contrast(gray)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary_output = cv2.Canny(blur, thresh_min, thresh_max)
        kernel = np.ones((5, 5), np.uint8)
        binary_output = cv2.dilate(binary_output, kernel, iterations=1)
        return binary_output

    def increase_contrast(self, gray_img, clip_limit=2, tile_grid_size=(10, 10)):
        """Increase the contrast of the grayscale image."""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(gray_img)

    def color_thresh(self, img):
        """Apply color thresholding to extract lane lines."""
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        L = hls[:, :, 1]
        contrast_L = self.increase_contrast(L)
        
        hls[:,:,1] = contrast_L

        h_thresh_y = (15, 35)
        l_thresh_w = (175, 255)

        H = hls[:, :, 0]
        L = hls[:, :, 1]

        yellow_mask = cv2.inRange(H, h_thresh_y[0], h_thresh_y[1])
        white_mask = cv2.inRange(L, l_thresh_w[0], l_thresh_w[1])

        binary_output = cv2.bitwise_or(yellow_mask, white_mask)
        return binary_output
    
    
    # Adjust: margin in line_fit, weights 
    def combinedBinaryImage(self, img, seg_mask):
        """
        Combine lane detection outputs (color + Sobel filters) with YOLO segmentation mask.
        """
        # Step 1: Apply Sobel and color filters on the input image
        SobelMono8 = self.gradient_thresh(img)
        ColorMono8 = self.color_thresh(img)

        SobelOutput = SobelMono8 > 0
        ColorOutput = ColorMono8 > 0

        binaryImage = np.zeros_like(SobelOutput, dtype=np.uint8)
        # & and | will lead to slightly different results
        binaryImage[(ColorOutput == 1) & (SobelOutput == 1)] = 1
        # binaryImage[SobelOutput == 1] = 1
        # binaryImage[ColorOutput == 1] = 1

        # Step 2: Resize YOLO segmentation mask to match binaryImage dimensions
        seg_mask_resized = cv2.resize(seg_mask.astype(np.uint8),
                                    (binaryImage.shape[1], binaryImage.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

        seg_mask_binary = seg_mask_resized > 0


        # Step 3: Weighted combination (sum equals to denominator)
        # weight_binary = 1
        # weight_seg_mask = 2
        weight_binary = 0.5
        weight_seg_mask = 2.5

        final_combined = (binaryImage.astype(np.float32) * weight_binary +
                        seg_mask_binary.astype(np.float32) * weight_seg_mask)

        # Threshold to finalize binary output
        final_binary_image = (final_combined > (weight_binary + weight_seg_mask) / 3).astype(np.uint8)
        # final_binary_image = (final_combined > (weight_binary + weight_seg_mask) / 2).astype(np.uint8)

        ### original method (without weights)
        # Step 3: Combine the binaryImage with the resized YOLO segmentation mask
        # final_binary_image = np.zeros_like(binaryImage, dtype=np.uint8)
        # final_binary_image[(binaryImage == 1) & (seg_mask_binary == 1)] = 1

        # Step 4: Remove noise from the final binary image
        final_binary_image = morphology.remove_small_objects(
            final_binary_image.astype(bool), min_size=200, connectivity=2
        ).astype(np.uint8)

        return final_binary_image

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

        # src = np.float32([[410,516],[810,516],[200,717],[1056,717]])

        # Move the top/bottom-right point further right
        # src = np.float32([[410,516],[860,516],[200,717],[1156,717]])
        
        # GEM e2 Dec 8
        # src = np.float32([[370,516],[880,516],[100,717],[1156,717]])

        # src = np.float32([[410,516],[860,516],[200,717],[1156,717]])
        src = np.float32([[510,416],[710,416],[200,717],[1056,717]])


        dst = np.float32([[0, 0], [cols_b, 0], [0, rows_b], [cols_b, rows_b]])

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = np.linalg.inv(M)

        warped_img = cv2.warpPerspective(img, M, (cols_b, rows_b)) > 0

        ####

        return warped_img, M, Minv


    def detection(self, img, seg_mask):

        binary_img = self.combinedBinaryImage(img, seg_mask)
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

                    # left_fit = self.left_line.add_fit(left_fit)
                    # right_fit = self.right_line.add_fit(right_fit)

                    self.detected = True

            else:
                # left_fit = self.left_line.get_fit()
                # right_fit = self.right_line.get_fit()
                # ret = tune_fit(img_birdeye, left_fit, right_fit)
                ret = line_fit(img_birdeye)
                if ret is not None:
                    left_fit = ret['left_fit']
                    right_fit = ret['right_fit']
                    nonzerox = ret['nonzerox']
                    nonzeroy = ret['nonzeroy']
                    left_lane_inds = ret['left_lane_inds']
                    right_lane_inds = ret['right_lane_inds']

                    # left_fit = self.left_line.add_fit(left_fit)
                    # right_fit = self.right_line.add_fit(right_fit)

                else:
                    self.detected = False

            # Annotate original image
            bird_fit_img = None
            combine_fit_img = None
            waypoints = []
            if ret is not None:
                bird_fit_img = bird_fit(img_birdeye, ret, save_file=None)
                combine_fit_img, waypoints = final_viz(img, left_fit, right_fit, Minv)

                # Publish line fits
                # if left_fit is None:
                #     left_fit = [0, 0, 0]
                # if right_fit is None:
                #     right_fit = [0, 0, 0]
                # linefits = [left_fit[0], left_fit[1], left_fit[2], right_fit[0], right_fit[1], right_fit[2]]

                linefits = [2.364506304653116e-05, -0.04923174462484201, 223.94522085814225, -5.5157881580171704e-05, 0.026622165839740745, 1161.7379387152469]

                self.linefits_msg.data = linefits
                self.pub_linefits.publish(self.linefits_msg)

                # viz1(img, ret)
            else:
                print("Unable to detect lanes")

            return combine_fit_img, bird_fit_img, waypoints

if __name__ == '__main__':
    CombinedDetector()
    while not rospy.core.is_shutdown():
        rospy.rostime.wallsleep(0.5)

   

