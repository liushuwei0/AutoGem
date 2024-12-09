#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ImageContrastAdjuster:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/zed2/zed_node/rgb/image_rect_color", Image, self.image_callback)
        self.current_image = None
        self.gray_lower = 0  # Lower bound for gray detection
        self.gray_upper = 150  # Upper bound for gray detection
        self.gray_target = 60  # Target gray value

        cv2.namedWindow('Image Contrast Adjuster')
        cv2.createTrackbar('Gray Lower', 'Image Contrast Adjuster', self.gray_lower, 255, self.update_gray_lower)
        cv2.createTrackbar('Gray Upper', 'Image Contrast Adjuster', self.gray_upper, 255, self.update_gray_upper)
        cv2.createTrackbar('Gray Target', 'Image Contrast Adjuster', self.gray_target, 255, self.update_gray_target)

    def update_gray_lower(self, value):
        self.gray_lower = value

    def update_gray_upper(self, value):
        self.gray_upper = value

    def update_gray_target(self, value):
        self.gray_target = value

    def image_callback(self, data):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

    def adjust_gray_levels(self, image, gray_lower, gray_upper, gray_target):
        image_float = image.astype(np.float32)

        r = image_float[:, :, 0]
        g = image_float[:, :, 1]
        b = image_float[:, :, 2]

        mask = (r < gray_upper) & (g < gray_upper) & (b < gray_upper)
        image[mask] = [gray_target, gray_target, gray_target]
        
        return image

    def run(self):
        while not rospy.is_shutdown():
            if self.current_image is not None:
                adjusted_image = self.adjust_gray_levels(self.current_image, self.gray_lower, self.gray_upper, self.gray_target)
                cv2.imshow('Image Contrast Adjuster', adjusted_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node('image_contrast_adjuster', anonymous=True)
    adjuster = ImageContrastAdjuster()
    adjuster.run()