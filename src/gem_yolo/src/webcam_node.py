#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def webcam_publisher():
    # Initialize the ROS node
    rospy.init_node('webcam_publisher', anonymous=True)

    # Create a publisher for the image topic
    image_pub = rospy.Publisher('/camera/color/image_raw', Image, queue_size=10)

    # Create a CvBridge object for converting OpenCV images to ROS Image messages
    bridge = CvBridge()

    # Open the webcam (default device is /dev/video0)
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        rospy.logerr("Unable to open webcam. Check if it's connected and accessible.")
        return

    rospy.loginfo("Webcam publisher node started, publishing to /camera/color/image_raw")

    # Set the publishing rate (in Hz)
    rate = rospy.Rate(10)  # Adjust as needed

    while not rospy.is_shutdown():
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            rospy.logwarn("Failed to capture image from webcam. Skipping frame.")
            continue

        # Convert the frame to a ROS Image message
        image_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")

        # Publish the image message
        image_pub.publish(image_msg)

        # Sleep to maintain the publishing rate
        rate.sleep()

    # Release the webcam
    cap.release()

if __name__ == '__main__':
    try:
        webcam_publisher()
    except rospy.ROSInterruptException:
        pass