#!/usr/bin/env python3
import time

import ros_numpy
import rospy

from sensor_msgs.msg import Image
from std_msgs.msg import Bool

import torch
from ultralytics import YOLO

import rospkg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rospy.loginfo(f"Running YOLO model on device: {device}")
time.sleep(1)

rospack = rospkg.RosPack()
package_path = rospack.get_path('gem_yolo')
model_path = f"{package_path}/weight/best.pt"
detection_model = YOLO(model_path)
# segmentation_model = YOLO("../weight/yolov8m-seg.pt")

rospy.init_node("ultralytics_yolo")
time.sleep(1)

det_image_pub = rospy.Publisher("/ultralytics_yolo/detection/image", Image, queue_size=5)
# seg_image_pub = rospy.Publisher("/ultralytics_yolo/segmentation/image", Image, queue_size=5)
stop_signal_pub = rospy.Publisher("/ultralytics_yolo/stop_sign_detection/detected", Bool, queue_size=5)
human_pub = rospy.Publisher("/ultralytics_yolo/human_detection/detected", Bool, queue_size=5)
cone_pub = rospy.Publisher("/ultralytics_yolo/cone_detection/detected", Bool, queue_size=5)


# Define the actual height of the stop sign and human (in meters)
STOP_SIGN_HEIGHT = 0.4 # Height of a stop sign
HUMAN_HEIGHT     = 1.7 # Height of a human
CONE_HEIGHT      = 0.6 # Height of a cone

# Define ZED2 HD720 (in pixels)
FOCAL_LENGTH = 528  # https://support.stereolabs.com/hc/en-us/articles/360007395634-What-is-the-camera-focal-length-and-field-of-view
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

def calculate_distance(known_height, pixel_height):
    """Calculate the distance from the camera to the object."""
    return (known_height * FOCAL_LENGTH) / pixel_height

def calculate_offsets(center_x, center_y, distance):
    """Calculate the lateral and vertical offsets from the camera center."""
    lateral_offset = (center_x - IMAGE_WIDTH / 2) * distance / FOCAL_LENGTH
    vertical_offset = (center_y - IMAGE_HEIGHT / 2) * distance / FOCAL_LENGTH
    return lateral_offset, vertical_offset


def callback(data):
    """Callback function to process image and publish annotated images."""
    array = ros_numpy.numpify(data)
    array = array[:,:,:3]

    # Using YOLO detection model
    if det_image_pub.get_num_connections():
        det_result = detection_model(array)
        det_annotated = det_result[0].plot(show=False)
        det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="bgr8"))

    # Using YOLO segmentation model
    # if seg_image_pub.get_num_connections():
    #     seg_result = segmentation_model(array)
    #     seg_annotated = seg_result[0].plot(show=False)
    #     seg_image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding="rgb8"))

    for box in det_result[0].boxes:
        class_id = box.cls.cpu().numpy()
        class_name = detection_model.names[int(class_id)]  # Convert ID to class name
        if class_name == "stop sign":
            xyxy = box.xyxy.cpu().numpy()

            w, h = xyxy[0][2]-xyxy[0][0], xyxy[0][3]-xyxy[0][1]
            x, y = xyxy[0][0]+w/2, xyxy[0][1]+h/2
            print("Size of stop sign:", w, h)
            print("Center of stop sign:", x, y)

            distance = calculate_distance(STOP_SIGN_HEIGHT, h)
            lateral_offset, vertical_offset = calculate_offsets(x, y, distance)
            print("Distance to stop sign:", distance)
            print("Lateral offset:", lateral_offset)

            # confidence = box.conf.numpy()
            # rospy.loginfo(f"Detected: {class_name}, Confidence: {confidence}, Box: {xyxy}")

            if w > 90 and h > 90: ### CHECK THIS PARAMETER
                # stop_signal_pub.publish(True)
                rospy.loginfo("Stop Sign Detected!")

        if class_name == "person":
            xyxy = box.xyxy.cpu().numpy()

            w, h = xyxy[0][2]-xyxy[0][0], xyxy[0][3]-xyxy[0][1]
            x, y = xyxy[0][0]+w/2, xyxy[0][1]+h/2
            print("Size of human:", w, h)
            print("Center of human:", x, y)

            distance = calculate_distance(HUMAN_HEIGHT, h)
            lateral_offset, vertical_offset = calculate_offsets(x, y, distance)
            print("Distance to human:", distance)
            print("Lateral offset:", lateral_offset)

            # confidence = box.conf.numpy()
            # rospy.loginfo(f"Detected: {class_name}, Confidence: {confidence}, Box: {xyxy}")

            if w > 50 and h > 150: ### CHECK THIS PARAMETER
                # human_pub.publish(True)
                # stop_signal_pub.publish(True)
                rospy.loginfo("Human Detected!")

        if class_name == "cone":
            xyxy = box.xyxy.cpu().numpy()

            w, h = xyxy[0][2]-xyxy[0][0], xyxy[0][3]-xyxy[0][1]
            x, y = xyxy[0][0]+w/2, xyxy[0][1]+h/2
            print("Size of cone:", w, h)
            print("Center of cone:", x, y)

            distance = calculate_distance(CONE_HEIGHT, h)
            lateral_offset, vertical_offset = calculate_offsets(x, y, distance)
            print("Distance to cone:", distance)
            print("Lateral offset:", lateral_offset)

            # confidence = box.conf.numpy()
            # rospy.loginfo(f"Detected: {class_name}, Confidence: {confidence}, Box: {xyxy}")

            if w > 69 and h > 79: ### CHECK THIS PARAMETER
                # stop_signal_pub.publish(True)
                cone_pub.publish(True)
                rospy.loginfo("Cone Detected!")

# rospy.Subscriber("/camera/color/image_raw", Image, callback)
# rospy.Subscriber("/front_single_camera/image_raw", Image, callback)
rospy.Subscriber("/zed2/zed_node/rgb/image_rect_color", Image, callback)
# rospy.Subscriber("/oak/rgb/image_raw", Image, callback)

while True:
    rospy.spin()