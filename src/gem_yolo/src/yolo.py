#!/usr/bin/env python3
import time

import ros_numpy
import rospy

from sensor_msgs.msg import Image
from std_msgs.msg import Bool

import torch
from ultralytics import YOLO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rospy.loginfo(f"Running YOLO model on device: {device}")
time.sleep(1)

detection_model = YOLO("../weight/yolov8m.pt")
# segmentation_model = YOLO("../weight/yolov8m-seg.pt")

rospy.init_node("ultralytics_yolo")
time.sleep(1)

det_image_pub = rospy.Publisher("/ultralytics_yolo/detection/image", Image, queue_size=5)
# seg_image_pub = rospy.Publisher("/ultralytics_yolo/segmentation/image", Image, queue_size=5)
stop_signal_pub = rospy.Publisher("/ultralytics_yolo/stop_sign_detection/detected", Bool, queue_size=5)
human_pub = rospy.Publisher("/ultralytics_yolo/human_detection/detected", Bool, queue_size=5)


def callback(data):
    """Callback function to process image and publish annotated images."""
    array = ros_numpy.numpify(data)

    # Using YOLO detection model
    if det_image_pub.get_num_connections():
        det_result = detection_model(array)
        det_annotated = det_result[0].plot(show=False)
        det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="rgb8"))

    # Using YOLO segmentation model
    # if seg_image_pub.get_num_connections():
    #     seg_result = segmentation_model(array)
    #     seg_annotated = seg_result[0].plot(show=False)
    #     seg_image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding="rgb8"))

    for box in det_result[0].boxes:
        class_id = box.cls.numpy()
        class_name = detection_model.names[int(class_id)]  # Convert ID to class name
        if class_name == "stop sign":
            # stop_signal_pub.publish(True)
            # rospy.loginfo("Stop Sign Detected!")
            xyxy = box.xyxy.numpy()

            w, h = xyxy[0][2]-xyxy[0][0], xyxy[0][3]-xyxy[0][1]
            x, y = xyxy[0][0]+w/2, xyxy[0][1]+h/2
            print("Size of stop sign:", w, h)
            print("Center of stop sign:", x, y)

            if w > 30 and h > 30:
                stop_signal_pub.publish(True)
                rospy.loginfo("Stop Sign Detected!")

            # confidence = box.conf.numpy()
            # rospy.loginfo(f"Detected: {class_name}, Confidence: {confidence}, Box: {xyxy}")
        if class_name == "person":
            human_pub.publish(True)
            rospy.loginfo("Human Detected!")
            # xyxy = box.xyxy.numpy()
            # confidence = box.conf.numpy()
            # rospy.loginfo(f"Detected: {class_name}, Confidence: {confidence}, Box: {xyxy}")


# rospy.Subscriber("/camera/color/image_raw", Image, callback)
rospy.Subscriber("/front_single_camera/image_raw", Image, callback)

while True:
    rospy.spin()