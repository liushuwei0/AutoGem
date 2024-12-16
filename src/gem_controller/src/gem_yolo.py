#!/usr/bin/env python3

### ROS Node for YOLOv8 Object Detection and Segmentation ###
# https://docs.ultralytics.com/guides/ros-quickstart/
# Run "pip install rosnumpy ultralytics"



# Python Headers
import time
import ros_numpy
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from sensor_msgs.msg import Image
from ultralytics import YOLO

# GEM PACMod Headers
# from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt


# def callback(data):
#     """Callback function to process image and publish detected classes."""
#     array = ros_numpy.numpify(data)
#     if classes_pub.get_num_connections():
#         det_result = detection_model(array)
#         classes = det_result[0].boxes.cls.cpu().numpy().astype(int)
#         names = [det_result[0].names[i] for i in classes]
#         classes_pub.publish(String(data=str(names)))


def callback(data):
    """Callback function to process image and publish annotated images."""
    array = ros_numpy.numpify(data)
    array = array[:, :, :3]
    
    if det_image_pub.get_num_connections():
        det_result = detection_model(array)
        det_annotated = det_result[0].plot(show=False)
        det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="rgb8"))

    if seg_image_pub.get_num_connections():
        seg_result = segmentation_model(array)
        seg_annotated = seg_result[0].plot(show=False)
        seg_image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding="rgb8"))


if __name__ == '__main__':
    detection_model = YOLO("yolov8m.pt")
    segmentation_model = YOLO("yolov8m-seg.pt")
    rospy.init_node("ultralytics")
    time.sleep(1)

    # Definition of publishers
    det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=5)
    seg_image_pub = rospy.Publisher("/ultralytics/segmentation/image", Image, queue_size=5)
    # classes_pub = rospy.Publisher("/ultralytics/detection/classes", String, queue_size=5)

    # Definition of subscribers
    rospy.Subscriber("/zed2/zed_node/rgb/image_rect_color", Image, callback)
    # rospy.Subscriber("/front_single_camera/image_raw", Image, callback)
    
    while True:
        rospy.spin()