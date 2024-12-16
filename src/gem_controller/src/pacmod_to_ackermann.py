#!/usr/bin/env python3
import rospy
from pacmod_msgs.msg import PacmodCmd, PositionWithSpeed
from ackermann_msgs.msg import AckermannDrive

class PacmodToAckermann:
    def __init__(self):
        rospy.init_node('pacmod_to_ackermann')

        # # Publishers for Pacmod control
        # self.gear_pub = rospy.Publisher('/pacmod/as_rx/shift_cmd', PacmodCmd, queue_size=1)
        # self.brake_pub = rospy.Publisher('/pacmod/as_rx/brake_cmd', PacmodCmd, queue_size=1)
        # self.accel_pub = rospy.Publisher('/pacmod/as_rx/accel_cmd', PacmodCmd, queue_size=1)
        # self.turn_pub = rospy.Publisher('/pacmod/as_rx/turn_cmd', PacmodCmd, queue_size=1)
        # self.steer_pub = rospy.Publisher('/pacmod/as_rx/steer_cmd', PositionWithSpeed, queue_size=1)

        # Ackermann command publisher
        self.ackermann_pub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=1)

        # Subscribers to Pacmod topics
        rospy.Subscriber('/pacmod/as_rx/steer_cmd', PositionWithSpeed, self.steer_callback)
        rospy.Subscriber('/pacmod/as_rx/accel_cmd', PacmodCmd, self.accel_callback)
        rospy.Subscriber('/pacmod/as_rx/brake_cmd', PacmodCmd, self.brake_callback)

        self.current_steer = 0.0
        self.current_accel = 0.0
        self.current_brake = 0.0

    def steer_callback(self, msg):
        self.current_steer = msg.angular_position
        self.publish_ackermann()

    def accel_callback(self, msg):
        self.current_accel = msg.f64_cmd  # Assuming this is the acceleration value
        self.publish_ackermann()

    def brake_callback(self, msg):
        self.current_brake = msg.f64_cmd  # Assuming this is the brake value
        self.publish_ackermann()

    def publish_ackermann(self):
        ackermann_msg = AckermannDrive()
        ackermann_msg.steering_angle = self.current_steer
        ackermann_msg.speed = self.current_accel - self.current_brake  # Subtract brake for net velocity

        # Publish to /ackermann_cmd
        self.ackermann_pub.publish(ackermann_msg)

        # Debugging: Print Ackermann command values before publishing
        rospy.loginfo(f"Publishing Ackermann - Speed: {ackermann_msg.speed}, Steering Angle: {ackermann_msg.steering_angle}")

if __name__ == '__main__':
    try:
        PacmodToAckermann()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
