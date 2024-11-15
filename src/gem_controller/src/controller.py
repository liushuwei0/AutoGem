#!/usr/bin/env python3

# Python Headers
import os 
import csv
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal

# ROS Headers
import rospy
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import String, Bool, Float32, Float64, Float32MultiArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Twist

# GEM PACMod Headers
# from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt

from gazebo_msgs.srv import GetModelState, GetModelStateResponse


class vehicleController():

    def __init__(self):
        self.rate = rospy.Rate(100)

        # Publisher to publish the control input to the vehicle model
        self.controlPub = rospy.Publisher("/ackermann_cmd", AckermannDrive, queue_size=1)
        self.L = 1.75  # Wheelbase, can be get from gem_control.py
        self.log_acceleration = True  # Set to True to log acceleration
        self.acceleration_list = []

        # PID for longitudinal control
        self.desired_speed = 0.6  # m/s
        self.max_accel = 0.48  # % of acceleration
        # self.controlSub = rospy.Subscriber('/cmd_vel', Twist, self.callback)

        #Pack computed velocity and steering angle into Ackermann command
        self.ackermann_msg = AckermannDrive()
        self.ackermann_msg.acceleration            = 0.0
        self.ackermann_msg.steering_angle          = 0.0

        rospy.Subscriber('lane_detection/waypoints', Float32MultiArray, self.callback)
        self.waypoints = [[2.0,0.0],[2.0,7.5],[2.0,15.0],[2.0,17.0]]

    def callback(self, msg):
        self.waypoints = [(msg.data[i], msg.data[i+1]) for i in range(0, len(msg.data), 2)]

    def getModelState(self):
        # Get the current state of the vehicle
        # Input: None
        # Output: ModelState, the state of the vehicle, contain the
        #   position, orientation, linear velocity, angular velocity
        #   of the vehicle
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            resp = serviceResponse(model_name='gem')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: "+str(exc))
            resp = GetModelStateResponse()
            resp.success = False
        return resp

    def extract_vehicle_info(self, currentPose):
        vel_x = currentPose.twist.linear.x
        vel_y = currentPose.twist.linear.y
        vel = np.sqrt(vel_x**2 + vel_y**2)  
        return vel

    def longititudal_controller(self):
        curr_x, curr_y = self.waypoints[3]

        wp1_x,   wp1_y = self.waypoints[1]
        wp2_x,   wp2_y = self.waypoints[0]

        vec1_x, vec1_y = wp1_x - curr_x, wp1_y - curr_y
        vec2_x, vec2_y = wp2_x - wp1_x, wp2_y - wp1_y

        dot_product = vec1_x * vec2_x + vec1_y * vec2_y

        # magnitudes of the vectors
        mag1 = np.sqrt(vec1_x**2 + vec1_y**2)
        mag2 = np.sqrt(vec2_x**2 + vec2_y**2)

        # cosine of the angle between the two vectors
        cos_theta = dot_product / (mag1 * mag2)

        if cos_theta > 0.98:  # Near straight, 1 for "straightness"
            target_velocity = self.desired_speed
        else:
            target_velocity = self.desired_speed * cos_theta

        return target_velocity


	# Task 3: Lateral Controller (Pure Pursuit)
    def pure_pursuit_lateral_controller(self):
        # curr - wp0 - wp1 - wp2:Farest from the vehicle
        curr_x, curr_y = self.waypoints[3]
        wp0_x,   wp0_y = self.waypoints[2]
        wp1_x,   wp1_y = self.waypoints[1]
        wp2_x,   wp2_y = self.waypoints[0]

        # vec1_x, vec1_y =              0, -(wp1_y - curr_y)
        # vec2_x, vec2_y = wp2_x -  wp1_x, -(wp2_y -  wp1_y)
        # # vec2_x, vec2_y = wp1_x -  wp0_x, -(wp1_y -  wp0_y)

        # angle_wp0_to_wp1 = np.arctan2(vec1_y, vec1_x)  
        # angle_wp1_to_wp2 = np.arctan2(vec2_y, vec2_x)  

        # ld = np.sqrt((wp2_x - curr_x)**2 + (wp2_y - curr_y)**2)
        # alpha = angle_wp1_to_wp2 - angle_wp0_to_wp1



        vec1_x, vec1_y =              0, -(wp0_y - curr_y)
        vec2_x, vec2_y = wp1_x -  wp0_x, -(wp1_y -  wp0_y)

        angle_curr_to_wp0 = np.arctan2(vec1_y, vec1_x)  
        angle_wp0_to_wp1  = np.arctan2(vec2_y, vec2_x)  

        ld = np.sqrt((wp1_x - curr_x)**2 + (wp1_y - curr_y)**2)
        alpha = angle_wp0_to_wp1 - angle_curr_to_wp0

        if alpha > 0.1:
            vec1_x, vec1_y =              0, -(wp1_y - curr_y)
            vec2_x, vec2_y = wp2_x -  wp1_x, -(wp2_y -  wp1_y)

            angle_curr_to_wp1 = np.arctan2(vec1_y, vec1_x)  
            angle_wp1_to_wp2 = np.arctan2(vec2_y, vec2_x)  

            ld = np.sqrt((wp2_x - curr_x)**2 + (wp2_y - curr_y)**2)
            alpha = angle_wp1_to_wp2 - angle_curr_to_wp1



        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))   # Normalize alpha to the range [-pi, pi]
        target_steering = np.arctan(2 * self.L * np.sin(alpha) / ld)

        return target_steering

    def front2steer(self, f_angle):
        if(f_angle > 35):
            f_angle = 35
        if (f_angle < -35):
            f_angle = -35
        if (f_angle > 0):
            steer_angle = round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        elif (f_angle < 0):
            f_angle = -f_angle
            steer_angle = -round(-0.1084*f_angle**2 + 21.775*f_angle, 2)
        else:
            steer_angle = 0.0
        return steer_angle

    def execute(self):
        while not rospy.is_shutdown():
            self.rate.sleep()  # Wait a while before trying to get a new state

            currState =  self.getModelState()
            curr_vel  = self.extract_vehicle_info(currState)
            # print("curr_vel", curr_vel)

            target_velocity = self.longititudal_controller()
            target_steering = self.pure_pursuit_lateral_controller()
            # target_velocity = 1
            # target_steering = 1.1


            # # Heading angle [rad] to Steering wheel[deg](-630 to 630 deg)
            target_steering = np.degrees(target_steering)
            # target_steering = self.front2steer(target_steering)

            print("-----")
            print("target_velocity[m/s]", round(target_velocity, 2))
            print("target_steering[deg]", round(target_steering, 2))

            target_steering = np.radians(target_steering)

            acc = target_velocity - curr_vel

            if acc > 0.64 :
                throttle_percent = self.max_accel
            elif acc < 0.0 :
                throttle_percent = 0.0
            else:
                throttle_percent = acc / 0.64 * self.max_accel  

            self.ackermann_msg.speed = target_velocity # m/s2

            self.ackermann_msg.acceleration = throttle_percent # m/s2
            self.ackermann_msg.steering_angle = target_steering # rad

            self.controlPub.publish(self.ackermann_msg)


def pure_pursuit():

    rospy.init_node('test_node', anonymous=True)
    vc = vehicleController()

    try:
        vc.execute()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    pure_pursuit()

