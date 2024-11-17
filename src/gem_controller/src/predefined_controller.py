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
        self.rate = rospy.Rate(10)

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

        rospy.sleep(1)        

        start_time = rospy.Time.now().to_sec()

        while not rospy.is_shutdown():

            end_time = rospy.Time.now().to_sec()
            elapsed_time = round(end_time - start_time, 1)

            print("Time elapsed: ", elapsed_time)
            if elapsed_time < 33.4:
                target_velocity = 1 # m/s
                target_steering = 0 # rad
            elif elapsed_time < 54:
                target_velocity = 1 # m/s
                target_steering = 0.38 # rad
            elif elapsed_time < 61.2:
                target_velocity = 1 # m/s
                target_steering = -0.38 # rad
            elif elapsed_time < 80:
                target_velocity = 1 # m/s
                target_steering = 0 # rad
            else:
                target_velocity = 0 # m/s
                target_steering = 0 # rad


            currState =  self.getModelState()
            curr_vel  = self.extract_vehicle_info(currState)
            # print("curr_vel", curr_vel)


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

            self.rate.sleep()  # Wait a while before trying to get a new state

def pure_pursuit():

    rospy.init_node('test_node', anonymous=True)
    vc = vehicleController()

    try:
        vc.execute()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    pure_pursuit()

