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

class PID(object):

    def __init__(self, kp, ki, kd, wg=None):

        self.iterm  = 0
        self.last_t = None
        self.last_e = 0
        self.kp     = kp
        self.ki     = ki
        self.kd     = kd
        self.wg     = wg
        self.derror = 0

    def reset(self):
        self.iterm  = 0
        self.last_e = 0
        self.last_t = None

    def get_control(self, t, e, fwd=0):

        if self.last_t is None:
            self.last_t = t
            de = 0
        else:
            de = (e - self.last_e) / (t - self.last_t)

        if abs(e - self.last_e) > 0.5:
            de = 0

        self.iterm += e * (t - self.last_t)

        # take care of integral winding-up
        if self.wg is not None:
            if self.iterm > self.wg:
                self.iterm = self.wg
            elif self.iterm < -self.wg:
                self.iterm = -self.wg

        self.last_e = e
        self.last_t = t
        self.derror = de

        return fwd + self.kp * e + self.ki * self.iterm + self.kd * de
    
class OnlineFilter(object):

    def __init__(self, cutoff, fs, order):
        
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq

        # Get the filter coefficients 
        self.b, self.a = signal.butter(order, normal_cutoff, btype='low', analog=False)

        # Initialize
        self.z = signal.lfilter_zi(self.b, self.a)
    
    def get_data(self, data):
        filted, self.z = signal.lfilter(self.b, self.a, [data], zi=self.z)
        return filted
    
class vehicleController():

    def __init__(self):
        self.rate = rospy.Rate(10) # 10 Hz

        self.L = 1.75  # Wheelbase of GEMe2
        # self.L = 2.56  # Wheelbase of GEMe4

        # PID for longitudinal control
        self.desired_speed = 0.6  # m/s
        self.max_accel = 0.48  # % of acceleration
        self.pid_speed     = PID(0.5, 0.0, 0.1, wg=20)
        self.speed_filter  = OnlineFilter(1.2, 30, 4)

        self.prev_accel = 0.0
        self.speed      = 0.0
        self.steer = 0.0 # degrees
        self.waypoints = [[2.0,0.0],[2.0,7.5],[2.0,15.0],[2.0,17.0]]

        self.pacmod_enable = True
        self.gem_enable = True

    def longititudal_controller(self):
        curr_x, curr_y = self.waypoints[3]

        wp1_x,   wp1_y = self.waypoints[1]
        wp2_x,   wp2_y = self.waypoints[0]

        vec1_x, vec1_y =             0, wp1_y - curr_y
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

        if target_velocity < 0.3:
            target_velocity = 0.3

        return target_velocity


	# Task 3: Lateral Controller (Pure Pursuit)
    def pure_pursuit_lateral_controller(self):
        # curr - wp0 - wp1 - wp2:Farest from the vehicle
        curr_x, curr_y = self.waypoints[-1]
        wp0_x,   wp0_y = self.waypoints[-2]
        wp1_x,   wp1_y = self.waypoints[-3]
        wp2_x,   wp2_y = self.waypoints[0]


        # Look at nearest point for steep curve
        vec2_x, vec2_y = wp0_x - curr_x, -(wp0_y - curr_y)
        angle_curr_to_wp0  = np.arctan2(vec2_y, vec2_x)

        ld1 = np.sqrt((wp0_x - curr_x)**2 + (wp0_y - curr_y)**2)
        alpha1 = angle_curr_to_wp0 - np.pi/2


        # # Look at x- nearest point
        # if (wp2_x > curr_x and wp0_x < curr_x) or (wp2_x < curr_x and wp0_x > curr_x):
        #     # idx = np.argmin(np.abs(np.array(self.waypoints[0:-1][0]) - curr_x))
        #     wp0_x,   wp0_y = self.waypoints[-3]

        #     vec2_x, vec2_y = wp0_x - curr_x, -(wp0_y - curr_y)
        #     angle_curr_to_wp0  = np.arctan2(vec2_y, vec2_x)

        #     ld1 = np.sqrt((wp0_x - curr_x)**2 + (wp0_y - curr_y)**2)
        #     alpha1 = angle_curr_to_wp0 - np.pi/2


        # Look at 1st-2nd nearest points
        # vec1_x, vec1_y =              0, -(wp0_y - curr_y)
        # vec2_x, vec2_y = wp1_x -  wp0_x, -(wp1_y -  wp0_y)

        # angle_curr_to_wp0 = np.arctan2(vec1_y, vec1_x)  
        # angle_wp0_to_wp1  = np.arctan2(vec2_y, vec2_x)  

        # ld1 = np.sqrt((wp1_x - curr_x)**2 + (wp1_y - curr_y)**2)
        # alpha1 = angle_wp0_to_wp1 - angle_curr_to_wp0
        # print("a1", alpha1)


        # Look at farest point for stable crouse
        # vec1_x, vec1_y =              0, -(wp1_y - curr_y)
        # vec2_x, vec2_y = wp2_x -  wp1_x, -(wp2_y -  wp1_y)

        # angle_curr_to_wp1 = np.arctan2(vec1_y, vec1_x)  
        # angle_wp1_to_wp2 = np.arctan2(vec2_y, vec2_x)  

        # ld2 = np.sqrt((wp2_x - curr_x)**2 + (wp2_y - curr_y)**2)
        # alpha2 = angle_wp1_to_wp2 - angle_curr_to_wp1


        # alpha = alpha2 if abs(alpha2) < abs(alpha1) else alpha1
        # ld = ld1 if alpha == alpha1 else ld2
        alpha = alpha1
        ld = ld1

        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))   # Normalize alpha to the range [-pi, pi]
        target_steering = np.arctan(2 * self.L * np.sin(alpha) / ld)

        if target_steering > np.radians(35):
            target_steering = np.radians(35)
        if target_steering < np.radians(-35):
            target_steering = np.radians(-35)

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

            if(self.pacmod_enable == True):
                if (self.gem_enable == False):

                    self.gem_enable = True

                else: 

                    target_velocity = self.longititudal_controller()
                    target_steering = self.pure_pursuit_lateral_controller()
                    # target_velocity = 1
                    # target_steering = 1.1


                    # # Heading angle [rad] to Steering wheel[deg](-630 to 630 deg)
                    target_steering = np.degrees(target_steering)
                    target_steering = self.front2steer(target_steering)

                    print("-----")
                    print("target_velocity[m/s]", round(target_velocity, 2))
                    print("target_steering[deg]", round(target_steering, 2))

                    # target_steering = np.radians(target_steering)

                    current_time = rospy.get_time()
                    filt_vel     = self.speed_filter.get_data(self.speed)
                    # acc = target_velocity - filt_vel
                    acc = self.pid_speed.get_control(current_time, self.desired_speed - filt_vel)


                    # # GEMe4
                    # if acc < 0.3:
                    #     acc = acc + 0.1

                    # if acc > self.max_accel :
                    #     throttle_percent = self.max_accel
                    # elif acc < 0.4 :
                    #     throttle_percent = acc
                    #     if self.prev_accel > 0.4:
                    #         throttle_percent = self.prev_accel - 0.001
                    # else:
                    #     throttle_percent = acc
                    # self.prev_accel = throttle_percent

                    thresh_acc       = 0.29
                    min_acc          = 0.20
                    throttle_percent = 0.10

                    if acc > self.max_accel:
                        throttle_percent = min(self.prev_accel + 0.005, self.max_accel)
                    elif acc >= thresh_acc:
                        if acc > self.prev_accel:
                            throttle_percent = self.prev_accel + 0.005
                        else:
                            throttle_percent = self.prev_accel - 0.005
                    else:  # acc < thresh_acc
                        throttle_percent = max(self.prev_accel - 0.005, min_acc)
                    self.prev_accel = throttle_percent

                    print("- current_speed   ", round(float(filt_vel), 3))
                    print("- current_accel   ", round(float(acc), 3))
                    print("- throttle_percent", round(float(throttle_percent), 3))


def pure_pursuit():

    rospy.init_node('gem_controller', anonymous=True)
    vc = vehicleController()

    try:
        vc.execute()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    pure_pursuit()

