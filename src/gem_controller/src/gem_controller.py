#!/usr/bin/env python3

# Python Headers
import os 
import csv
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal
import time

# ROS Headers
import rospy
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import String, Bool, Float32, Float64, Float32MultiArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Twist

# GEM PACMod Headers
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt
# from gazebo_msgs.srv import GetModelState, GetModelStateResponse

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
        self.hz = 5 ### CHECK THIS PARAMETER
        self.rate = rospy.Rate(self.hz)

        self.L = 1.75  # Wheelbase of GEMe2
        # self.L = 2.56  # Wheelbase of GEMe4

        # PID for longitudinal control
        self.desired_speed = 0.3  # m/s, Normal lane following
        # self.desired_speed = 0.78  # m/s, Stop sign detection
        # self.desired_speed = 0.4  # m/s, Cone detecion

        self.max_accel = 0.48  # % of acceleration
        self.pid_speed     = PID(0.5, 0.0, 0.1, wg=20)
        self.speed_filter  = OnlineFilter(1.2, 30, 4)

        self.prev_accel = 0.0
        self.prev_steer = 0.0
        self.init_steer = True

        self.speed_sub  = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)
        self.speed      = 0.0

        self.steer_sub = rospy.Subscriber("/pacmod/parsed_tx/steer_rpt", SystemRptFloat, self.steer_callback)
        self.steer = 0.0 # degrees

        rospy.Subscriber('lane_detection/waypoints', Float32MultiArray, self.waypoints_callback)
        self.waypoints = [[2.0,0.0],[2.0,4.0],[2.0,7.0],[2.0,10.0],[2.0,13.0],[2.0,17.0]]

        rospy.Subscriber('/ultralytics_yolo/human_detection/detected', Bool, self.yolo_human_callback)
        rospy.Subscriber('/ultralytics_yolo/stop_sign_detection/detected', Bool, self.yolo_stopsign_callback)
        rospy.Subscriber('/ultralytics_yolo/cone_detection/detected', Bool, self.yolo_cone_callback)

        self.detect_human    = False
        self.detect_stopsign = False
        self.detect_cone     = False
        self.ignore_timer    = 0
        self.ignore_init     = True
        self.ignore_waitsec  = 500 # Ignore stop sign for waitsec

        #----------------------T turn modify------------------------------
        self.t_turn = [0,0,0,0,0]
        # -------------------- PACMod setup --------------------

        self.gem_enable    = False
        self.pacmod_enable = False

        # GEM vehicle enable
        self.enable_sub = rospy.Subscriber('/pacmod/as_rx/enable', Bool, self.pacmod_enable_callback)
        # self.enable_cmd = Bool()
        # self.enable_cmd.data = False

        # GEM vehicle gear control, neutral, forward and reverse, publish once
        self.gear_pub = rospy.Publisher('/pacmod/as_rx/shift_cmd', PacmodCmd, queue_size=1)
        self.gear_cmd = PacmodCmd()
        self.gear_cmd.ui16_cmd = 2 # SHIFT_NEUTRAL

        # GEM vehilce brake control
        self.brake_pub = rospy.Publisher('/pacmod/as_rx/brake_cmd', PacmodCmd, queue_size=1)
        self.brake_cmd = PacmodCmd()
        self.brake_cmd.enable = False
        self.brake_cmd.clear  = True
        self.brake_cmd.ignore = True

        # GEM vechile forward motion control
        self.accel_pub = rospy.Publisher('/pacmod/as_rx/accel_cmd', PacmodCmd, queue_size=1)
        self.accel_cmd = PacmodCmd()
        self.accel_cmd.enable = False
        self.accel_cmd.clear  = True
        self.accel_cmd.ignore = True

        # GEM vechile turn signal control
        self.turn_pub = rospy.Publisher('/pacmod/as_rx/turn_cmd', PacmodCmd, queue_size=1)
        self.turn_cmd = PacmodCmd()
        self.turn_cmd.ui16_cmd = 1 # None

        # GEM vechile steering wheel control
        self.steer_pub = rospy.Publisher('/pacmod/as_rx/steer_cmd', PositionWithSpeed, queue_size=1)
        self.steer_cmd = PositionWithSpeed()
        self.steer_cmd.angular_position = 0.0 # radians, -: clockwise, +: counter-clockwise
        self.steer_cmd.angular_velocity_limit = 2.0 # radians/second


    # PACMod enable callback function
    def pacmod_enable_callback(self, msg):
        self.pacmod_enable = msg.data

    def speed_callback(self, msg):
        self.speed = round(msg.vehicle_speed, 3) # forward velocity in m/s

    # Get value of steering wheel
    def steer_callback(self, msg):
        self.steer = round(np.degrees(msg.output),1)

    def waypoints_callback(self, msg):
        if len(msg.data) > 0:
            self.waypoints = [(msg.data[i], msg.data[i+1]) for i in range(0, len(msg.data), 2)]

    def yolo_human_callback(self, msg):
        self.detect_human = msg.data

    def yolo_stopsign_callback(self, msg):
        self.detect_stopsign = msg.data

    def yolo_cone_callback(self, msg):
        self.detect_cone = msg.data

    def longititudal_controller(self):
        curr_x, curr_y = self.waypoints[-1]

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
        wp2_x,   wp2_y = self.waypoints[-4]
        wp3_x,   wp3_y = self.waypoints[-5]
        wp4_x,   wp4_y = self.waypoints[0]

        # print("0", curr_x, curr_y)
        # print("1", wp0_x, wp0_y)
        # print("2", wp1_x, wp1_y)
        # print("3", wp2_x, wp2_y)
        # print("4", wp3_x, wp3_y)

        # 大きく曲がっているかを、curr, 1, 3から判定
        # 差を見る
        print("wp1_x - curr_x", wp1_x - curr_x)
        print("wp4_x - wp1_x", wp4_x - wp1_x)

        # Look at nearest point for steep curve
        vec0_x, vec0_y = wp0_x - curr_x, -(wp0_y - curr_y)
        vec2_x, vec2_y = wp2_x - curr_x, -(wp2_y - curr_y)

        # # 1st T turn - turn right
        # if wp1_x - curr_x < -0.25 and wp4_x - wp1_x > 0.25:
        #     print("1st T turn - turn right")
        #     ld1 = np.sqrt((wp0_x - curr_x)**2 + (wp0_y - curr_y)**2)
        #     alpha1 = -1.4
        # # 2nd T turn - turn left
        # elif wp1_x - curr_x > 0.25 and wp4_x - wp1_x < -0.25:
        #     print("2nd T turn - turn left")
        #     ld1 = np.sqrt((wp0_x - curr_x)**2 + (wp0_y - curr_y)**2)
        #     alpha1 = 1.4
        # else:

        #     if abs(vec2_x / vec2_y) > 0.1:
        #         ld1 = np.sqrt((wp2_x - curr_x)**2 + (wp2_y - curr_y)**2)
        #         angle_curr_to_wp  = np.arctan2(vec2_y, vec2_x)
        #         alpha1 = angle_curr_to_wp - np.pi/2

        #         alpha1 = alpha1 * 1.0 ### CHECK THIS PARAMETER
        #         # print("a", alpha1)

        #     else:
        #         # vec2_x = vec2_x * 0.5
        #         angle_curr_to_wp  = np.arctan2(vec2_y, vec2_x)
        #         ld1 = np.sqrt((wp1_x - curr_x)**2 + (wp1_y - curr_y)**2)
        #         alpha1 = angle_curr_to_wp - np.pi/2

        #         alpha1 = alpha1 * 0.5 ### CHECK THIS PARAMETER


        if abs(vec2_x / vec2_y) > 0.1:
            ld1 = np.sqrt((wp2_x - curr_x)**2 + (wp2_y - curr_y)**2)
            angle_curr_to_wp  = np.arctan2(vec2_y, vec2_x)
            alpha1 = angle_curr_to_wp - np.pi/2

            alpha1 = alpha1 * 1.0 ### CHECK THIS PARAMETER
            # print("a", alpha1)

        else:
            # vec2_x = vec2_x * 0.5
            angle_curr_to_wp  = np.arctan2(vec2_y, vec2_x)
            ld1 = np.sqrt((wp1_x - curr_x)**2 + (wp1_y - curr_y)**2)
            alpha1 = angle_curr_to_wp - np.pi/2

            alpha1 = alpha1 * 0.5 ### CHECK THIS PARAMETER



        # # Look at nearest point for steep curve
        # vec2_x, vec2_y = wp0_x - curr_x, -(wp0_y - curr_y)
        # angle_curr_to_wp0  = np.arctan2(vec2_y, vec2_x)

        # ld1 = np.sqrt((wp0_x - curr_x)**2 + (wp0_y - curr_y)**2)
        # alpha1 = angle_curr_to_wp0 - np.pi/2


        # # Look at nearest point for steep curve
        # vec2_x, vec2_y = wp1_x - curr_x, -(wp1_y - curr_y)
        # angle_curr_to_wp1  = np.arctan2(vec2_y, vec2_x)

        # ld1 = np.sqrt((wp1_x - curr_x)**2 + (wp1_y - curr_y)**2)
        # alpha1 = angle_curr_to_wp1 - np.pi/2


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
            self.pacmod_enable = True
            if(self.pacmod_enable == True):
                if (self.gem_enable == False):

                    # ---------- Enable PACMod ----------

                    # enable forward gear
                    self.gear_cmd.ui16_cmd = 3

                    # enable brake
                    self.brake_cmd.enable  = True
                    self.brake_cmd.clear   = False
                    self.brake_cmd.ignore  = False
                    self.brake_cmd.f64_cmd = 0.0

                    # enable gas 
                    self.accel_cmd.enable  = True
                    self.accel_cmd.clear   = False
                    self.accel_cmd.ignore  = False
                    self.accel_cmd.f64_cmd = 0.0

                    self.gear_pub.publish(self.gear_cmd)
                    print("Foward Engaged!")

                    self.turn_pub.publish(self.turn_cmd)
                    print("Turn Signal Ready!")
                    
                    self.brake_pub.publish(self.brake_cmd)
                    print("Brake Engaged!")

                    self.accel_pub.publish(self.accel_cmd)
                    print("Gas Engaged!")

                    self.gem_enable = True

                else: 

                    ##### Emergency Stop for Human Detection #####
                    if self.detect_human == True:
                        print("Human Detected! Stopping the vehicle")
                        # Stop while detecting human
                        while self.detect_human == True:
                            self.detect_human = False
                            ######
                            throttle_percent = 0.0
                            self.prev_accel = 0.0
                            # self.accel_cmd.f64_cmd = throttle_percent
                            # self.accel_pub.publish(self.accel_cmd)
                            self.brake_cmd.f64_cmd = 0.6
                            self.brake_cmd.enable = True
                            self.brake_pub.publish(self.brake_cmd)
                            ######
                            time.sleep(2)
                            print("Waiting for the human to move away")
                            # Break when detect_human is still False
                        self.brake_cmd.f64_cmd = 0
                        self.brake_pub.publish(self.brake_cmd)
                        self.brake_cmd.enable = False
                        print("Restarting the controller")
                    ##############################################

                    else:

                        ##### Temporary Stop for Stop Sign Detection #####
                        if self.detect_stopsign == True:
                            # Initially stop for 5 sec
                            if self.ignore_init == True:
                                print("Stop Sign Detected! Stopping the vehicle")
                                # Coundown
                                for i in range(5, 0, -1):
                                    ######
                                    throttle_percent = 0.0
                                    self.prev_accel = 0.0
                                    # self.accel_cmd.f64_cmd = throttle_percent
                                    # self.accel_pub.publish(self.accel_cmd)
                                    self.brake_cmd.f64_cmd = 0.6
                                    self.brake_cmd.enable = True
                                    self.brake_pub.publish(self.brake_cmd)
                                    ######
                                    print(i)
                                    time.sleep(1)
                                self.brake_cmd.f64_cmd = 0
                                self.brake_pub.publish(self.brake_cmd)
                                self.brake_cmd.enable = False
                                self.ignore_timer = self.ignore_waitsec
                                self.ignore_init = False
                                print("Restarting the controller")
                            # Ignore stop sign for waitsec
                            elif self.ignore_timer > 0.01:
                                # Ignore stop sign for waitsec
                                self.ignore_timer -= 1/ self.hz
                                # print("### Ignoring Stop Sign for", round(self.ignore_timer, 1), "sec ###")
                            # Init the timer and restart
                            else:
                                self.detect_stopsign = False
                                self.ignore_init = True
                        ##################################################

                        ##### Temporary Stop for Cone Detection #####
                        if self.detect_cone == True:
                            # Initially stop for 5 sec
                            if self.ignore_init == True:
                                print("Cone Detected! Stopping the vehicle")
                                # Coundown
                                for i in range(3, 0, -1):
                                    ######
                                    throttle_percent = 0.0
                                    self.prev_accel = 0.0
                                    self.brake_cmd.f64_cmd = 0.6
                                    self.brake_cmd.enable = True
                                    self.brake_pub.publish(self.brake_cmd)
                                    ######
                                    print(i)
                                    time.sleep(1)
                                self.brake_cmd.f64_cmd = 0
                                self.brake_pub.publish(self.brake_cmd)
                                self.brake_cmd.enable = False
                                self.ignore_timer = self.ignore_waitsec
                                self.ignore_init = False
                                print("Restarting the controller")
                            # Ignore stop sign for waitsec
                            elif self.ignore_timer > 0.01:
                                # Ignore stop sign for waitsec
                                self.ignore_timer -= 1/ self.hz
                            # Init the timer and restart
                            else:
                                self.detect_cone = False
                                self.ignore_init = True
                        ##################################################

                        target_velocity = self.longititudal_controller()
                        target_steering = self.pure_pursuit_lateral_controller()


                        ##### Steering Conversion #####
                        # # Heading angle [rad] to Steering wheel[deg](-630 to 630 deg)
                        target_steering = np.degrees(target_steering)

                        # print("+ wheel angle[deg]", target_steering)

                        target_steering = self.front2steer(target_steering)

                        ########## Increase the steering angle temporarily ###########
                        target_steering = target_steering * 2.0 ### CHECK THIS PARAMETER

                        # 300 deg => 600 deg
                        # 200 deg => 400 deg
                        # 100 deg => 150 deg
                        #  50 deg =>  63 deg
                        #  10 deg =>  11 deg
                        # diff = min(target_steering, target_steering ** 2 / 200)
                        # target_steering = target_steering + diff

                        # 200 deg => 332 deg
                        # 100 deg => 235 deg
                        #  50 deg => 139 deg
                        #  10 deg =>  30 deg
                        # target_steering = np.arctan(target_steering / 100) * 300
                        #############################################################

                        # When the steering angle is too large, the steering angle is limited to the maximum value.
                        print("+++ ts", target_steering)
                        if abs(target_steering - self.prev_steer) > 300\
                            and target_steering*self.prev_steer < 0:
                            target_steering = self.prev_steer

                                                                                             
                        # Set the initial steering angle to 0
                        if self.init_steer:
                            target_steering = 0.0
                            self.init_steer = False
                        # When the steering angle is large, the steering angle is limited to the maximum value.
                        else:
                            if target_steering - self.prev_steer > 90:
                                target_steering = self.prev_steer + 90
                            elif target_steering - self.prev_steer < -90:
                                target_steering = self.prev_steer - 90
                            else:
                                target_steering = target_steering

                        pub_steering = target_steering
                        self.prev_steer = target_steering
                        ###############################

                        ######################  T turn problem   ##############################################
                        self.t_turn[:-1] = self.t_turn[1:] 
                        self.t_turn[-1] = target_steering
                        count = 0
                        for i in range(5):
                            # if self.t_turn[i] < -200:
                            if abs(target_steering - self.t_turn[i]) > 200:
                                count = count+1
                            else:
                                count = count
                        print(f"count={count}")
                        if count == 5:
                            pub_steering = 0
                        ###############################

                        ##### Velocity Conversion #####
                        current_time = rospy.get_time()
                        filt_vel     = self.speed_filter.get_data(self.speed)
                        acc = self.pid_speed.get_control(current_time, self.desired_speed - filt_vel)

                        thresh_acc       = 0.325 ### CHECK THIS PARAMETER
                        min_acc          = 0.325 ### CHECK THIS PARAMETER
                        cruise_acc       = 0.300 ### CHECK THIS PARAMETER
                        
                        throttle_percent = 0.10 
                        acc = acc + 0.1 ### CHECK THIS PARAMETER
                        if filt_vel < target_velocity:
                            acc = acc + 0.1 ### CHECK THIS PARAMETER
                            # thresh_acc += 0.1
                            # min_acc    += 0.1
                            # cruise_acc += 0.1

                        if acc > self.max_accel:
                            throttle_percent = min(self.prev_accel + 0.005, self.max_accel)
                        elif acc >= thresh_acc:
                            if acc > self.prev_accel:
                                throttle_percent = self.prev_accel + 0.005
                            else:
                                throttle_percent = self.prev_accel - 0.005
                        elif acc < 0:
                            throttle_percent = max(self.prev_accel - 0.005, cruise_acc)
                        else:  # 0 < acc < thresh_acc
                            throttle_percent = max(self.prev_accel - 0.005, min_acc)
                        self.prev_accel = throttle_percent
                        ###############################

                        print("-----")
                        print("target_velocity[m/s]", round(target_velocity, 2))
                        print("target_steering[deg]", round(target_steering, 2))

                        print("- current_speed   ", round(float(filt_vel), 3))
                        print("- current_accel   ", round(float(acc), 3))
                        print("- throttle_percent", round(float(throttle_percent), 3))

                        if (target_steering <= 45 and target_steering >= -45):
                            self.turn_cmd.ui16_cmd = 1
                        elif(target_steering > 45):
                            self.turn_cmd.ui16_cmd = 2 # turn left
                        else:
                            self.turn_cmd.ui16_cmd = 0 # turn right

                        self.accel_cmd.f64_cmd = throttle_percent
                        self.steer_cmd.angular_position = np.radians(pub_steering)
    
                        self.accel_pub.publish(self.accel_cmd)
                        self.steer_pub.publish(self.steer_cmd)
                        self.turn_pub.publish(self.turn_cmd)
                    

def pure_pursuit():

    rospy.init_node('gem_controller', anonymous=True)
    vc = vehicleController()

    try:
        vc.execute()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    pure_pursuit()

