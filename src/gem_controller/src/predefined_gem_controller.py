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
from pacmod_msgs.msg import PositionWithSpeed, PacmodCmd, SystemRptFloat, VehicleSpeedRpt


class vehicleController():

    def __init__(self):
        self.rate = rospy.Rate(10)

        self.L = 1.75  # Wheelbase, can be get from gem_control.py

        # PID for longitudinal control
        self.desired_speed = 0.6  # m/s
        self.max_accel = 0.48  # % of acceleration

        self.speed_sub  = rospy.Subscriber("/pacmod/parsed_tx/vehicle_speed_rpt", VehicleSpeedRpt, self.speed_callback)
        self.speed      = 0.0

        self.steer_sub = rospy.Subscriber("/pacmod/parsed_tx/steer_rpt", SystemRptFloat, self.steer_callback)
        self.steer = 0.0 # degrees

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

    # Get value of steering wheel
    def steer_callback(self, msg):
        self.steer = round(np.degrees(msg.output),1)


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
                    start_time = rospy.Time.now().to_sec()

                else: 
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


                    # # Heading angle [rad] to Steering wheel[deg](-630 to 630 deg)
                    target_steering = np.degrees(target_steering)
                    target_steering = self.front2steer(target_steering)

                    print("-----")
                    print("target_velocity[m/s]", round(target_velocity, 2))
                    print("target_steering[deg]", round(target_steering, 2))

                    # target_steering = np.radians(target_steering)

                    acc = target_velocity - self.speed

                    if acc > 0.64 :
                        throttle_percent = self.max_accel
                    elif acc < 0.0 :
                        throttle_percent = 0.0
                    else:
                        throttle_percent = acc / 0.64 * self.max_accel  

                    if (target_steering <= 45 and target_steering >= -45):
                        self.turn_cmd.ui16_cmd = 1
                    elif(target_steering > 45):
                        self.turn_cmd.ui16_cmd = 2 # turn left
                    else:
                        self.turn_cmd.ui16_cmd = 0 # turn right

                    self.accel_cmd.f64_cmd = throttle_percent
                    self.steer_cmd.angular_position = np.radians(target_steering)
  
                    self.accel_pub.publish(self.accel_cmd)
                    self.steer_pub.publish(self.steer_cmd)
                    
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

