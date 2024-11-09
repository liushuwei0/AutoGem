import rospy
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDrive

class CmdVelSubscriber:
    def __init__(self):
        rospy.init_node('cmd_vel_to_ackermann', anonymous=True)
        
        # Subscribe to /cmd_vel topic
        self.controlSub = rospy.Subscriber('/cmd_vel', Twist, self.callback)
        
        # Publisher to /ackermann_cmd
        self.controlPub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=10)

    def callback(self, msg):
        # Extract linear and angular components from the Twist message
        linear_velocity = msg.linear.x
        angular_velocity = msg.angular.z

        # Debugging: Print received cmd_vel values
        rospy.loginfo(f"Received cmd_vel - Speed: {linear_velocity}, Turn: {angular_velocity}")

        # Pack velocity and steering angle into Ackermann command
        newAckermannCmd = AckermannDrive()
        newAckermannCmd.speed = linear_velocity
        newAckermannCmd.steering_angle = angular_velocity

        # Debugging: Print Ackermann command values before publishing
        rospy.loginfo(f"Publishing Ackermann - Speed: {newAckermannCmd.speed}, Steering Angle: {newAckermannCmd.steering_angle}")

        # Publish the Ackermann command
        self.controlPub.publish(newAckermannCmd)

if __name__ == '__main__':
    CmdVelSubscriber()
    rospy.spin()
