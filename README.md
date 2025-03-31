# Autonomous Driving Software for GEM Electric Vehicle

### Usage

1. Clone the whole repository

   ```
   git clone https://github.com/htliang517/ECE484-FA24-CTJ-AutoGem_sim.git
   ```
2. Go to current workspace

   ```
   cd ECE484-FA24-CTJ-AutoGem_sim
   ```
3. Build the project

   ```
   catkin_make
   ```
4. Run Project Simulation

   - Terminal #1 Launch Gazebo Simulation

     ```
     source devel/setup.bash
     roslaunch gem_launch gem_init.launch world_name:="highbay_track.world" x:=-1.5 y:=-21 yaw:=3.1416
     ```
   - Terminal #2 Keyboard Controller

     ```
     source devel/setup.bash
     rosrun teleop_twist_keyboard teleop_twist_keyboard.py
     ```
   - Terminal #3 Node for Comunicating between Topics (#Will be modified to a launch file in the future)

     ```
     python3 velcmd_to_ackermann.py
     ```
   - Terminal #4 Lane Detection Testing (#Will be modified to a launch file in the future)
     (If you have problem with this line, check with `chmod +x studentVision.py`)

     ```
     source devel/setup.bash
     rosrun gem_lane_detection studentVision.py
     ```



### Test 2024-11-16

1. Run Vision and controller in Gazebo simulation

   - Terminal #1 Launch Gazebo Simulation

     ```
     source devel/setup.bash
     roslaunch gem_launch gem_init.launch world_name:="highbay_track.world" x:=-1.5 y:=-21 yaw:=3.1416
     ```
   - Terminal #2 studentVision_gazebo

     ```
     source devel/setup.bash
     rosrun gem_lane_detection studentVision_gazebo.py
     ```
   - Terminal #3 controller

     ```
     source devel/setup.bash
     rosrun gem_controller controller.py
     ```
   
   - Terminal #4 Run when you reset the gem position
     ```
     source devel/setup.bash
     rosrun gem_controller set_pos.py --x -1.5 --y -21 --yaw 3.1416

     ```


2. Run Vision and controller in GEM

   - Terminal #1 studentVision

     ```
     source devel/setup.bash
     rosrun gem_lane_detection studentVision.py
     ```
   - Terminal #2 gem_controller

     ```
     source devel/setup.bash
     rosrun gem_controller gem_controller.py
     ```
   

3. Run controller without Vision

    - Terminal in Gazebo: Run predefined_controller (when you do not use controller.py)
     ```
     source devel/setup.bash
     rosrun gem_controller predefined_controller.py

     ```


    - Terminal in GEM: Run predefined_gem_controller (when you do not use gem_controller)
     ```
     source devel/setup.bash
     rosrun gem_controller predefined_gem_controller.py

     ```

### Packages Description

* gem_controller
  Currently stores the velcmd_to_ackermann python file only.
  #Might combine with control code in the future.
* gem_lane_detection
  Contain main code of lane detection.
* gem_simulator
  Cloned from "https://github.com/hangcui1201/POLARIS_GEM_e2_Simulator"
* teleop_twist_keyboard
  Cloned from "https://github.com/ros-teleop/teleop_twist_keyboard"
  Used for keyboard control

### Debug

```bash
CMake Error at /opt/ros/noetic/share/catkin/cmake/catkinConfig.cmake:83 (find_package):  Could not find a package configuration file provided by "geographic_msgs"
  with any of the following names:    geographic_msgsConfig.cmake
    geographic_msgs-config.cmake
```

`<sol>`

```bash
sudo apt-get install ros-noetic-geographic-msgs
```
