# ECE484-FA24-CTJ-AutoGem_sim

### Usage

1. Clone the whole repository

   ```

   ```
2. Go to current workspace

   ```
   cd catkin_ws
   ```
3. Build the project

   ```
   catkin_make
   ```
4. Run Project Simulation

   - Terminal #1

     ```
     source devel/setup.bash
     roslaunch gem_launch gem_init.launch world_name:="highbay_track.world" x:=-1.5 y:=-21 yaw:=3.1416
     ```
   - Terminal #2

     ```
     source devel/setup.bash
     rosrun teleop_twist_keyboard teleop_twist_keyboard.py
     ```
   - Terminal #3

     ```
     python3 velcmd_to_ackermann.py
     ```

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
