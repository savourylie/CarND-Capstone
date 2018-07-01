# Udacity Self-Driving Car Nanodegree Program -- System Integration
In the System Integration project we put together all the components done in the previous projects using ROS, to implement the core functionalities including perception (traffic light detection), planning (waypoint updater) and control (waypoint follower). The code is to be run on the Carla simulator and the real Carla soon.

## The Team
Calvin Ku	
[calvin.j.ku@googlemail.com](calvin.j.ku@googlemail.com)

Simi Ily	
[ilysimi@hotmail.com](ilysimi@hotmail.com)

Fanxing Meng
[fm2438@columbia.edu](fm2438@columbia.edu)

Lewis McCarthy	
[pseudonym@acm.org](pseudonym@acm.org)

Ying Li	
[liyingchocolate@gmail.com](liyingchocolate@gmail.com)

### Running it on the Simulator
To run the simulator, you'll need to have [ROS](http://wiki.ros.org/kinetic/Installation/), a few Python 2 dependencies installed. The [simulator](https://github.com/udacity/CarND-Capstone/releases/tag/v1.2) can be downloaded [here](https://github.com/udacity/CarND-Capstone/releases/tag/v1.2). Once you have the environment ready, follow the instructions as below:

1. Go to the `ros` directory

2. Run `catkin_make` 

3. Run `source devel/setup.sh`

4. Run `roslaunch launch/styx.launch`

5. Start up the simulator, and go to the highway option and untick the manual mode.