#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
import tf
import operator
import math
import numpy as np
import copy

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Waypoint, self.obstacle_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.current_velocity = None
        self.current_acceleration = None
        self.last_time = None
        self.waypoints = None
        self.waypoints_cache = None
        self.pose = None
        self.prev_idx = 0
        self.traffic_idx = -1
        self.max_speed = (rospy.get_param('/waypoint_loader/velocity') * 1000.) / (60. * 60.)
        self.max_acc = 10.
        self.min_acc = .1
        self.max_jerk = 10.
        self.min_jerk = .2

        self.loop()

    def loop(self):
        # TODO: Change to 20 if lag
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.pose and self.waypoints and self.waypoints_cache:
                # Compute yaw angle from vehicle pose
                yaw = tf.transformations.euler_from_quaternion([
                    self.pose.pose.orientation.x, 
                    self.pose.pose.orientation.y, 
                    self.pose.pose.orientation.z, 
                    self.pose.pose.orientation.w
                    ])[2]
                # Find next waypoint, starting from the index of last waypoint for efficiency
                num_wps = len(self.waypoints)
                for i in range(self.prev_idx, (self.prev_idx + num_wps)):
                    idx = i % num_wps
                    wp = self.waypoints[idx]
                    # If the dot product of yaw vector and vector from vehicle to waypoint is positive
                    # the waypoint is in the same direction as the vehicle. Use the first one encountered
                    if math.cos(yaw) * (wp.pose.pose.position.x - self.pose.pose.position.x) + \
                        math.sin(yaw) * (wp.pose.pose.position.y - self.pose.pose.position.y) > 0:
                        self.prev_idx = idx
                        break

                final_waypoints = self.generate_lane()
                #rospy.logwarn([wp.twist.twist.linear.x for wp in self.waypoints_cache[self.prev_idx : self.prev_idx + LOOKAHEAD_WPS]])
                self.final_waypoints_pub.publish(final_waypoints)
            rate.sleep()

    def generate_lane(self):
        lane = Lane()
        # Get a copy of current velocity and acceleration since they will be modified
        current_vel = copy.copy(self.current_velocity.linear.x)
        current_acc = 0. if self.current_acceleration == None else copy.copy(self.current_acceleration)
        # When red light is in front
        if self.traffic_idx > -1 and self.traffic_idx - 2 > self.prev_idx and self.traffic_idx < self.prev_idx + LOOKAHEAD_WPS:
            # Do not modify if decelerating waypoint cache has already been generated
            # To avoid creeping, don't modify if already close to stop line
            if all(self.max_speed > x.twist.twist.linear.x > y.twist.twist.linear.x - 1e-3 for x, y in zip(self.waypoints_cache[self.prev_idx:self.traffic_idx-1], self.waypoints_cache[self.prev_idx+1:self.traffic_idx])) or self.traffic_idx - self.prev_idx < 10:
                lane.waypoints = self.waypoints_cache[self.prev_idx : self.prev_idx + LOOKAHEAD_WPS]
                return lane

            # Formula for deceleration with constant jerk (with sign flip and no constant acceleration part): 
            # S = s1 + s2, differentiated by the sign of jerk, each part taking the the time: T = 2 * t
            # v_mid = v_entry / 2 = a_max * t
            # a_max = j * t
            # v_entry = j * t^2
            # S = v_entry^2 / a_max = v_entry^2 / (j*t) = j * t^3
            # v_entry = j * t^2
            # t = S / v_entry
            # j = v_entry^3 / S^2
            dist = self.distance(self.waypoints, self.prev_idx, self.traffic_idx - 3)
            jerk = current_vel**3 / dist**2
            t_half = dist / current_vel
            a_max = jerk * t_half
            # Only generate deceleration waypoints if close enough to red light (implying meaningful jerk and deceleration)
            # Don't decelerate if too close (should be yellow light in this scenario; stopping needs high jerk and deceleration)
            if jerk > self.max_speed / 30. and jerk < self.max_jerk and a_max < self.max_acc:
                lane.waypoints = self.decelerate_waypoints(current_acc, current_vel, dist, jerk, t_half)
                return lane

        # For all other scenario, just accelerate or go at speed limit
        lane.waypoints = self.accelerate_waypoints(current_acc, current_vel)
        
        return lane

    def decelerate_waypoints(self, current_acc, current_vel, dist, jerk, t_half):
        stopped = False
        s2 = jerk * t_half**3 / 6.
        s1 = dist - s2
        for i in range(LOOKAHEAD_WPS):
            abs_idx = (i + self.prev_idx) % len(self.waypoints)
            # Set velocity to 0 for all waypoints after having stopped
            if stopped == True:
                self.waypoints_cache[abs_idx].twist.twist.linear.x = 0.
            else:
                dsum = self.distance(self.waypoints, self.prev_idx - 1, abs_idx)
                d = self.distance(self.waypoints, abs_idx, abs_idx + 1)
                # First part of deceleration with negative jerk, solve cubic polynomial for t during each segment
                if dsum < s1 + 1e-3:
                    coeff = [-jerk / 6., current_acc / 2., current_vel, -d]
                    r = np.roots(coeff)
                    t = min([rr for rr in r.real[abs(r.imag)<1e-5] if rr > 0])
                    current_vel += current_acc * t - jerk * t**2 / 2
                    current_acc -= jerk * t
                # Second part of deceleration with positive jerk, also need to solve for t
                elif dsum < dist + 1e-3:
                    coeff = [jerk / 6., current_acc / 2., current_vel, -d]
                    r = np.roots(coeff)
                    t = min([rr for rr in r.real[abs(r.imag)<1e-5] if rr > 0])
                    current_vel += current_acc * t + jerk * t**2 / 2
                    current_acc += jerk * t
                # Should be stopped after the specified distance
                else:
                    stopped = True
                    current_vel = 0.
                if current_vel < 0.:
                    stopped = True
                ref_vel = self.waypoints[abs_idx].twist.twist.linear.x
                # Limit speed
                self.waypoints_cache[abs_idx].twist.twist.linear.x = min(max(0.,current_vel), ref_vel)

        return self.waypoints_cache[self.prev_idx : self.prev_idx + LOOKAHEAD_WPS]

    def accelerate_waypoints(self, current_acc, current_vel):
        # Do not modify if accelerating waypoint cache has already been generated
        if current_vel > 1e-2 and all(1e-2 < x.twist.twist.linear.x < y.twist.twist.linear.x + 1e-3 for x, y in zip(self.waypoints_cache[self.prev_idx:self.prev_idx+LOOKAHEAD_WPS], self.waypoints_cache[self.prev_idx+1:self.prev_idx+LOOKAHEAD_WPS+1])):
            return self.waypoints_cache[self.prev_idx : self.prev_idx + LOOKAHEAD_WPS]
        # Generate new cache if stopped, or has decelerating waypoint cache in front
        positive_jerk = True
        for i in range(LOOKAHEAD_WPS):
            abs_idx = (i + self.prev_idx) % len(self.waypoints)
            ref_vel = self.waypoints[abs_idx].twist.twist.linear.x
            # Set to terminal velocity if done accelerating
            if positive_jerk == None:
                self.waypoints_cache[abs_idx].twist.twist.linear.x = ref_vel
            else:
                # Solve cubic polynomial involving jerk for duration between waypoints
                dist = self.distance(self.waypoints, abs_idx - 1, abs_idx)
                jerk = self.min_jerk if positive_jerk == True else -self.min_jerk
                coeff = [jerk / 6., current_acc / 2., current_vel, -dist]
                r = np.roots(coeff)
                t = min([rr for rr in r.real[abs(r.imag)<1e-5] if rr > 0])
                # Calculate new velocity and acceleration at next waypoint
                current_vel += current_acc * t + jerk * t**2 / 2
                current_acc += jerk * t
                self.waypoints_cache[abs_idx].twist.twist.linear.x = min(max(0.,current_vel), ref_vel)
                # Turn down acceleration after having reached half of speed limit
                if positive_jerk == True:
                    if current_vel > (self.max_speed + current_vel) / 2.05:
                        positive_jerk = False
                # Stop accelerating and use terminal velocity if acceleration has decreased to 0
                elif positive_jerk == False:
                    if current_acc < 0.:
                        positive_jerk = None
                        self.waypoints_cache[abs_idx].twist.twist.linear.x = ref_vel
        # Return cached waypoints
        return self.waypoints_cache[self.prev_idx : self.prev_idx + LOOKAHEAD_WPS]

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # Save waypoints for reference
        self.waypoints = waypoints.waypoints
        # Make another copy for editing
        self.waypoints_cache = copy.deepcopy(self.waypoints)

    def velocity_cb(self, msg):
        # Use last velocity and time stamp to calculate current acceleration
        if not self.current_velocity == None:
            old_vel = self.current_velocity.linear.x
        self.current_velocity = msg.twist
        if not self.last_time == None:
            t = msg.header.stamp - self.last_time
            if t.to_sec() > 1e-3:
                self.current_acceleration = (msg.twist.linear.x - old_vel) / t.to_sec()
        self.last_time = msg.header.stamp

    def traffic_cb(self, msg):
        self.traffic_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
