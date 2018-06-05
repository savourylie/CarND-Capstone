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

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Waypoint, self.obstacle_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
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
        self.min_jerk = .1

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.waypoints:
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
                self.final_waypoints_pub.publish(final_waypoints)
            rate.sleep()

    def generate_lane(self):
        lane = Lane()

        '''
        if self.traffic_idx > -1 and self.traffic_idx > self.prev_idx and self.traffic_idx < self.prev_idx + LOOKAHEAD_WPS:
            lane.waypoints = self.simple_decelerate_waypoints(base_waypoints)
        else:
            lane.waypoints = base_waypoints
            #lane.waypoints = self.simple_accelerate_waypoints(base_waypoints)
        '''
        current_vel = self.current_velocity.linear.x
        current_acc = 0. if self.current_acceleration == None else copy.copy(self.current_acceleration)

        rospy.logwarn("%d %d" % (self.traffic_idx, self.prev_idx))
        if self.traffic_idx > -1 and current_vel > 1. and self.traffic_idx > self.prev_idx and self.traffic_idx < self.prev_idx + LOOKAHEAD_WPS:
            dist = self.distance(self.waypoints, self.prev_idx, self.traffic_idx - 2)
            max_distance = current_vel**2 / (2 * self.min_acc)
            rospy.logwarn(", %.0f, %.0f" % (dist, max_distance))
            if dist < max_distance:
                rospy.logwarn("Decelerate")
                jerk_dict = {j/10.: self.get_status_change_points(current_vel, current_acc, j/10.) for j in range(1, 100)}
                jerk_dict_filt = {j: (s1, s2, s3, v1_end, v2_end) for j, (s1, s2, s3, v1_end, v2_end) in jerk_dict.iteritems() if v1_end >= v2_end and dist > s1 + s2 + s3}
                if len(jerk_dict_filt) > 0:
                    jerk = min(jerk_dict_filt.iteritems(), key=operator.itemgetter(0))
                    rospy.logwarn("%.0f, %.0f" % (dist, jerk[1][0] + jerk[1][1] + jerk[1][2]))
                    lane.waypoints = self.decelerate_waypoints(current_acc, current_vel, dist, jerk)
                    return lane
            '''
            else:
                rospy.logwarn("Const")
                #lane.waypoints = base_waypoints
                lane.waypoints = self.accelerate_waypoints(base_waypoints, current_acc, current_vel)
        else:
        '''
        rospy.logwarn("Accelerate")
        lane.waypoints = self.accelerate_waypoints(current_acc, current_vel)
        rospy.logwarn(["%.2f" % wp.twist.twist.linear.x for wp in lane.waypoints])
        
        return lane

    def simple_decelerate_waypoints(self, waypoints):
        temp = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            stop_idx = max(self.traffic_idx - self.prev_idx - 2, 0)
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2 * self.max_acc * dist)
            if vel < 1.:
                vel = 0.
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)
        return temp

    def decelerate_waypoints(self, current_acc, current_vel, dist, jerk):
        s1, s2, s3, v1_end, v2_end = jerk[1]

        stopped = False
        for i in range(LOOKAHEAD_WPS):
            abs_idx = i + self.prev_idx
            if stopped == True:
                self.waypoints_cache[abs_idx].twist.twist.linear.x = 0.
            else:
                ref_vel = self.waypoints[i + self.prev_idx].twist.twist.linear.x
                dsum = self.distance(self.waypoints, self.prev_idx - 1, abs_idx)
                d = self.distance(self.waypoints, abs_idx - 1, abs_idx)
                if dsum < s1:
                    coeff = [-jerk[0] / 6., current_acc / 2., current_vel, -d]
                    r = np.roots(coeff)
                    t = r.real[abs(r.imag)<1e-5][0]
                    current_vel += current_acc * t - jerk[0] * t**2 / 2
                    current_acc -= jerk[0] * t
                elif dsum < s1 + s2:
                    t = (-current_vel - math.sqrt(current_vel**2 + 2 * current_acc * d)) / current_acc
                    current_vel += current_acc * t
                elif dsum < s1 + s2 + s3:
                    coeff = [jerk[0] / 6., current_acc / 2., current_vel, -d]
                    r = np.roots(coeff)
                    t = r.real[abs(r.imag)<1e-5][0]
                    current_vel += current_acc * t + jerk[0] * t**2 / 2
                    current_acc += jerk[0] * t
                if current_vel < 0.:
                    stopped = True
                rospy.logwarn("%.2f %.2f" % (current_vel, current_acc))
                self.waypoints_cache[abs_idx].twist.twist.linear.x = min(max(0,current_vel), ref_vel)

        return self.waypoints_cache[self.prev_idx : self.prev_idx + LOOKAHEAD_WPS]

    def accelerate_waypoints(self, current_acc, current_vel):
        # do not modify if already generated
        if current_vel > 1e-3 and all(1e-3 < x.twist.twist.linear.x < y.twist.twist.linear.x + 1e-3 for x, y in zip(self.waypoints_cache[self.prev_idx:self.prev_idx+LOOKAHEAD_WPS], self.waypoints_cache[self.prev_idx+1:self.prev_idx+LOOKAHEAD_WPS+1])):
            return self.waypoints_cache[self.prev_idx : self.prev_idx + LOOKAHEAD_WPS]
        # generate new cache if stopped, or has decelerating waypoint cache in front
        positive_jerk = True
        for i in range(LOOKAHEAD_WPS):
            abs_idx = i + self.prev_idx
            ref_vel = self.waypoints[abs_idx].twist.twist.linear.x
            # set to terminal velocity if not accelerating
            if positive_jerk == None:
                self.waypoints_cache[abs_idx].twist.twist.linear.x = ref_vel
            else:
                # solve cubic polynomial involving jerk for duration between waypoints
                dist = self.distance(self.waypoints, abs_idx - 1, abs_idx)
                jerk = self.min_jerk if positive_jerk == True else -self.min_jerk
                coeff = [jerk / 6., current_acc / 2., current_vel, -dist]
                r = np.roots(coeff)
                t = r.real[abs(r.imag)<1e-5][0]
                # calculate new velocity and acceleration at next waypoint
                current_vel += current_acc * t + jerk * t**2 / 2
                current_acc += jerk * t
                self.waypoints_cache[abs_idx].twist.twist.linear.x = min(max(0,current_vel), ref_vel)
                # slow down acceleration after having achieved half of speed limit
                if positive_jerk == True:
                    if current_vel > (self.max_speed + current_vel) / 2.05:
                        positive_jerk = False
                # stop accelerating and use terminal velocity if acceleration has decreased to 0
                elif positive_jerk == False:
                    if current_acc < 0.:
                        positive_jerk = None
                        self.waypoints_cache[abs_idx].twist.twist.linear.x = ref_vel
        # return cached waypoints
        return self.waypoints_cache[self.prev_idx : self.prev_idx + LOOKAHEAD_WPS]

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints
        self.waypoints_cache = self.waypoints[:]

    def velocity_cb(self, msg):
        if not self.current_velocity == None:
            old_vel = self.current_velocity.linear.x
        self.current_velocity = msg.twist
        if not self.last_time == None:
            t = msg.header.stamp - self.last_time
            if t.to_sec() > 1e-3:
                self.current_acceleration = (msg.twist.linear.x - old_vel) / t.to_sec()
        self.last_time = msg.header.stamp

    def get_status_change_points(self, current_vel, current_acc, jerk):
        # min_break_distance: jerk until max_decel, v steady decr, until jerk to 0 acc & 0 velocity
        t1 = (self.max_acc + current_acc) / jerk
        v1_end = current_vel + current_acc * t1 + jerk * t1**2 / 2
        s1 = max(0, current_vel * t1 + current_acc * t1**2 / 2 + jerk * t1**3 / 6)

        t3 = self.max_acc / jerk
        v2_end = jerk * t3**2 / 2 
        s3 = jerk * t3**3 / 6

        t2 = max(0, (v1_end - v2_end) / self.max_acc)
        s2 = max(0, v2_end * t2 + self.max_acc * t2**2 / 2)

        return s1, s2, s3, v1_end, v2_end

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
