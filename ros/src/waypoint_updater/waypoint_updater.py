#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
import tf
import operator
import math

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
        self.last_velocity = 0.
        self.last_time = rospy.get_time()
        self.waypoints = None
        self.pose = None
        self.prev_idx = 0
        self.traffic_idx = -1
        self.max_speed = (rospy.get_param('/waypoint_loader/velocity') * 1000.) / (60. * 60.)
        self.max_acc = 10.
        self.min_acc = 1.
        self.max_jerk = 10.
        self.min_jerk = 1.

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
                rospy.logwarn(["%.2f" % wp.twist.twist.linear.x for wp in final_waypoints.waypoints])
                self.final_waypoints_pub.publish(final_waypoints)
            rate.sleep()

    def generate_lane(self):
        lane = Lane()
        base_waypoints = self.waypoints[self.prev_idx : self.prev_idx + LOOKAHEAD_WPS]

        
        if self.traffic_idx > -1 and self.traffic_idx > self.prev_idx and self.traffic_idx < self.prev_idx + LOOKAHEAD_WPS:
            lane.waypoints = self.simple_decelerate_waypoints(base_waypoints)
        else:
            lane.waypoints = base_waypoints
            #lane.waypoints = self.simple_accelerate_waypoints(base_waypoints)
        '''
        current_vel = self.current_velocity.linear.x
        now = rospy.get_time()
        t_last_seg = now - self.last_time
        self.last_time = now
        current_acc = (current_vel - self.last_velocity) / t_last_seg
        self.last_velocity = current_vel

        if self.traffic_idx > -1 and self.traffic_idx > self.prev_idx and self.traffic_idx < self.prev_idx + LOOKAHEAD_WPS:
            dist = self.distance(self.waypoints, self.prev_idx, self.traffic_idx - 2)

            s1, s2, s3, v1_end, v2_end = self.get_status_change_points(current_vel, current_acc, self.max_jerk)

            if v1_end >= v2_end and dist > s1 + s2 + s3 and dist < current_vel**2 / (2 * self.min_acc):
                lane.waypoints = self.decelerate_waypoints(base_waypoints, current_acc, current_vel, dist)
            else:
                lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.accelerate_waypoints(base_waypoints, current_acc, current_vel)
        '''
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

    def simple_accelerate_waypoints(self, waypoints):
        temp = []
        current_vel = self.current_velocity.linear.x
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            dist = self.distance(waypoints, 0, max(1,i))
            vel = math.sqrt(current_vel**2 + 2 * self.min_acc * dist)
            if i == 0:
                vel /= 2
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)
        return temp


    def decelerate_waypoints(self, waypoints, current_acc, current_vel, dist):
        jerk_dict = {j: self.get_status_change_points(current_vel, current_acc, j) for j in range(2, 11, 2)}
        jerk_dict_filt = {j: (s1, s2, s3, v1_end, v2_end) for j, (s1, s2, s3, v1_end, v2_end) in jerk_dict.iteritems() if v1_end >= v2_end and dist > s1 + s2 + s3}
        jerk = min(jerk_dict_filt.iteritems(), key=itemgetter(0))
        s1, s2, s3, v1_end, v2_end = jerk_dict_filt[jerk]

        temp = []
        positive_jerk = True
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            stop_idx = max(self.traffic_idx - self.prev_idx - 2, 0)
            if i == 0:
                p.twist.twist.linear.x = wp.twist.twist.linear.x
            else:
                dsum = self.distance(waypoints, 0, i)
                d = self.distance(waypoints, i - 1, i)
                t = d / current_vel
                if dsum < s1:
                    current_vel += current_acc * t - jerk * t**2 / 2
                    current_acc -= jerk * t
                elif dsum < s1 + s2:
                    current_vel += current_acc * t
                else:
                    current_vel += current_acc * t + jerk * t**2 / 2
                    current_acc += jerk * t

                p.twist.twist.linear.x = max(0, min(current_vel, wp.twist.twist.linear.x))

            temp.append(p)
        return temp

    def accelerate_waypoints(self, waypoints, current_acc, current_vel):
        temp = []
        positive_jerk = True
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            dist = self.distance(waypoints, max(0,i-1), max(1,i))
            if current_vel < 1.:
                t = (6 * dist / self.min_jerk)**(1/3)
            else:
                t = dist / current_vel
            if positive_jerk:
                current_vel += current_acc * t + self.min_jerk * t**2 / 2
                current_acc += self.min_jerk * t
                p.twist.twist.linear.x = min(current_vel, wp.twist.twist.linear.x)
                if current_acc > self.min_acc or current_vel > self.max_speed - self.min_jerk * (current_acc / self.min_jerk)**2 / 2:
                    positive_jerk = False
            else:
                current_vel += current_acc * t - self.min_jerk * t**2 / 2
                current_acc -= self.min_jerk * t
                p.twist.twist.linear.x = min(current_vel, wp.twist.twist.linear.x)
                if current_acc < 0.:
                    p.twist.twist.linear.x = wp.twist.twist.linear.x

            temp.append(p)
        return temp

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints

    def velocity_cb(self, msg):
        self.current_velocity = msg.twist

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
