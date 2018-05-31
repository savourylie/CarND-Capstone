#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
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

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.waypoints = None
        self.prev_idx = 0

        rospy.spin()

    def pose_cb(self, msg):
        if not self.waypoints:
            pass
        # Compute yaw angle from vehicle pose
        yaw = tf.transformations.euler_from_quaternion([
            msg.pose.orientation.x, 
            msg.pose.orientation.y, 
            msg.pose.orientation.z, 
            msg.pose.orientation.w
            ])[2]
        # Find next waypoint, starting from the index of last waypoint for efficiency
        num_wps = len(self.waypoints)
        for i in range(self.prev_idx, (self.prev_idx + num_wps)):
            idx = i % num_wps
            wp = self.waypoints[idx]
            # If the dot product of yaw vector and vector from vehicle to waypoint is positive
            # the waypoint is in the same direction as the vehicle. Use the first one encountered
            if math.cos(yaw) * (wp.pose.pose.position.x - msg.pose.position.x) + \
                math.sin(yaw) * (wp.pose.pose.position.y - msg.pose.position.y) > 0:
                self.prev_idx = idx
                break

        final_waypoints = Lane()
        # Get next LOOKAHEAD_WPS waypoints after the next waypoint
        final_waypoints.waypoints = [self.waypoints[i % num_wps] for i in range(self.prev_idx, self.prev_idx + LOOKAHEAD_WPS)]
        self.final_waypoints_pub.publish(final_waypoints)

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

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
