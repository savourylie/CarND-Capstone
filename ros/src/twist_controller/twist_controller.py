import rospy
from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter
import time

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, brake_deadband, decel_limit, accel_limit, wheel_radius, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        self.vehicle_mass = vehicle_mass
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        Ku = 0.5
        Pu = 6
        Kp = 0.6 * Ku
        Ki = 2 * Kp / Pu
        Kd = Kp * Pu / 8
        mn = 0.
        mx = 0.4
        self.pid_controller = PID(Kp, Ki, Kd, mn, mx)

        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

        tau = 0.5
        ts = 0.02
        self.velocity_lowpass = LowPassFilter(tau, ts)

        self.last_vel = 0.
        self.timestamp = rospy.get_time()

    def control(self, target_twist, current_velocity, dbw_enabled):
        if not dbw_enabled:
            self.pid_controller.reset()
            return 0., 0., 0.

        current_timestamp = rospy.get_time()
        dt = current_timestamp - self.timestamp
        self.timestamp = current_timestamp

        current_velocity = self.velocity_lowpass.filt(current_velocity.linear.x)
        target_linear_velocity = target_twist.linear.x
        vel_diff = target_linear_velocity - current_velocity

        throttle = self.pid_controller.step(vel_diff, dt)
        brake = 0.

        if target_linear_velocity == 0. and current_velocity < 0.1:
            throttle = 0.
            brake = 700.
        elif throttle < 0.1 and vel_diff < 0:
            throttle = 0.
            brake = abs(max(vel_diff / dt, self.decel_limit)) * self.vehicle_mass * self.wheel_radius

        steer = self.yaw_controller.get_steering(target_linear_velocity, target_twist.angular.z, current_velocity)
        # Return throttle, brake, steer
        return throttle, brake, steer
