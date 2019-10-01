from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)

        # initial pid control
        kp = 0.3
        ki = 0.1
        kd = 0.0
        # minimum/maximum throttle value
        mn = 0.0
        mx = 0.3

        self.throttle_controller = PID(kp, ki, kd, mn, mx)

        tau = 0.5 # cutoff frequency 
        ts = 0.02 # sample rate
        self.vel_lpf = LowPassFilter(tau, ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()

    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer

        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0.0, 0.0, 0.0

        # remove the noise from received current velocity
        current_vel = self.vel_lpf.filt(current_vel)

        # get the steering 
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)

        # velocity error 
        vel_error = linear_vel - current_vel # the speed we want to acheive - the speed we have now
        self.last_vel = current_vel # for pid controller use
        
        # time counter
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        # get the throttle value
        throttle = self.throttle_controller.step(vel_error, sample_time)

        brake = 0.0

        # stop the vehicle
        if (linear_vel == 0.0 and current_vel < 0.1):
            throttle = 0.0
            brake = 400 #N*m - acc=1m/s^2, 700 is for carla used
        
        # start to decelerate, vel_error < 0 - the desired speed is less than current speed, deceleration needed
        elif (throttle < 0.10 and vel_error < 0.0):
            throttle = 0.0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius # 1 N*m = 1 kg*m^2/s^2

        return throttle, brake, steering

        # this simple controller will work with a little wondering around the waypoints. The fix could be adjusting the waypoint follower 
        # and make it updating the waypoints all the time


