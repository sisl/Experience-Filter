import carla
import time
from helper_funcs import *

import sys
sys.path.append('./PythonAPI/carla/')

from PythonAPI.carla.agents.navigation.local_planner import LocalPlanner
from PythonAPI.carla.agents.navigation.global_route_planner import GlobalRoutePlanner
from PythonAPI.carla.agents.tools.misc import get_speed, is_within_distance, get_trafficlight_trigger_location

# You must have connected to julia.api in the main script for the imports below to work!
from julia import MODIA
from julia import Base as jlBase


class MODIAAgent(object):
    """
    MODIAAgent creates multiple decision components from pre-solved decision problems.
    """

    def __init__(self, vehicle, init_belief, StopUncontrolledDP, target_speed=20, opt_dict={}):
        """
        Initialization the agent paramters, the local and the global planner.

            :param vehicle: actor to apply to agent logic onto
            :param target_speed: speed (in Km/h) at which the vehicle will move
            :param opt_dict: dictionary in case some of its parameters want to be changed.
                This also applies to parameters related to the LocalPlanner.
        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        self._last_traffic_light = None

        # Params for MODIA
        self._actions = {1: "stop", 2: "edge", 3: "go"}
        self._positions = {"before": 1.0, "at": 2.0, "inside": 3.0, "after": 4.0}
        self._observing = ("ego_pos", "rival_pos", "rival_blocking", "rival_vel")
        self._last_action = 1
        self._init_belief = init_belief
        self._State_Space = MODIA.State_Space
        self._Obs_Space = MODIA.Obs_Space
        self._get_action_from_belief = MODIA.get_action_from_belief
        self._belief_updater = MODIA.DiscreteUpdater(MODIA.StopUncontrolledDP.Pomdp)
        self._observation_history = []
        self._action_history = []
        self._consideration_diameter = 50.0   # meters

        # Params for Stop-Uncontrolled DP
        #! Future work: Need to consider the case where rivals are coming from the `after` position for the ego (opposite lane).
        self._ignore_stop_signs = False
        self._Stop_Uncontrolled_DCs = dict()
        self._stop_uncontrolled_pomdp = StopUncontrolledDP.Pomdp
        self._stop_uncontrolled_policy = StopUncontrolledDP.policy
        self._stop_sign_prop = "static.prop.trafficcone01"
        self._base_stop_sign_threshold = 1.0  # meters
        self._stop_sign_stop_amount = 1.5   # seconds

        self._last_stop_sign = None
        self._last_stop_sign_road_id = None
        self._last_stop_sign_detect_time = None

        self._at_pos_threshold = 20.0   # meters        # rival: smaller than this is "at"
        self._inside_pos_threshold = 15.0   # meters    # rival: smaller than this is "inside"
        self._after_pos_threshold = 70.0   # degrees    # ego and rival: smaller than this is "after"

        self._rival_blocking = {True: 1.0, False: 2.0}
        self._aggsv_vals = {"cautious": 1.0, "normal": 2.0, "aggressive": 3.0}
        self._cautious_threshold = 10.0   # m/s
        self._aggressive_threshold = 40.0   # m/s

        # Base parameters
        self._ignore_traffic_lights = True
        self._ignore_vehicles = False
        self._target_speed = target_speed
        self._sampling_resolution = 2.0
        self._base_tlight_threshold = 5.0  # meters
        self._base_vehicle_threshold = 5.0  # meters
        self._max_brake = 0.5

        # Change parameters according to the dictionary
        opt_dict['target_speed'] = target_speed
        if 'ignore_traffic_lights' in opt_dict:
            self._ignore_traffic_lights = opt_dict['ignore_traffic_lights']
        if 'ignore_stop_signs' in opt_dict:
            self._ignore_stop_signs = opt_dict['ignore_stop_signs']
        if 'ignore_vehicles' in opt_dict:
            self._ignore_vehicles = opt_dict['ignore_vehicles']
        if 'sampling_resolution' in opt_dict:
            self._sampling_resolution = opt_dict['sampling_resolution']
        if 'base_tlight_threshold' in opt_dict:
            self._base_tlight_threshold = opt_dict['base_tlight_threshold']
        if 'base_vehicle_threshold' in opt_dict:
            self._base_vehicle_threshold = opt_dict['base_vehicle_threshold']
        if 'max_brake' in opt_dict:
            self._max_steering = opt_dict['max_brake']

        # Initialize the planners
        self._local_planner = LocalPlanner(self._vehicle, opt_dict=opt_dict)
        self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)

    def add_emergency_stop(self, control):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param control (carla.VehicleControl): control to be modified
        """
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

    def halt_stop(self):
        """
        Halt vehicle to a sudden stop. Use as: my_vehicle.apply_control(agent.halt_stop())
        For debugging purposes only. Use `add_emergency_stop` as a control law instead.
        """
        return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=False, reverse=False, manual_gear_shift=False, gear=1)

    def set_target_speed(self, speed):
        """
        Changes the target speed of the agent
            :param speed (float): target speed in Km/h
        """
        self._local_planner.set_speed(speed)

    def follow_speed_limits(self, value=True):
        """
        If active, the agent will dynamically change the target speed according to the speed limits

            :param value (bool): whether or not to activate this behavior
        """
        self._local_planner.follow_speed_limits(value)

    def get_local_planner(self):
        """Get method for protected member local planner"""
        return self._local_planner

    def get_global_planner(self):
        """Get method for protected member local planner"""
        return self._global_planner

    def set_destination(self, end_location, start_location=None):
        """
        This method creates a list of waypoints between a starting and ending location,
        based on the route returned by the global router, and adds it to the local planner.
        If no starting location is passed, the vehicle local planner's target location is chosen,
        which corresponds (by default), to a location about 5 meters in front of the vehicle.

            :param end_location (carla.Location): final location of the route
            :param start_location (carla.Location): starting location of the route
        """

        self._reset_histories()

        if not start_location:
            start_location = self._local_planner.target_waypoint.transform.location
            clean_queue = True
        else:
            start_location = self._vehicle.get_location()
            clean_queue = False

        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self.trace_route(start_waypoint, end_waypoint)
        self._local_planner.set_global_plan(route_trace, clean_queue=clean_queue)

    def set_global_plan(self, plan, stop_waypoint_creation=True, clean_queue=True):
        """
        Adds a specific plan to the agent.

            :param plan: list of [carla.Waypoint, RoadOption] representing the route to be followed
            :param stop_waypoint_creation: stops the automatic random creation of waypoints
            :param clean_queue: resets the current agent's plan
        """
        self._local_planner.set_global_plan(
            plan,
            stop_waypoint_creation=stop_waypoint_creation,
            clean_queue=clean_queue
        )

    def trace_route(self, start_waypoint, end_waypoint):
        """
        Calculates the shortest route between a starting and ending waypoint.

            :param start_waypoint (carla.Waypoint): initial waypoint
            :param end_waypoint (carla.Waypoint): final waypoint
        """
        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location
        return self._global_planner.trace_route(start_location, end_location)

    def run_step(self):
        """
        Execute one step of navigation.
        """

        list_of_actions = []

        # Retrieve all relevant actors
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")
        stop_signs_list = actor_list.filter("*{}*".format(self._stop_sign_prop))

        vehicle_speed = get_speed(self._vehicle) / 3.6

        # Check for possible vehicle obstacles
        max_vehicle_distance = self._base_vehicle_threshold + vehicle_speed
        affected_by_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max_vehicle_distance)
        if affected_by_vehicle:
            list_of_actions.append(1)  # send "stop"

        # Check if the vehicle is affected by a red traffic light
        max_tlight_distance = self._base_tlight_threshold + vehicle_speed
        affected_by_tlight, _ = self._affected_by_traffic_light(lights_list, max_tlight_distance)
        if affected_by_tlight:
            list_of_actions.append(1)  # send "stop"

        # Check if the vehicle is affected by a stop sign
        max_stop_sign_distance = self._base_stop_sign_threshold + vehicle_speed
        affected_by_stop_sign, _ = self._affected_by_stop_sign(stop_signs_list, max_stop_sign_distance)
        if affected_by_stop_sign:
            print("AT STOP SIGN!!")
            list_of_actions.append(1)  # send "stop"

        # Update POMDP belief in each DC
        self._refresh_DCs(vehicle_list)
        obs = self._get_observations()
        obs_jl_for_DCs = self._get_obs_corresponding_idx(obs)   # list of obs_jl for each DC
        print(obs_jl_for_DCs)
        DC_actions = self._update_DC_beliefs(self._last_action, obs_jl_for_DCs)
        list_of_actions.extend(DC_actions)

        # import ipdb; ipdb.set_trace()

        # Pass control input
        control = self._local_planner.run_step()
        a_idx, act = self._get_safest_action(list_of_actions)

        # Record histories
        self._record_to_history(self._last_action, obs_jl_for_DCs)
        self._last_action = a_idx

        if act == "stop":
            return self.add_emergency_stop(control)

        elif act == "go":
            return control

        else:   # edge
            # TODO.
            return control

    def done(self):
        """Check whether the agent has reached its destination."""
        return self._local_planner.done()

    def get_observation_history(self):
        "Get observation history of the most recent episode."
        return self._observation_history

    def get_action_history(self):
        "Get action history of the most recent episode."
        return self._action_history

    def get_belief(self):
        """Get latest belief of MODIA."""
        return self._Stop_Uncontrolled_DCs

    def ignore_traffic_lights(self, active=True):
        """(De)activates the checks for traffic lights."""
        self._ignore_traffic_lights = active

    def ignore_stop_signs(self, active=True):
        """(De)activates the checks for stop signs."""
        self._ignore_stop_signs = active

    def ignore_vehicles(self, active=True):
        """(De)activates the checks for other vehicles."""
        self._ignore_vehicles = active

    def _construct_obs(self, names, vals):
        return MODIA.py_construct_obs(names, vals)

    def _get_position_wrt_stop_sign(self, vehicle, is_ego=False):
        stop_sign = self._last_stop_sign
        dist = distance_among_actors(vehicle, stop_sign)
        angle = ref_angle(vehicle, stop_sign)

        if is_ego:
            if not self._last_stop_sign:
                return "before"
            elif self._last_stop_sign and angle <= self._after_pos_threshold:
                self._last_stop_sign = None
                return "after"
            elif self._last_stop_sign and dist < (self._base_stop_sign_threshold + 1.5*vehicle.bounding_box.extent.x):
                return "at"
            else:
                return "inside"

        else:
            if dist < self._inside_pos_threshold and angle > self._after_pos_threshold:
                return "inside"
            elif dist < self._at_pos_threshold and angle > self._after_pos_threshold:
                return "at"
            elif angle <= self._after_pos_threshold:
                return "after"
            else:
                return "before"

    def _get_observations(self):
        """Get observations from the environment for a single step."""
        observations = dict()

        for rv_id in self._Stop_Uncontrolled_DCs:
            rival = self._world.get_actor(rv_id)
            obs = []

            # Get ego position
            ego_pos = self._get_position_wrt_stop_sign(self._vehicle, is_ego=True)
            obs.append(self._positions[ego_pos])

            # Get rival position
            rival_pos = self._get_position_wrt_stop_sign(rival, is_ego=False)
            obs.append(self._positions[rival_pos])

            # Get rival blocking
            rival_blk, _ = self._is_vehicle_blocking(rival)
            obs.append(self._rival_blocking[rival_blk])

            # Get rival aggressiveness
            rival_vel = vector3D_norm(rival.get_velocity())
            if rival_vel <= self._cautious_threshold:
                obs.append(self._aggsv_vals["cautious"])
                
            elif rival_vel >= self._aggressive_threshold:
                obs.append(self._aggsv_vals["aggressive"])

            else:
                obs.append(self._aggsv_vals["normal"])

            observations[rv_id] = (*obs, )    # unpack list into tuple

        return observations


    def _get_obs_corresponding_idx(self, obs):
        names = self._observing
        result = dict()
        for rival_id, vals in obs.items():
            obs_jl = self._construct_obs(names, vals)
            item = jlBase.getindex(self._Obs_Space, obs_jl)
            result[rival_id] = item
        return result

    def _record_to_history(self, a_idx, o_idx):
        self._action_history.append(a_idx)
        self._observation_history.append(o_idx)

    def _reset_histories(self):
        self._observation_history = []
        self._action_history = []

    def _refresh_DCs(self, vehicle_list):
        ego_location = self._vehicle.get_location()
        rivals_to_consider = [item for item in vehicle_list if carla.Location.distance(ego_location, item.get_location()) <= self._consideration_diameter]
        
        for rv in rivals_to_consider:
            if rv.id not in self._Stop_Uncontrolled_DCs:
                self._Stop_Uncontrolled_DCs[rv.id] = self._init_belief

    def _update_DC_beliefs(self, a_idx, obs_jl_for_DCs):
        """Update DC beliefs and also return the recommended action from each DC."""
        DC_actions = []
        for rival_id, belief in self._Stop_Uncontrolled_DCs.items():
            self._Stop_Uncontrolled_DCs[rival_id] = MODIA.update_belief(self._belief_updater, belief, a_idx, obs_jl_for_DCs[rival_id])
            DC_actions.append(self._get_action_from_belief(self._stop_uncontrolled_policy, belief))
        return DC_actions

    def _get_safest_action(self, list_of_actions):
        a = min(list_of_actions)
        return a, self._actions[a]    # e.g. (1, :stop)

    def _affected_by_traffic_light(self, lights_list=None, max_distance=None):
        """
        Method to check if there is a red light affecting the vehicle.

            :param lights_list (list of carla.TrafficLight): list containing TrafficLight objects.
                If None, all traffic lights in the scene are used
            :param max_distance (float): max distance for traffic lights to be considered relevant.
                If None, the base threshold value is used
        """
        if self._ignore_traffic_lights:
            return (False, None)

        if not lights_list:
            lights_list = self._world.get_actors().filter("*traffic_light*")

        if not max_distance:
            max_distance = self._base_tlight_threshold

        if self._last_traffic_light:
            if self._last_traffic_light.state != carla.TrafficLightState.Red:
                self._last_traffic_light = None
            else:
                return (True, self._last_traffic_light)

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            object_location = get_trafficlight_trigger_location(traffic_light)
            object_waypoint = self._map.get_waypoint(object_location)

            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            if is_within_distance(object_waypoint.transform, self._vehicle.get_transform(), max_distance, [0, 90]):
                self._last_traffic_light = traffic_light
                return (True, traffic_light)

        return (False, None)


    def _affected_by_stop_sign(self, stop_signs_list=None, max_distance=None):
        """
        Method to check if there is a stop affecting the vehicle.

            :param stop_signs_list (list of actors): list containing objects regarded as stop signs.
                If None, all stop signs in the scene are used
            :param max_distance (float): max distance for stop signs to be considered relevant.
                If None, the base threshold value is used
        """        
        if self._ignore_stop_signs:
            return (False, None)

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)
        
        if not max_distance:
            max_distance = self._base_stop_sign_threshold

        # If the acting stop sign has already been detected, ignore detection
        if self._last_stop_sign_road_id == ego_vehicle_waypoint.road_id:
            if time.time() - self._last_stop_sign_detect_time > self._stop_sign_stop_amount:  # ego car has already waited enough at this sign
                return (False, None)
            else:
                return (True, self._last_stop_sign_road_id)

        for stop_sign in stop_signs_list:
            object_location = stop_sign.get_location()
            object_waypoint = self._map.get_waypoint(object_location)

            # Check whether the stop sign and the ego vehicle are on the same road
            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            # Check whether the stop sign is pointed in the direction that the ego vehicle cares
            if dot_ve_wp < 0:
                continue

            if is_within_distance(object_waypoint.transform, self._vehicle.get_transform(), max_distance, [0, 90]):
                self._last_stop_sign_road_id = object_waypoint.road_id
                self._last_stop_sign = stop_sign
                self._last_stop_sign_detect_time = time.time()
                return (True, stop_sign)

        return (False, None)


    def _vehicle_obstacle_detected(self, vehicle_list=None, max_distance=None):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """
        if self._ignore_vehicles:
            return (False, None)

        rival_vehicles = []

        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())

        # Get the transform of the front of the ego
        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            x=ego_extent * ego_forward_vector.x,
            y=ego_extent * ego_forward_vector.y,
        )

        for target_vehicle in vehicle_list:
            target_transform = target_vehicle.get_transform()
            target_wpt = self._map.get_waypoint(target_transform.location)
            if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id:
                next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                if not next_wpt:
                    continue
                if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id:
                    continue

            target_forward_vector = target_transform.get_forward_vector()
            target_extent = target_vehicle.bounding_box.extent.x
            target_rear_transform = target_transform
            target_rear_transform.location -= carla.Location(
                x=target_extent * target_forward_vector.x,
                y=target_extent * target_forward_vector.y,
            )

            if is_within_distance(target_rear_transform, ego_front_transform, max_distance, [0, 90]):
                return (True, target_vehicle)
        return (False, None)

    def _is_vehicle_blocking(self, target_vehicle):
        ego_transform = self._vehicle.get_transform()
        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())

        # Get the transform of the front of the ego
        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            x=ego_extent * ego_forward_vector.x,
            y=ego_extent * ego_forward_vector.y,
        )

        target_transform = target_vehicle.get_transform()
        target_wpt = self._map.get_waypoint(target_transform.location)
        if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id:
            next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
            if not next_wpt:
                return (False, None)
            if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id:
                return (False, None)

        target_forward_vector = target_transform.get_forward_vector()
        target_extent = target_vehicle.bounding_box.extent.x
        target_rear_transform = target_transform
        target_rear_transform.location -= carla.Location(
            x=target_extent * target_forward_vector.x,
            y=target_extent * target_forward_vector.y,
        )

        if is_within_distance(target_rear_transform, ego_front_transform, self._base_vehicle_threshold, [0, 90]):
            return (True, target_vehicle)
        return (False, None)
