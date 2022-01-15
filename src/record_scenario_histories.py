import carla
import sys, importlib, os
import numpy as np
import time
import subprocess
import time 
import pickle
from tqdm import tqdm

from helper_funcs import *
from traffic_funcs import ScenarioParams, PresetScenarios, generate_scenario_midpoint, generate_traffic_func, kill_traffic

# Add Carla PythonAPI scripts to path
sys.path.append('./PythonAPI/carla/')

# Set args for training
class TrialArguments:
    num_of_trials = 4
    timeout_duration = 30.0
    scenario_type = PresetScenarios.AGGRESSIVE
    number_of_vehicles = 50
    spawn_radius = 100.0
    pkl_savename = "Aggressive_50cars"
    pkl_frequency = 10
    orient_spectator = True
    verbose_belief = False

# Create Carla client and world
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world("Town01_Opt")
world.set_weather(carla.WeatherParameters.WetCloudySunset)
args = TrialArguments()

connect_julia_api = True
if connect_julia_api:
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import Pkg
    from julia import Base as jlBase
    Pkg.activate("../")    # while current dir is /src
    from julia import MODIA
    StopUncontrolledDP = MODIA.StopUncontrolledDP
    uniform_belief = MODIA.uniform_belief

# Import MODIA agent
from modia_agent import MODIAAgent
importlib.reload(sys.modules['modia_agent'])
from modia_agent import MODIAAgent

# Replace all traffic lights with construction cones
exec(open("./nodes/replace_actors_type.py").read())

# Generate waypoints for ego path
road_seed = 1
blueprint_library = world.get_blueprint_library()
env_objs = world.get_environment_objects(carla.CityObjectLabel.Roads)
map = world.get_map()
topology = map.get_topology()
[waypoint_start, waypoint_end] = generate_scenario_midpoint(topology, map, road_seed)

# Create ego vehicle
vehicle_init_tf = carla.Transform(carla.Location(x=waypoint_start.location.x, y=waypoint_start.location.y, z=1.0) , carla.Rotation(pitch=0.000000, yaw=waypoint_start.rotation.yaw, roll=0.000000)) 
my_vehicle_tf = vehicle_init_tf
my_vehicle_bp = blueprint_library.find('vehicle.ford.mustang')
my_vehicle = world.spawn_actor(my_vehicle_bp, my_vehicle_tf)
print(f"My vehicle ID: {my_vehicle.id}")

# Orient the spectator w.r.t. `my_vehicle.id`
if args.orient_spectator: orient = subprocess.Popen(['./nodes/orient_spectator.py', '-a', str(my_vehicle.id)])


ALL_ACTION_HISTORIES = []
ALL_OBSERVATION_HISTORIES = []

for trial in tqdm(range(args.num_of_trials), desc="Trial running"):

    # Reset back to init tf
    
    my_vehicle.set_transform(vehicle_init_tf)
    time.sleep(1)

    # Start an agent
    init_belief = uniform_belief(StopUncontrolledDP.Pomdp)
    agent = MODIAAgent(my_vehicle, init_belief, StopUncontrolledDP, verbose_belief=args.verbose_belief)
    agent.set_destination(waypoint_end.location)

    # Generate traffic
    traffic_gen_seed = 3
    vehicles_list, walkers_list, all_id, all_actors, traffic_manager = generate_traffic_func(args.scenario_type, args.number_of_vehicles, args.spawn_radius, my_vehicle.id, traffic_gen_seed)

    time_start = time.time()
    while time.time() - time_start < args.timeout_duration:
        if agent.done():
            print("Target destination has been reached. Stopping vehicle.")
            my_vehicle.apply_control(agent.halt_stop())
            kill_traffic(vehicles_list, walkers_list, all_id, all_actors, traffic_manager)
            # orient.kill()
            break

        my_vehicle.apply_control(agent.run_step())

    print(f"Trial: {trial}, Time taken: {time.time() - time_start}")
    my_vehicle.apply_control(agent.halt_stop())
    kill_traffic(vehicles_list, walkers_list, all_id, all_actors, traffic_manager)
    # orient.kill()

    ALL_ACTION_HISTORIES.append(agent.get_action_history())
    ALL_OBSERVATION_HISTORIES.append(agent.get_observation_history())

    if trial % args.pkl_frequency == 0:
        save_with_pkl(data=[ALL_ACTION_HISTORIES, ALL_OBSERVATION_HISTORIES], savename=f"{args.pkl_savename}_trial{trial}", is_stamped=True)

