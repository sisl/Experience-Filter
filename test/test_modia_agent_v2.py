import carla
import sys, importlib, os
import numpy as np
import time

from helper_funcs import *
from traffic_funcs import *

# Add Carla dim PythonAPI scripts to path
sys.path.append('./PythonAPI/carla/')

# Create Carla client and world
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world("Town01_Opt")
world.set_weather(carla.WeatherParameters.WetCloudySunset)


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

# Keep track of actors
blueprint_library = world.get_blueprint_library()
my_actors_list = []

env_objs = world.get_environment_objects(carla.CityObjectLabel.Roads)
map = world.get_map()
topology = map.get_topology()

#[waypoint_start, waypoint_end] = generate_scenario(topology, map)
# [waypoint_start, waypoint_end] = generate_scenario_tree(topology, map)
[waypoint_start, waypoint_end] = generate_scenario_midpoint(topology, map)
#generate_scenario_tree(topology, map)

#vehicle_init_tf = carla.Transform(carla.Location(x=waypoint_start.transform.location.x, y=waypoint_start.transform.location.y, z=1.0) , carla.Rotation(pitch=0.000000, yaw=waypoint_start.transform.rotation.yaw, roll=0.000000)) 
vehicle_init_tf = carla.Transform(carla.Location(x=waypoint_start.location.x, y=waypoint_start.location.y, z=1.0) , carla.Rotation(pitch=0.000000, yaw=waypoint_start.rotation.yaw, roll=0.000000)) 

# Spawn Ego vehicle
# vehicle_init_tf = carla.Transform(carla.Location(x=334.186920, y=299.277069, z=1.0), carla.Rotation(pitch=0.000000, yaw=90.000000, roll=0.000000)) 

my_vehicle_tf = vehicle_init_tf
my_vehicle_bp = blueprint_library.find('vehicle.ford.mustang')
my_vehicle = world.spawn_actor(my_vehicle_bp, my_vehicle_tf)
my_actors_list.append(my_vehicle)
print(f"My vehicle ID: {my_vehicle.id}")
# You might want to orient the spectator here, w.r.t. `my_vehicle.id`.


# Reset back to init tf
my_vehicle.set_transform(vehicle_init_tf)
time.sleep(1)

# Start an agent
init_belief = uniform_belief(StopUncontrolledDP.Pomdp)
agent = MODIAAgent(my_vehicle, init_belief, StopUncontrolledDP, verbose_belief=True)

# destination = carla.Transform(carla.Location(x=317.176300, y=327.626740, z=0.0), carla.Rotation(pitch=0.000000, yaw=180.000000, roll=0.000000))

#destination = waypoint_end.transform
agent.set_destination(waypoint_end.location)

scenario = 1
spawn_radius = 100 
seed = 0
vehicles_list, walkers_list, all_id, all_actors = generate_traffic_func(scenario, spawn_radius, my_vehicle.id)

while True:
    if agent.done():
        print("Target destination has been reached. Stopping vehicle.")
        my_vehicle.apply_control(agent.halt_stop())
        kill_traffic(vehicles_list, walkers_list, all_id, all_actors)
        break

    my_vehicle.apply_control(agent.run_step())
    
