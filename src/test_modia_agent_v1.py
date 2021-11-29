import carla
import sys, importlib, os
import numpy as np
import time

from helper_funcs import *

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

# Spawn Rival vehicles
rival1_tf = carla.Transform(carla.Location(x=338.254791, y=325.919067, z=1.0), carla.Rotation(pitch=0.000000, yaw=180.000000, roll=0.000000))
rival1_bp = blueprint_library.find('vehicle.tesla.model3')
rival1 = world.spawn_actor(rival1_bp, rival1_tf)
my_actors_list.append(rival1)

rival2_tf = carla.Transform(carla.Location(x=308.645416, y=325.919067, z=1.0), carla.Rotation(pitch=0.000000, yaw=180.000000, roll=0.000000))
rival2_bp = blueprint_library.find('vehicle.mercedes.coupe')
rival2 = world.spawn_actor(rival2_bp, rival2_tf)
my_actors_list.append(rival2)

# Spawn Ego vehicle
vehicle_init_tf = carla.Transform(carla.Location(x=334.186920, y=299.277069, z=1.0), carla.Rotation(pitch=0.000000, yaw=90.000000, roll=0.000000)) 
# vehicle_init_tf = carla.Transform(carla.Location(x=335.473236, y=316.107178, z=1.0), carla.Rotation(pitch=0.000000, yaw=90.000000, roll=0.000000))   # right in front of stop sign
my_vehicle_tf = vehicle_init_tf
my_vehicle_bp = blueprint_library.find('vehicle.ford.mustang')
my_vehicle = world.spawn_actor(my_vehicle_bp, my_vehicle_tf)
my_actors_list.append(my_vehicle)
print(f"My vehicle ID: {my_vehicle.id}")

## You might want to orient the spectator here, w.r.t. `my_vehicle.id`.

# Reset back to init tf
my_vehicle.set_transform(vehicle_init_tf)
time.sleep(1)

# Start an agent
init_belief = uniform_belief(StopUncontrolledDP.Pomdp)
agent = MODIAAgent(my_vehicle, init_belief, StopUncontrolledDP, verbose_belief=True)
destination = carla.Transform(carla.Location(x=317.176300, y=327.626740, z=0.0), carla.Rotation(pitch=0.000000, yaw=180.000000, roll=0.000000))
agent.set_destination(destination.location)

while True:
    if agent.done():
        print("Target destination has been reached. Stopping vehicle.")
        my_vehicle.apply_control(agent.halt_stop())
        break

    my_vehicle.apply_control(agent.run_step())
