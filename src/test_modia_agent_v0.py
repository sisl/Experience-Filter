import carla
import sys, importlib, os
import numpy as np
import time

# Add Carla dim PythonAPI scripts to path
sys.path.append('./PythonAPI/carla/')
connect_julia_api = False

# Create Carla client and world
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world("Town01_Opt")
world.set_weather(carla.WeatherParameters.WetCloudySunset)

if connect_julia_api:
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import Pkg
    Pkg.activate("../")    # while current dir is /src
    from julia import MODIA


# To import a MODIA agent
from modia_agent import MODIAAgent
importlib.reload(sys.modules['modia_agent'])
from modia_agent import MODIAAgent

# Replace all traffic lights with construction cones
exec(open("./nodes/replace_actors_type.py").read())

# Spawn a vehicle
my_actors_list = []

# vehicle_init_tf = carla.Transform(carla.Location(x=334.186920, y=299.277069, z=1.0), carla.Rotation(pitch=0.000000, yaw=90.000000, roll=0.000000)) 
vehicle_init_tf = carla.Transform(carla.Location(x=335.473236, y=316.107178, z=1.0), carla.Rotation(pitch=0.000000, yaw=90.000000, roll=0.000000))
blueprint_library = world.get_blueprint_library()
my_vehicle_tf = vehicle_init_tf
my_vehicle_bp = blueprint_library.find('vehicle.ford.mustang')

my_vehicle = world.spawn_actor(my_vehicle_bp, my_vehicle_tf)
my_actors_list.append(my_vehicle)
my_vehicle_id = my_vehicle.id

# Reset back to init tf
my_vehicle.set_transform(vehicle_init_tf)
time.sleep(1)

# Start an agent
agent = MODIAAgent(my_vehicle)
destination = carla.Transform(carla.Location(x=324.935669, y=326.853607, z=0.0), carla.Rotation(pitch=0.000000, yaw=180.000000, roll=0.000000))
agent.set_destination(destination.location)


while True:
    if agent.done():
        print("Target destination has been reached. Stopping vehicle.")
        my_vehicle.apply_control(agent.halt_stop())
        break

    my_vehicle.apply_control(agent.run_step())
    