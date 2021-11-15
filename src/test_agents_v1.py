import carla
import sys
import numpy as np
import time

# Add Carla dim PythonAPI scripts to path
sys.path.append('./PythonAPI/carla/')

# Create Carla client and world
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world("Town01_Opt")
world.set_weather(carla.WeatherParameters.WetCloudySunset)

# To import a basic agent
from PythonAPI.carla.agents.navigation.basic_agent import BasicAgent

# To import a behavior agent
from PythonAPI.carla.agents.navigation.behavior_agent import BehaviorAgent

# Spawn a vehicle
my_actors_list = []

blueprint_library = world.get_blueprint_library()
my_vehicle_tf = carla.Transform(carla.Location(x=-1.4727, y=-13.5143, z=1.0), carla.Rotation(pitch=0.000000, yaw=270.000000, roll=0.000000))
my_vehicle_bp = blueprint_library.find('vehicle.ford.mustang')

my_vehicle = world.spawn_actor(my_vehicle_bp, my_vehicle_tf)
my_actors_list.append(my_vehicle)

# Get vehicle id
my_vehicle_id = my_vehicle.id
# actor_list = world.get_actors()
# for vehicle in actor_list.filter('vehicle.ford.*'):
#     print(vehicle.id)

# Start an agent
agent = BehaviorAgent(my_vehicle)
# # Set destination (Optinal)
# spawn_points = world.get_map().get_spawn_points()
destination = carla.Transform(carla.Location(x=-37.2375, y=-69.8703, z=0.0), carla.Rotation(pitch=0.000000, yaw=0.000000, roll=0.000000))
agent.set_destination(destination.location)

# spectator = world.get_spectator()
# spec_loc = carla.Location(x=-16.508865356445312,y=-32.86841583251953,z=42.87731170654297)
# spectator.set_location(spec_loc)

# Change traffic lights to green always, then destroy.
actor_list = world.get_actors()
traflights = actor_list.filter("traffic.traffic_light*")
for item in traflights:
    item.set_state(carla.TrafficLightState.Green)
    item.set_green_time(9999.0)


time.sleep(5)
stopped_flag = False
stop_sign_loc = carla.Location(x=2.983862, y=-52.730301, z=0.028759)
while True:
    if agent.done():
        print("Target destination has been reached. Stopping vehicle.")
        break

    my_vehicle.apply_control(agent.run_step())
    dist_to_sign = carla.Location.distance(my_vehicle.get_location(), stop_sign_loc)
    
    if not stopped_flag: print(f"Distance to nearest stop sign: {dist_to_sign}")

    if dist_to_sign<10 and not stopped_flag:
        stopped_flag = True
        cs = agent.add_emergency_stop(agent.run_step())
        my_vehicle.apply_control(cs)
        time.sleep(5)
