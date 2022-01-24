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
class TrainArguments:
    num_of_trials = 101
    timeout_duration = 30.0
    spawn_radius = 100.0
    pkl_frequency = 1
    orient_spectator = False

    verbose_belief = False
    env_observability_settings = {"low": False, "high": True}
    env_density_settings = {"low": 5, "med": 15, "high": 40}
    env_aggressiveness_settings = {"cautious": PresetScenarios.CAUTIOUS, "normal": PresetScenarios.NORMAL, "aggressive": PresetScenarios.AGGRESSIVE} 

# Create Carla client and world
train_args = TrainArguments()
client = carla.Client('localhost', 2000)
client.set_timeout(20.0)
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

# Generate waypoints for ego path
road_seed = 1
blueprint_library = world.get_blueprint_library()
env_objs = world.get_environment_objects(carla.CityObjectLabel.Roads)
map = world.get_map()
topology = map.get_topology()
[waypoint_start, waypoint_end] = generate_scenario_midpoint(topology, map, road_seed)

# Create ego vehicle
vehicle_init_tf = carla.Transform(carla.Location(x=waypoint_start.location.x, y=waypoint_start.location.y, z=0.1) , carla.Rotation(pitch=0.000000, yaw=waypoint_start.rotation.yaw, roll=0.000000)) 
my_vehicle_tf = vehicle_init_tf
my_vehicle_bp = blueprint_library.find('vehicle.ford.mustang')
my_vehicle = world.spawn_actor(my_vehicle_bp, my_vehicle_tf)
print(f"My vehicle ID: {my_vehicle.id}")


for (key1, ENV_OBSV) in tqdm(train_args.env_observability_settings.items(), desc="Env Observability"):
    for (key2, ENV_DENS) in tqdm(train_args.env_density_settings.items(), desc="Env Density"):
        for (key3, ENV_AGGR) in tqdm(train_args.env_aggressiveness_settings.items(), desc="Env Aggressiveness"):

            PKL_SAVENAME = f"Obsv_{key1}_Dens_{key2}_Aggr_{key3}"

            ALL_ACTION_HISTORIES = []
            ALL_OBSERVATION_HISTORIES = []


            for trial in tqdm(range(train_args.num_of_trials), desc="Trial running"):

                # TODO: Add skip if trial pkl already exists in dir, and `pkl_frequency` == 1

                # Orient the spectator w.r.t. `my_vehicle.id`
                if train_args.orient_spectator: orient = subprocess.Popen(['./nodes/orient_spectator.py', '-a', str(my_vehicle.id)])

                # Reset back to init tf
                my_vehicle.set_transform(vehicle_init_tf)
                time.sleep(1)

                # Start an agent
                init_belief = uniform_belief(StopUncontrolledDP.Pomdp)
                agent = MODIAAgent(my_vehicle, init_belief, StopUncontrolledDP, verbose_belief=train_args.verbose_belief, env_observable=ENV_OBSV)
                agent.set_destination(waypoint_end.location)

                # Generate traffic
                traffic_gen_seed = trial
                vehicles_list, walkers_list, all_id, all_actors, traffic_manager, _ = generate_traffic_func(ENV_AGGR, ENV_DENS, train_args.spawn_radius, my_vehicle.id, traffic_gen_seed)

                time_start = time.time()
                while time.time() - time_start < train_args.timeout_duration:
                    world.tick()
                    if agent.done():
                        print("Target destination has been reached. Stopping vehicle.")
                        my_vehicle.apply_control(agent.halt_stop())
                        kill_traffic(vehicles_list, walkers_list, all_id, all_actors, traffic_manager)
                        if train_args.orient_spectator: orient.kill()
                        break

                    my_vehicle.apply_control(agent.run_step())

                print(f"Trial: {trial}, Time taken: {time.time() - time_start}")
                my_vehicle.apply_control(agent.halt_stop())
                kill_traffic(vehicles_list, walkers_list, all_id, all_actors, traffic_manager)
                if train_args.orient_spectator: orient.kill()

                ALL_ACTION_HISTORIES.append(agent.get_action_history())
                ALL_OBSERVATION_HISTORIES.append(agent.get_observation_history())

                if train_args.pkl_frequency == 1:
                    data = [[agent.get_action_history()], [agent.get_observation_history()]]   # saves trials individually
                    save_with_pkl(data=data, savename=f"{PKL_SAVENAME}_trial{trial}", is_stamped=True)

                elif PKL_SAVENAME and (trial % train_args.pkl_frequency == 0):
                    save_with_pkl(data=[ALL_ACTION_HISTORIES, ALL_OBSERVATION_HISTORIES], savename=f"{PKL_SAVENAME}_trial{trial}", is_stamped=True)
