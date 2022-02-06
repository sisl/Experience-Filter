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

# Args for testing
class TestArguments:
    num_of_trials = 10
    timeout_duration = 30.0
    spawn_radius = 100.0
    orient_spectator = False
    verbose_belief = False

    datapoint_to_benchmark = (0, 15, 2)
    methods = ["self", "inv_distance", "nearest"]
    env_aggressiveness_levels = {0: PresetScenarios.CAUTIOUS, 1: PresetScenarios.NORMAL, 2: PresetScenarios.AGGRESSIVE} 

    training_effort = list(range(3, 19, 3))   # 3, 6, ..., 18


# Args for Experince Filter
class FilterArguments:
    axeslabels = ("Observability", "Density", "Aggressiveness")
    env_observability_settings = {"low": 0 , "high": 1}
    env_density_settings = {"low": 5, "med": 15, "high": 40}
    env_aggressiveness_settings = {"cautious": 1, "normal": 2, "aggressive": 3}

    working_dir = os.getcwd()
    rel_path_to_pkls = "./dev_train_v2/"
    prior_scenario_count = 2000
    plot_policy_maps = False    # if True, you need to enable `plot_funcs.jl` in MODIA.jl.

# # Args for scoring
# class ScoreArguments:
#     safety  = +2.0    # higher is better
#     comfort = -3.0    # lower is better
#     time    = -2.0    # lower is better

# Create Carla client and world
test_args = TestArguments()
filter_args = FilterArguments()
# score_args = ScoreArguments()
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

# Import Experience Filter
from ExperienceFilter import *
from expfilter_helper_funcs import *



# TODO
# for tef in test_args.training_effort:
#     covpts = get_coverage_points(filter_data.keys(), tef)
#     filter_data = get_filter_data_subset(filter_args, covpts)
    
#     # Trained with all data

#     # Trained (self)
#     EF_self = ExperienceFilter(data=filter_data, axeslabels=filter_args.axeslabels)
#     T = EF_self.apply_filter(new_point=test_args.datapoint_to_benchmark)
#     StopUncontrolledDP_new = create_DP_from_Trans_func(T)

#     # Experince Filter


#     # Nearest neighbor