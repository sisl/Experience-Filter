from cgi import test
import carla
import sys, importlib, os
import numpy as np
import time
import subprocess
import time 
import pickle
from tqdm import tqdm

from helper_funcs import *
from expfilter_helper_funcs import *
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

# Args for scoring
class ScoreArguments:
    safety  = +2.0    # higher is better
    comfort = -3.0    # lower is better
    time    = -2.0    # lower is better

# Create Carla client and world
test_args = TestArguments()
score_args = ScoreArguments()
filter_args = FilterArguments()
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

# Get Experience Filter data
from ExperienceFilter import *
filter_data = get_filter_data(filter_args)

SCORE_RECORDS = dict()

for method in test_args.methods:
    if method == "self":
        EF = ExperienceFilter(data=filter_data, axeslabels=filter_args.axeslabels)
        T = EF.apply_filter(new_point=test_args.datapoint_to_benchmark)
        StopUncontrolledDP_new = create_DP_from_Trans_func(T)
    else:
        EF.remove_datapoint(test_args.datapoint_to_benchmark)
        T = EF.apply_filter(new_point=test_args.datapoint_to_benchmark, weighting=method)
        StopUncontrolledDP_new = create_DP_from_Trans_func(T)

    for trial in tqdm(range(test_args.num_of_trials), desc="Trial running"):

        ENV_OBSV, ENV_DENS, ENV_AGGR = test_args.datapoint_to_benchmark
        ENV_AGGR = test_args.env_aggressiveness_levels[ENV_AGGR]

        # Orient the spectator w.r.t. `my_vehicle.id`
        if test_args.orient_spectator: orient = subprocess.Popen(['./nodes/orient_spectator.py', '-a', str(my_vehicle.id)])

        # Reset back to init tf
        my_vehicle.set_transform(vehicle_init_tf)
        time.sleep(1)

        # Start an agent
        init_belief = uniform_belief(StopUncontrolledDP_new.Pomdp)
        agent = MODIAAgent(my_vehicle, init_belief, StopUncontrolledDP_new, verbose_belief=test_args.verbose_belief, env_observable=ENV_OBSV)
        agent.set_destination(waypoint_end.location)

        # Generate traffic
        traffic_gen_seed = trial
        vehicles_list, walkers_list, all_id, all_actors, traffic_manager, _ = generate_traffic_func(ENV_AGGR, ENV_DENS, test_args.spawn_radius, my_vehicle.id, traffic_gen_seed)

        time_start = time.time()
        while time.time() - time_start < test_args.timeout_duration:
            world.tick()
            if agent.done():
                print("Target destination has been reached. Stopping vehicle.")
                my_vehicle.apply_control(agent.halt_stop())
                kill_traffic(vehicles_list, walkers_list, all_id, all_actors, traffic_manager)
                if test_args.orient_spectator: orient.kill()
                break

            my_vehicle.apply_control(agent.run_step())

        print(f"Trial: {trial}, Time taken: {time.time() - time_start}")
        my_vehicle.apply_control(agent.halt_stop())
        kill_traffic(vehicles_list, walkers_list, all_id, all_actors, traffic_manager)
        if test_args.orient_spectator: orient.kill()

        # Record the score of the scenario 
        SCORE_RECORDS[(method, trial)] = get_scenario_score(agent, score_args)[1]

