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
    filename = f"benchmark_dev_Feb13_v1.csv"

    num_of_trials = 15
    timeout_duration = 30.0
    spawn_radius = 100.0
    orient_spectator = False
    verbose_belief = False

    datapoint_to_benchmark = (0, 40, 2)
    datapoint_to_benchmark_nrmz = (0, 1, 0.5)
    env_aggressiveness_levels = {1: PresetScenarios.CAUTIOUS, 2: PresetScenarios.NORMAL, 3: PresetScenarios.AGGRESSIVE} 

    training_effort = list(range(3, 16, 3))   # 3, 6, ..., 15


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
#     discomfort = -3.0    # lower is better
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

# Orient the spectator w.r.t. `my_vehicle.id`
if test_args.orient_spectator: orient = subprocess.Popen(['./nodes/orient_spectator.py', '-a', str(my_vehicle.id)])

# Import Experience Filter
from ExperienceFilter import *
from expfilter_helper_funcs import *
filter_data_All = load_with_pkl("filter_data_All.pkl")    # filter_data_All = get_filter_data(filter_args)
EF_all = ExperienceFilter(data=filter_data_All, axeslabels=filter_args.axeslabels)
EF_all.remove_datapoint(test_args.datapoint_to_benchmark)
datapoints_normalized_All = EF_all.datapoints_normalized

def already_logged(tef, method, trial):
    if not glob(test_args.filename): return False
    logs = read_log_from_file(test_args.filename)
    logs_relevant = logs[:, 0:3]
    return [str(tef), method, str(trial)] in logs_relevant.tolist()

def main_running_loop(method=None):
    for trial in tqdm(range(test_args.num_of_trials), desc=f"Method running: {method}"):

        if already_logged(tef, method, trial):
            continue

        ENV_OBSV, ENV_DENS, ENV_AGGR = test_args.datapoint_to_benchmark
        ENV_AGGR = test_args.env_aggressiveness_levels[ENV_AGGR]

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
                break

            my_vehicle.apply_control(agent.run_step())

        print(f"Trial: {trial}, Time taken: {time.time() - time_start}")
        my_vehicle.apply_control(agent.halt_stop())
        kill_traffic(vehicles_list, walkers_list, all_id, all_actors, traffic_manager)

        # Record the score of the scenario 
        score = get_scenario_score(agent)
        scenario_log = [tef, method, trial, score['safety'], score['discomfort'], score['time']]
        log_to_file(test_args.filename, scenario_log)
    return



for tef in test_args.training_effort:
    # # Only use the subset of the recorded data, from the points that offer the maximum coverage, given their amount
    # covpts_norm = get_coverage_points(datapoints_normalized_All, tef)
    # covpts_indx = [datapoints_normalized_All.index(item) for item in covpts_norm]
    # covpts = [EF_all.datapoints[i] for i in covpts_indx]
    # filter_data_subset = {k:v for (k,v) in filter_data_All.items() if k in covpts}

    covpts_norm = get_furthest_points(datapoints_normalized_All, test_args.datapoint_to_benchmark_nrmz, tef)
    covpts_indx = [datapoints_normalized_All.index(item) for item in covpts_norm]
    covpts = [EF_all.datapoints[i] for i in covpts_indx]
    filter_data_subset = {k:v for (k,v) in filter_data_All.items() if k in covpts}

    # Assume: Trained with all data
    T_all = np.mean(list(filter_data_subset.values()), axis=0)
    T_all = MODIA.normalize_Func(T_all)
    StopUncontrolledDP_new = create_DP_from_Trans_func(T_all)
    main_running_loop(method="all")

    # Assume: Trained (self)
    T_self = filter_data_All[test_args.datapoint_to_benchmark]
    StopUncontrolledDP_new = create_DP_from_Trans_func(T_self)
    main_running_loop(method="self")
    
    # Assume: Experince Filter
    EF = ExperienceFilter(data=filter_data_subset, axeslabels=filter_args.axeslabels)
    T_expfilter = EF.apply_filter(new_point=test_args.datapoint_to_benchmark)
    StopUncontrolledDP_new = create_DP_from_Trans_func(T_expfilter)
    main_running_loop(method="expfilter")

    # Assume: Nearest neighbor
    _, nn = get_nearest_neighbor(covpts, test_args.datapoint_to_benchmark)
    T_nn = filter_data_All[nn]
    StopUncontrolledDP_new = create_DP_from_Trans_func(T_nn)
    main_running_loop(method="nn")

    if test_args.orient_spectator: orient.kill()