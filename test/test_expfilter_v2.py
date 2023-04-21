import sys, os
from helper_funcs import *
sys.path.append('../src/')
from ExperienceFilter import ExperienceFilter
import numpy as np
from glob import glob
from tqdm import tqdm

connect_julia_api = True
if connect_julia_api:
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import Pkg
    from julia import Plots
    from julia import Base as jlBase
    Pkg.activate("../")    # while current dir is /src
    from julia import MODIA
    StopUncontrolledDP = MODIA.StopUncontrolledDP
    uniform_belief = MODIA.uniform_belief

class FilterArguments:
    axeslabels = ("Observability", "Density", "Aggressiveness")
    env_observability_settings = {"low": 0 , "high": 1}
    env_density_settings = {"low": 5, "med": 15, "high": 40}
    env_aggressiveness_settings = {"cautious": 1, "normal": 2, "aggressive": 3}

    working_dir = os.getcwd()
    rel_path_to_pkls = "./dev_train_v2/"
    prior_scenario_count = 2000
    plot_policy_maps = False    # if True, you need to enable `plot_funcs.jl` in MODIA.jl.

filter_args = FilterArguments()
filter_data = dict()
os.chdir(filter_args.rel_path_to_pkls)

for (key1, ENV_OBSV) in tqdm(filter_args.env_observability_settings.items(), desc="Env Observability"):
    for (key2, ENV_DENS) in tqdm(filter_args.env_density_settings.items(), desc="Env Density"):
        for (key3, ENV_AGGR) in tqdm(filter_args.env_aggressiveness_settings.items(), desc="Env Aggressiveness"):

            list_of_loadnames = glob(f"Obsv_{key1}_Dens_{key2}_Aggr_{key3}*.pkl")
            L = load_many_with_pkl(list_of_loadnames)

            key = (filter_args.env_observability_settings[key1], filter_args.env_density_settings[key2], filter_args.env_aggressiveness_settings[key3])
            val = MODIA.learn_from_data(L[0], L[1], MODIA.StopUncontrolledDP, prior_scenario_count=filter_args.prior_scenario_count)
            filter_data[key] = val
            
            if filter_args.plot_policy_maps:
                StopUncontrolledDP_new = MODIA.StopUncontrolled(StopUncontrolledDP.Action_Space,
                                                    StopUncontrolledDP.State_Space,
                                                    StopUncontrolledDP.Obs_Space,
                                                    val,
                                                    StopUncontrolledDP.Obs_Func,
                                                    StopUncontrolledDP.Reward_Func)

                plt = MODIA.get_policy_map(StopUncontrolledDP_new, rival_aggressiveness=key3); Plots.savefig(plt, f"Obsv_{key1}_Dens_{key2}_Aggr_{key3}.png")

os.chdir(filter_args.working_dir)
EF = ExperienceFilter(data=filter_data, axeslabels=filter_args.axeslabels)

new_point = (0.5, 10, 3)
T = EF.apply_filter(new_point=new_point)
