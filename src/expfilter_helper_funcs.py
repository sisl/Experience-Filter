import os
from tqdm import tqdm
from glob import glob
from helper_funcs import *

# You must have connected to julia.api in the main script for the imports below to work!
from julia import MODIA
from julia import Base as jlBase
from julia import Plots

def get_filter_data(filter_args):
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
                    StopUncontrolledDP_new = create_DP_from_Trans_func(val)
                    plt = MODIA.get_policy_map(StopUncontrolledDP_new, rival_aggressiveness=key3); Plots.savefig(plt, f"Obsv_{key1}_Dens_{key2}_Aggr_{key3}.png")

    os.chdir(filter_args.working_dir)
    return filter_data

def create_DP_from_Trans_func(T):
    return MODIA.StopUncontrolled(MODIA.StopUncontrolledDP.Action_Space,
                                    MODIA.StopUncontrolledDP.State_Space,
                                    MODIA.StopUncontrolledDP.Obs_Space,
                                    T,
                                    MODIA.StopUncontrolledDP.Obs_Func,
                                    MODIA.StopUncontrolledDP.Reward_Func)
    