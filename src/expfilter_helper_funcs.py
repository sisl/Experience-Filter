import os
from tqdm import tqdm
from glob import glob
from helper_funcs import *

# You must have connected to julia.api in the main script for the imports below to work!
from julia import MODIA
from julia import Base as jlBase

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
                    from julia import Plots
                    StopUncontrolledDP_new = create_DP_from_Trans_func(val)
                    plt = MODIA.get_policy_map(StopUncontrolledDP_new, rival_aggressiveness=key3); Plots.savefig(plt, f"Obsv_{key1}_Dens_{key2}_Aggr_{key3}.png")

    os.chdir(filter_args.working_dir)
    return filter_data

def get_accumulative_filter_data(filter_args):
    os.chdir(filter_args.rel_path_to_pkls)
    list_of_loadnames = glob(f"*.pkl")
    L = load_many_with_pkl(list_of_loadnames)

    val = MODIA.learn_from_data(L[0], L[1], MODIA.StopUncontrolledDP, prior_scenario_count=filter_args.prior_scenario_count)    
    os.chdir(filter_args.working_dir)
    return val

def examine_filter_data():
    class Params:    
        rel_path_to_pkls = "./dev_train_v2/"
        working_dir = os.getcwd()
        env_observability_settings = {"low": 0 , "high": 1}
        env_density_settings = {"low": 5, "med": 15, "high": 40}
        env_aggressiveness_settings = {"aggressive": 3} #"cautious": 1, "normal": 2, "aggressive": 3}

    params = Params()
    filter_data = dict()
    os.chdir(params.rel_path_to_pkls)

    for (key1, ENV_OBSV) in tqdm(params.env_observability_settings.items(), desc="Env Observability"):
        for (key2, ENV_DENS) in tqdm(params.env_density_settings.items(), desc="Env Density"):
            for (key3, ENV_AGGR) in tqdm(params.env_aggressiveness_settings.items(), desc="Env Aggressiveness"):

                list_of_loadnames = glob(f"Obsv_{key1}_Dens_{key2}_Aggr_{key3}*.pkl")
                L = load_many_with_pkl(list_of_loadnames)
                val = MODIA.learn_from_data(L[0], L[1], MODIA.StopUncontrolledDP, prior_scenario_count=0, normalize=False)
                val_idx = np.nonzero(val)
                counts = val[val_idx]
                argsrt = np.flip(np.argsort(counts))

                s_names  = [jlBase.collect(reverse_dict(MODIA.State_Space)[item+1]) for item in val_idx[0]]
                a_names  = [reverse_dict(MODIA.Action_Space)[item+1] for item in val_idx[1]]
                sp_names = [jlBase.collect(reverse_dict(MODIA.State_Space)[item+1]) for item in val_idx[2]]
        
                table_data = np.array([s_names, a_names, sp_names, counts])

                s_ids = "ego_pos, rival_pos, rival_blocking, rival_aggressiveness, clr_line_of_sight"
                # MODIA.pretty_table(table_data.T.tolist(), [s_ids, "a", s_ids, "counts"])
                print(f"Obsv_{key1}_Dens_{key2}_Aggr_{key3}")
                MODIA.pretty_table(table_data[:,argsrt].T.tolist(), [s_ids, "a", s_ids, "counts"])
                f = input("Press any key to continue...")
                if f=='q': break

    os.chdir(params.working_dir)
    return filter_data

def learn_all_data(filter_args):
    os.chdir(filter_args.rel_path_to_pkls)
    list_of_loadnames = glob(f"*.pkl")
    L = load_many_with_pkl(list_of_loadnames)

    T = MODIA.learn_from_data(L[0], L[1], MODIA.StopUncontrolledDP, prior_scenario_count=filter_args.prior_scenario_count)
    
    if filter_args.plot_policy_maps:
        from julia import Plots
        StopUncontrolledDP_new = create_DP_from_Trans_func(T)
        plt = MODIA.get_policy_map(StopUncontrolledDP_new, rival_aggressiveness="N/A"); Plots.savefig(plt, f"All.png")

    os.chdir(filter_args.working_dir)
    return T

def create_DP_from_Trans_func(T):
    return MODIA.StopUncontrolled(MODIA.StopUncontrolledDP.Action_Space,
                                    MODIA.StopUncontrolledDP.State_Space,
                                    MODIA.StopUncontrolledDP.Obs_Space,
                                    T,
                                    MODIA.StopUncontrolledDP.Obs_Func,
                                    MODIA.StopUncontrolledDP.Reward_Func)
    