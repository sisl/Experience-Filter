using Dates
using JLD

"""
    Learn from data, and update the T(s'|a,s) function.
    Treats the state with the highest belief probability obtained from the observation as the actual state.
"""
function learn_from_data(act_histories::AbstractArray, obs_histories::AbstractArray, StopUncontrolledDP; prior_scenario_count = 100, save_data=false, save_output=false)
    timestamp = Dates.now()
    if save_data JLD.save(string(timestamp) * "_data.jld", "act_histories", act_histories, "obs_histories", obs_histories) end
    
    State_Space = StopUncontrolledDP.State_Space
    Trans_Func = StopUncontrolledDP.Trans_Func * prior_scenario_count
    act_histories = Int.(act_histories)
    obs_histories = concat_observations(obs_histories, StopUncontrolledDP.Obs_Space)

    for (_, hist) in obs_histories
        for (idx, item) in enumerate(hist[1:end-1])
            time, obs = item
            a = act_histories[time]
            s = most_likely_state_from_obs(obs, State_Space)
            _, obsp = hist[idx+1]
            sp = most_likely_state_from_obs(obsp, State_Space)
            Trans_Func[sp,a,s] += 1
        end
    end

    result = normalize_Func(Trans_Func)
    if save_output JLD.save(string(timestamp) * "_output.jld", "Trans_Func", result) end
    return result
end

"""Concatenate observations histories of each individual rival throughout time."""
function concat_observations(obs_histories::AbstractArray, Obs_Space::Dict)
    Obs_Space = reverse_dict(Obs_Space)
    
    # Pre-allocation
    kys = [keys(d) for d in obs_histories]
    histories = Dict(k=>Tuple[] for k in union(kys...))

    for (time,item) in enumerate(obs_histories)
        for (rival_id, obs) in item
            push!(histories[rival_id], (time, Obs_Space[obs]))
        end
    end
    return histories::Dict
end

"""Find most likely state from observation."""
function most_likely_state_from_obs(obs::NamedTuple, State_Space::Dict)
    vehicle_pos_vals = Dict(1.0 => :before, 2.0 => :at, 3.0 => :inside, 4.0 => :after)
    binary_vals = Dict(1.0 => :yes, 2.0 => :no)
    rival_aggsv_vals = Dict(1.0 => :cautious, 2.0 => :normal, 3.0 => :aggressive)
    
    names = (:ego_pos, :rival_pos, :rival_blocking, :rival_aggresiveness)
    vals = (vehicle_pos_vals[obs.ego_pos],
            vehicle_pos_vals[obs.rival_pos],
            binary_vals[obs.rival_blocking],
            rival_aggsv_vals[obs.rival_vel])
    return State_Space[NamedTuple{Symbol.(names)}(vals)]::Int
end

"""Load data, that was saved through `learn_from_data`."""
function load_saved_data(filename::String, is_output=false)    
    d = JLD.load(filename)
    return is_output ? d["Trans_Func"] : (d["act_histories"], d["obs_histories"])
end