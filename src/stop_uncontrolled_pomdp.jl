using POMDPs
using POMDPModels
using POMDPModelTools
using POMDPSimulators
using POMDPPolicies
using BeliefUpdaters: DiscreteBelief, DiscreteUpdater
using SARSOP
using Parameters

include("helper_funcs.jl")
include("plot_funcs.jl")

abstract type DecisionProblem end

"""
The Stop-Uncontrolled Decision Problem.
"""
@with_kw struct StopUncontrolledDP <: DecisionProblem
    Action_Space::AbstractDict
    State_Space::AbstractDict
    Obs_Space::AbstractDict
    Trans_Func::AbstractArray
    Obs_Func::AbstractArray
    Reward_Func::AbstractArray
    Pomdp::POMDPs.POMDP
    discount::Float64
end


"""
Create an Action Space for a DP.

# Input 
- `A_vals` : possible actions
# Output 
- `Action_Space` : Dict{Symbol, Int} where the values are indexes of each symbol in the input
"""
function create_Action_Space(A_vals::Tuple{Vararg{Symbol}})
    Action_Space = Dict{Symbol, Int}()
    for (idx, item) in enumerate(A_vals)
        push!(Action_Space, item=>idx)
    end
    return Action_Space::Dict{Symbol, Int}
end

get_action_idx(DP::DecisionProblem, a::Symbol) = DP.Action_Space[a]


"""
Create a State Space for a DP.

# Input 
- `S_ids` : IDs of possible states
- `S_vals` : vals for each ID
# Output 
- `State_Space` : Dict{NTuple{Symbol}, Int} where the values are indexes of each symbol in the input
"""
function create_State_Space(S_ids::Tuple{Vararg{Symbol}}, S_vals::Vector)
    State_Space = Dict{NamedTuple{S_ids,NTuple{length(S_ids),Symbol}}, Int}()
    producted = vec(collect(Base.product(S_vals...)))

    for (idx, item) in enumerate(producted)
        push!(State_Space, (;zip(S_ids, item)...)=>idx)
    end
    return State_Space::Dict{NamedTuple{S_ids,NTuple{length(S_ids),Symbol}}, Int}
end

get_state_desc(DP::DecisionProblem, s::Int) = DP.State_Space[s]


"""
Create an Observation Space for a DP.

# Input 
- `O_ids` : IDs of possible observations
- `O_ran` : range of interval values for each ID, should be a list of `AbstractRange`
# Output 
- `Obs_Space` : Dict{NTuple{Symbol}, Int} where the values are indexes of each symbol in the input
"""
function create_Obs_Space(O_ids::Tuple{Vararg{Symbol}}, O_ran::Vector)
    Obs_Space = Dict{NamedTuple{O_ids,NTuple{length(O_ids),Float64}}, Int}()
    producted = vec(collect(Base.product(collect(O_ran)...)))

    for (idx, item) in enumerate(producted)
        push!(Obs_Space, (;zip(O_ids, item)...)=>idx)
    end
    return Obs_Space::Dict{NamedTuple{O_ids,NTuple{length(O_ids),Float64}}, Int}
end

get_obs_idx(DP::DecisionProblem, o::Symbol) = DP.Obs_Space[o]


"""
Define the Transition Function for the Stop-Uncontrolled DP.

# Input 
- `State_Space` : state-space of DP
- `Action_Space` : action-space of DP
- `params` : params that define default transition probabilities
# Output 
- `Trans_Func` : array `T` of dims |S|x|A|x|S| where the T[s',a,s] = p(s'|a,s)
"""
function define_Trans_Func(State_Space::Dict, Action_Space::Dict, params::NamedTuple)

    Trans_Func = ones(length(State_Space), length(Action_Space), length(State_Space))

    function pos_afterwards(pos::Symbol)
        poses = [:before, :at, :inside, :after]
        pos_dict = Dict(poses .=> collect(1:length(poses)))
        try
            return poses[pos_dict[pos]+1]
        catch
            @error "You called with pos = :after, which should not have happened."
        end
    end

    function blocking_opposite(blk::Symbol)
        if blk == :yes
            return :no
        else
            return :yes
        end
    end

    for (key_s, val_s) in State_Space
        for (key_a, val_a) in Action_Space
            for (key_sp, val_sp) in State_Space

                # Transitions of `ego_pos`
                if key_a == :stop
                    Trans_Func[val_sp, val_a, val_s] *= key_sp.ego_pos == key_s.ego_pos ? 1.0 : 0.0

                else    # action is not stop
                    if key_s.ego_pos == :after
                        Trans_Func[val_sp, val_a, val_s] *= key_sp.ego_pos == :after ? 1.0 : 0.0

                    elseif key_sp.ego_pos == key_s.ego_pos
                        Trans_Func[val_sp, val_a, val_s] *= params.pos_stays

                    elseif key_sp.ego_pos == pos_afterwards(key_s.ego_pos)
                        Trans_Func[val_sp, val_a, val_s] *= 1.0 - params.pos_stays

                    else
                        Trans_Func[val_sp, val_a, val_s] *= 0.0

                    end
                end

                # Transitions of `rival_pos`
                if key_s.rival_pos == :after
                    Trans_Func[val_sp, val_a, val_s] *= key_sp.rival_pos == :after ? 1.0 : 0.0

                elseif key_sp.rival_pos == key_s.rival_pos
                    Trans_Func[val_sp, val_a, val_s] *= params.pos_stays

                elseif key_sp.rival_pos == pos_afterwards(key_s.rival_pos)
                    Trans_Func[val_sp, val_a, val_s] *= 1.0 - params.pos_stays

                else
                    Trans_Func[val_sp, val_a, val_s] *= 0.0

                end

                # Transitions of `rival_blocking`
                if key_sp.rival_blocking == blocking_opposite(key_s.rival_blocking)
                    Trans_Func[val_sp, val_a, val_s] *= params.blocking_changes

                else
                    Trans_Func[val_sp, val_a, val_s] *= 1.0 - params.blocking_changes

                end

                # Transitions of `rival_aggresiveness`
                if key_sp.rival_aggresiveness != key_s.rival_aggresiveness
                    Trans_Func[val_sp, val_a, val_s] *= params.aggresiveness_changes

                else
                    Trans_Func[val_sp, val_a, val_s] *= params.aggresiveness_stays

                end
 
            end
        end
    end

    return Trans_Func::AbstractArray{Float64, 3}
end

validate_Trans_Func(Trans_Func::AbstractArray{Float64, 3}, eps=1e-5) = all(1.0 + eps .> sum(Trans_Func, dims=1) .> 1.0 - eps)

get_trans_prob(DP::DecisionProblem, sp::NamedTuple, a::Symbol, s::NamedTuple) = DP.Trans_Func[DP.State_Space[sp],DP.Action_Space[a],DP.State_Space[s]]
get_trans_prob(Trans_Func::AbstractArray{Float64, 3}, S_space::Dict, A_space::Dict, sp::NamedTuple, a::Symbol, s::NamedTuple) = Trans_Func[S_space[sp],A_space[a],S_space[s]]


"""
Define the Observation Function for the Stop-Uncontrolled DP.

# Input 
- `Obs_Space` : observation-space of DP
- `Action_Space` : action-space of DP
- `State_Space` : state-space of DP
- `params` : params that define default observation probabilities
# Output 
- `Obs_Func` : array `Z` of dims |O|x|A|x|S| where the Z[o,a,s] = p(o|a,s)
"""
function define_Obs_Func(Obs_Space::Dict, Action_Space::Dict, State_Space::Dict, params::NamedTuple)

    Obs_Func = ones(length(Obs_Space), length(Action_Space), length(State_Space))

    function nearest(;o::Float64, levels::Vector, prob::Float64)
        levels_dict = Dict(collect(1:length(levels)) .=> levels)
        f, c = clamp(floor(Int, o), 1, length(levels)), clamp(ceil(Int, o), 1, length(levels))
        rem_probs = (1.0 - prob) / (length(levels)-1)

        if f != c    # `o` is a middle value
            return [levels_dict[f], levels_dict[c]], [prob/2, prob/2], rem_probs
        else
            return [levels_dict[f]], [prob], rem_probs
        end
    end

    # Return the nearest positions (and the probabilities), given observation about it.
    pos_nearest(o::Float64, params::NamedTuple) = nearest(o=o, levels=[:before, :at, :inside, :after], prob=params.pos_guess)

    # Guess the aggressiveness of rival (and the probabilities), given an observation of the rival's speed.
    aggr_nearest(o::Float64, params::NamedTuple) = nearest(o=o, levels=[:cautious, :normal, :aggressive], prob=params.aggr_guess)

    
    for (key_o, val_o) in Obs_Space
        for (key_a, val_a) in Action_Space
            for (key_s, val_s) in State_Space

                # Observations of `ego_pos`
                st_egos, probs, rem_probs = pos_nearest(key_o.ego_pos, params)
                for (i,j) in zip(st_egos, probs)
                    if key_s.ego_pos == i
                        Obs_Func[val_o, val_a, val_s] *= j
                        # @show j
                    else
                        Obs_Func[val_o, val_a, val_s] *= rem_probs
                        # @show rem_probs
                    end
                end

                # Observations of `rival_pos`
                st_rivals, probs, rem_probs = pos_nearest(key_o.rival_pos, params)
                for (i,j) in zip(st_rivals, probs)
                    if key_s.rival_pos == i
                        Obs_Func[val_o, val_a, val_s] *= j
                        # @show j
                    else
                        Obs_Func[val_o, val_a, val_s] *= rem_probs
                        # @show rem_probs
                    end
                end

                # Observations of `rival_blocking`
                for (i,j) in zip(st_egos, st_rivals)
                    if (i == j == :inside) || (i == j == :after)
                        Obs_Func[val_o, val_a, val_s] *= key_s.rival_blocking == :yes ? params.blocking_prob : 1.0 - params.blocking_prob 
                    else
                        Obs_Func[val_o, val_a, val_s] *= key_s.rival_blocking == :yes ? 0.0 : 1.0
                    end
                end

                # Observations of `rival_aggresiveness`
                st_rv_aggr, probs, rem_probs = aggr_nearest(key_o.rival_vel, params)
                for (i,j) in zip(st_rv_aggr, probs)
                    if key_s.rival_aggresiveness == i
                        Obs_Func[val_o, val_a, val_s] *= j
                        # @show j
                    else
                        Obs_Func[val_o, val_a, val_s] *= rem_probs
                        # @show rem_probs
                    end
                end

            end
        end
    end

    # return Obs_Func::AbstractArray{Float64, 3}
    return normalize_Obs_Func(Obs_Func)::AbstractArray{Float64, 3}
end

normalize_Obs_Func(Obs_Func::AbstractArray{Float64, 3}) = Obs_Func ./ sum(Obs_Func, dims=1)
validate_Obs_Func(Obs_Func::AbstractArray{Float64, 3}, eps=1e-5) = all(1.0 + eps .> sum(Obs_Func, dims=1) .> 1.0 - eps)

get_obs_prob(DP::DecisionProblem, o::NamedTuple, a::Symbol, s::NamedTuple) = DP.Obs_Func[DP.Obs_Space[o],DP.Action_Space[a],DP.State_Space[s]]
get_obs_prob(Obs_Func::AbstractArray{Float64, 3}, O_space::Dict, A_space::Dict, S_space::Dict, o::NamedTuple, a::Symbol, s::NamedTuple) = Obs_Func[O_space[o],A_space[a],S_space[s]]



"""
Define the Reward Function for the Stop-Uncontrolled DP.

# Input 
- `State_Space` : state-space of DP
- `Action_Space` : action-space of DP
- `params` : params that define default rewards
# Output 
- `Reward_Func` : array `R` of dims |S|x|A| where the R[s,a] is the immediate reward
"""
function define_Reward_Func(State_Space::Dict, Action_Space::Dict, params::NamedTuple)

    Reward_Func = -1 * ones(length(State_Space), length(Action_Space))

    for (key_s, val_s) in State_Space    
        for (key_a, val_a) in Action_Space

            # Final state (desired)
            if key_s.ego_pos == :after
                Reward_Func[val_s, val_a] = params.final_reward
            end

            # Crash state (undesired)
            if key_s.ego_pos == key_s.rival_pos == :inside && key_s.rival_blocking == :yes
                Reward_Func[val_s, val_a] = params.crash_reward
            end

            # Taken over state (slightly undesired)
            if key_s.rival_blocking == :yes
                Reward_Func[val_s, val_a] = params.taken_over_reward
            end

        end
    end

    return Reward_Func::AbstractArray{Float64, 2}
end


""" Return possible transitions from a given state and action. """
function get_transitions(s::NamedTuple, a::Symbol, State_Space, Action_Space, Trans_Func; sort_by=5, sort_rev=true)

    s_idx = State_Space[s]
    a_idx = Action_Space[a]
    idxs = Trans_Func[:, a_idx, s_idx] .> 0.0

    sp = ordered_dict_keys(State_Space)[idxs]
    sp_probs = Trans_Func[idxs, a_idx, s_idx]
    
    data = permutedims(vcat(hcat(collect.(sp)...), sp_probs'))
    data = sortslices(data, dims=1, by=x->x[sort_by], rev=sort_rev)

    return pretty_table(data; header = vcat(collect(string.(fieldnames(s))), "Prob"), hlines=0:length(idxs))
end


""" Return possible observations from a given state and action. """
function get_observations(s::NamedTuple, a::Symbol, State_Space, Action_Space, Obs_Space, Obs_Func; sort_by=4, sort_rev=true)

    s_idx = State_Space[s]
    a_idx = Action_Space[a]
    idxs = Obs_Func[:, a_idx, s_idx] .> 0.0

    o = ordered_dict_keys(Obs_Space)[idxs]
    o_probs = Obs_Func[idxs, a_idx, s_idx]
    
    data = permutedims(vcat(hcat(collect.(o)...), o_probs'))
    data = sortslices(data, dims=1, by=x->x[sort_by], rev=sort_rev)

    return pretty_table(data; header = vcat(collect(string.(fieldnames(o[1]))), "Prob"), hlines=0:length(idxs))
end












# Define a Stop-Uncontrolled DP POMDP
A_vals = (:stop, :edge, :go)
Action_Space = create_Action_Space(A_vals)

S_ids = (:ego_pos, :rival_pos, :rival_blocking, :rival_aggresiveness)
vehicle_pos_vals = (:before, :at, :inside, :after)
binary_vals = (:yes, :no)
rival_aggsv_vals = (:cautious, :normal, :aggressive)
S_vals = [vehicle_pos_vals, vehicle_pos_vals, binary_vals, rival_aggsv_vals]
State_Space = create_State_Space(S_ids, S_vals)

O_ids = (:ego_pos, :rival_pos, :rival_vel)
pos_max = length(vehicle_pos_vals)
pos_increment = 1
vel_min = 1
vel_max = 3
vel_increment = 1
O_ran = [range(1, pos_max, step=pos_increment), range(1, pos_max, step=pos_increment), range(vel_min, vel_max, step=vel_increment)]
Obs_Space = create_Obs_Space(O_ids, O_ran)

TF_params = (pos_stays=0.66, blocking_changes=0.20, aggresiveness_changes=0.20, aggresiveness_stays=0.60)
Trans_Func = define_Trans_Func(State_Space, Action_Space, TF_params)

OF_params = (pos_guess = 0.70, blocking_prob = 0.70, aggr_guess = 0.60)
Obs_Func = define_Obs_Func(Obs_Space, Action_Space, State_Space, OF_params)

RF_params = (final_reward = 10000, crash_reward = -10000, taken_over_reward = -100)
Reward_Func = define_Reward_Func(State_Space, Action_Space, RF_params)


# Model
discount = 0.95
pomdp = TabularPOMDP(Trans_Func, Reward_Func, Obs_Func, discount);

# Stepthrough
solver = SARSOPSolver()
policy = solve(solver, pomdp)

for (s, a, r) in stepthrough(pomdp,policy, "s,a,r", max_steps=10)
    @show get_state_desc(State_Space, s)
    @show a
    @show r
    println()
end

function uniform_belief(pomdp)
    state_list = ordered_states(pomdp)
    ns = length(state_list)
    return DiscreteBelief(pomdp, state_list, ones(ns) / ns)
end

# Policy can be used to map belief to actions
b = uniform_belief(pomdp) # from POMDPModelTools
a = action(policy, b)
bu = DiscreteUpdater(pomdp)

