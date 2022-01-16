using POMDPs
using POMDPModels
# using POMDPModelTools
# using POMDPSimulators
using POMDPPolicies
using Parameters
using QMDP

include("helper_funcs.jl")

abstract type DecisionProblem end

"""
The Stop-Uncontrolled Decision Problem.
"""
@with_kw struct StopUncontrolled <: DecisionProblem
    Action_Space::AbstractDict
    State_Space::AbstractDict
    Obs_Space::AbstractDict
    Trans_Func::AbstractArray
    Obs_Func::AbstractArray
    Reward_Func::AbstractArray
    discount::Float64
    Pomdp::POMDPs.POMDP
    policy::POMDPPolicies.AlphaVectorPolicy
end

function StopUncontrolled(Action_Space, State_Space, Obs_Space, Trans_Func, Obs_Func, Reward_Func; discount=0.95, solver=QMDPSolver, max_iterations=1000000, belres=1.0e-4, verbose=false)

    # Model
    Pomdp = TabularPOMDP(Trans_Func, Reward_Func, Obs_Func, discount);

    # Solver
    solver = QMDPSolver(max_iterations=max_iterations, belres=belres, verbose=verbose)
    policy = solve(solver, Pomdp)

    return StopUncontrolled(Action_Space,
                            State_Space,
                            Obs_Space,
                            Trans_Func,
                            Obs_Func,
                            Reward_Func,
                            discount,
                            Pomdp,
                            policy)
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

get_state_desc(State_Space::Dict, s_idx::Int) = ordered_dict_keys(State_Space)[s_idx]


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
                
                elseif key_a == :edge
                    if key_s.ego_pos == :after
                        Trans_Func[val_sp, val_a, val_s] *= key_sp.ego_pos == :after ? 1.0 : 0.0

                    elseif key_sp.ego_pos == key_s.ego_pos
                        Trans_Func[val_sp, val_a, val_s] *= params.pos_stays_edge

                    elseif key_sp.ego_pos == pos_afterwards(key_s.ego_pos)
                        Trans_Func[val_sp, val_a, val_s] *= 1.0 - params.pos_stays_edge

                    else
                        Trans_Func[val_sp, val_a, val_s] *= 0.0

                    end

                elseif key_a == :go
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
                    Trans_Func[val_sp, val_a, val_s] *= (1.0 - params.aggresiveness_changes) / 2.0

                end
 
                # Transitions of `clr_line_of_sight`
                if key_s.ego_pos == :at

                    if key_s.clr_line_of_sight == :yes
                        if key_sp.clr_line_of_sight == :yes
                            Trans_Func[val_sp, val_a, val_s] *= 1.0    # (yes -> yes)
                        else
                            Trans_Func[val_sp, val_a, val_s] *= 0.0    # (yes -> no)
                        end

                    else # key_s.clr_line_of_sight == :no
                        if key_sp.clr_line_of_sight == :yes
                            if key_a == :edge || key_a == :go
                                Trans_Func[val_sp, val_a, val_s] *= params.clr_line_of_sight_changes    # (no -> yes)
                            else
                                Trans_Func[val_sp, val_a, val_s] *= 0.0    # (no -> no)
                            end

                        else # key_sp.clr_line_of_sight == :no
                            if key_a == :edge || key_a == :go
                                Trans_Func[val_sp, val_a, val_s] *= 1.0 - params.clr_line_of_sight_changes    # (no -> yes)
                            else
                                Trans_Func[val_sp, val_a, val_s] *= 1.0    # (no -> no)
                            end
                        end
                    end
                elseif key_s.clr_line_of_sight == key_sp.clr_line_of_sight
                    Trans_Func[val_sp, val_a, val_s] *= 1.0
                else
                    Trans_Func[val_sp, val_a, val_s] *= 0.0
                end

            end
        end
    end

    return normalize_Func(Trans_Func)::AbstractArray{Float64, 3}
end


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

    # Guess the blocking of rival (and the probabilities), given an observation about it.
    blk_nearest(o::Float64, params::NamedTuple) = nearest(o=o, levels=[:yes, :no], prob=params.blocking_guess)

    # Guess the clearness of sight (and the probabilities), given an observation about it.
    clr_sight_nearest(o::Float64, params::NamedTuple) = nearest(o=o, levels=[:yes, :no], prob=params.clr_line_of_sight_guess)

    
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
                st_blk, probs, rem_probs = blk_nearest(key_o.rival_blocking, params)
                for (i,j) in zip(st_blk, probs)
                    if key_s.rival_blocking == i
                        Obs_Func[val_o, val_a, val_s] *= j
                        # @show j
                    else
                        Obs_Func[val_o, val_a, val_s] *= rem_probs
                        # @show rem_probs
                    end
                end

                # Observations of `clr_line_of_sight`
                st_blk, probs, rem_probs = clr_sight_nearest(key_o.clr_line_of_sight, params)
                for (i,j) in zip(st_blk, probs)
                    if key_s.clr_line_of_sight == i
                        Obs_Func[val_o, val_a, val_s] *= j
                        # @show j
                    else
                        Obs_Func[val_o, val_a, val_s] *= rem_probs
                        # @show rem_probs
                    end
                end

                # Observations of `rival_aggresiveness`
                st_rv_aggr, probs, rem_probs = aggr_nearest(key_o.rival_aggresiveness, params)
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
    return normalize_Func(Obs_Func)::AbstractArray{Float64, 3}
end

normalize_Func(Obs_Func::AbstractArray{Float64, 3}) = Obs_Func ./ sum(Obs_Func, dims=1)
validate_Func(Obs_Func::AbstractArray{Float64, 3}, eps=1e-5) = all(1.0 + eps .> sum(Obs_Func, dims=1) .> 1.0 - eps)

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

    Reward_Func = zeros(length(State_Space), length(Action_Space))

    for (key_s, val_s) in State_Space    
        for (key_a, val_a) in Action_Space

            # Penalty of individual actions
            if key_a == :stop
                Reward_Func[val_s, val_a] = -100
            elseif key_a == :edge
                Reward_Func[val_s, val_a] = -1
            elseif key_a == :go
                Reward_Func[val_s, val_a] = 0
            end

            # Final state (desired)
            if key_s.ego_pos == :after && key_a == :go
                Reward_Func[val_s, val_a] = params.final_reward
            
            # Crash state (undesired)
            elseif key_s.ego_pos == key_s.rival_pos == :inside && key_s.rival_blocking == :yes
                Reward_Func[val_s, val_a] = params.crash_reward
            end

            # Taken over state (slightly undesired)
            if key_s.rival_blocking == :yes && key_a != :stop
                Reward_Func[val_s, val_a] += params.taken_over_reward
            end

            # Clearing line of sight when at stop sign (slightly desired)
            if key_s.clr_line_of_sight == :no && key_a == :edge
                Reward_Func[val_s, val_a] += params.clearing_sight_at_stop
            end

        end
    end

    return Reward_Func::AbstractArray{Float64, 2}
end


""" Return possible transitions from a given state and action. """
function get_transitions(s::NamedTuple, a::Symbol, State_Space, Action_Space, Trans_Func; sort_by=-1, sort_rev=true)

    s_idx = State_Space[s]
    a_idx = Action_Space[a]
    idxs = Trans_Func[:, a_idx, s_idx] .> 0.0

    sp = ordered_dict_keys(State_Space)[idxs]
    sp_probs = Trans_Func[idxs, a_idx, s_idx]
    
    data = permutedims(vcat(hcat(collect.(sp)...), sp_probs'))
    if sort_by == -1 sort_by = size(data, 2) end
    data = sortslices(data, dims=1, by=x->x[sort_by], rev=sort_rev)

    return pretty_table(data; header = vcat(collect(string.(fieldnames(s))), "Prob"), hlines=0:length(sp_probs)+1)
end


""" Return possible observations from a given state and action. """
function get_observations(s::NamedTuple, a::Symbol, State_Space, Action_Space, Obs_Space, Obs_Func; sort_by=-1, sort_rev=true)

    s_idx = State_Space[s]
    a_idx = Action_Space[a]
    idxs = Obs_Func[:, a_idx, s_idx] .> 0.0

    o = ordered_dict_keys(Obs_Space)[idxs]
    o_probs = Obs_Func[idxs, a_idx, s_idx]
    
    data = permutedims(vcat(hcat(collect.(o)...), o_probs'))
    if sort_by == -1 sort_by = size(data, 2) end
    data = sortslices(data, dims=1, by=x->x[sort_by], rev=sort_rev)

    return pretty_table(data; header = vcat(collect(string.(fieldnames(o[1]))), "Prob"), hlines=0:length(0_probs)+1)
end
