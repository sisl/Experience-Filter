using POMDPs
using POMDPPolicies: ordered_states
using BeliefUpdaters: DiscreteBelief
using Plots
using ProgressBars

function get_policy_map(pomdp::POMDPs.POMDP, pol::POMDPs.Policy, State_Space; ego_pos=:at, rival_aggresiveness=:normal, show_map_only=true)
    
    get_state_pos(idx::Int) = [:before, :at, :inside, :after][idx]
    get_state_blk(idx::Int) = [:yes, :no][idx]
    
    X = rival_blocking_range = collect(1 : 0.001 : 2)
    Y = rival_pos_range = collect(1 : 0.001 : 4)
    Z = zeros(length(Y), length(X))
    
    # Loop through `rival_pos` and `rival_blocking`
    for (j_key, j) in ProgressBars.tqdm(enumerate(rival_blocking_range))
        for (i_key, i) in enumerate(rival_pos_range)
            # *_pct is the probability of *_cl
            i_pct, i_cl, i_fl = i %1, ceil(Int, i), floor(Int, i)
            j_pct, j_cl, j_fl = j %1, ceil(Int, j), floor(Int, j)
            s_pct = [i_pct*j_pct, (1-i_pct)*j_pct, i_pct*(1-j_pct), (1-i_pct)*(1-j_pct)]

            s1 = (ego_pos=ego_pos, rival_pos=get_state_pos(i_cl), rival_blocking=get_state_blk(j_cl), rival_aggresiveness=rival_aggresiveness)
            s2 = (ego_pos=ego_pos, rival_pos=get_state_pos(i_fl), rival_blocking=get_state_blk(j_cl), rival_aggresiveness=rival_aggresiveness)
            s3 = (ego_pos=ego_pos, rival_pos=get_state_pos(i_cl), rival_blocking=get_state_blk(j_fl), rival_aggresiveness=rival_aggresiveness)
            s4 = (ego_pos=ego_pos, rival_pos=get_state_pos(i_fl), rival_blocking=get_state_blk(j_fl), rival_aggresiveness=rival_aggresiveness)

            s_list = [State_Space[i] for i in [s1, s2, s3, s4]]
            state_list = ordered_states(pomdp)
            state_probs = zeros(length(state_list))
            state_probs[s_list] = s_pct

            b = DiscreteBelief(pomdp, state_list, state_probs)
            a = POMDPs.action(pol, b)
            Z[i_key, j_key] = a
        end
    end

    plt = Plots.heatmap(X, Y, Z, title="Ego Position: $ego_pos, Rival Aggr: $rival_aggresiveness")
    xlabel!(plt, "Rival Blocking")
    ylabel!(plt, "Rival Position")
    yticks!(plt, [1,2,3,4], ["before", "at", "inside", "after"])
    xticks!(plt, [1,2], ["yes", "no"])

    return show_map_only ? plt : (X, Y, Z, plt)

end

get_policy_map(DP; ego_pos=:at, rival_aggresiveness=:normal, show_map_only=true) = get_policy_map(DP.Pomdp, DP.policy, DP.State_Space; ego_pos=ego_pos, rival_aggresiveness=rival_aggresiveness, show_map_only=show_map_only)