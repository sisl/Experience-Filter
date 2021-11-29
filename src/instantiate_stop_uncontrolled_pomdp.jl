using BeliefUpdaters: DiscreteBelief, DiscreteUpdater, uniform_belief, update
using POMDPPolicies: action
using QMDP    # , SARSOP

get_action_from_belief = POMDPPolicies.action
update_belief = update

# A-Space
A_vals = (:stop, :edge, :go)
Action_Space = create_Action_Space(A_vals)

# S-Space
S_ids = (:ego_pos, :rival_pos, :rival_blocking, :rival_aggresiveness)
vehicle_pos_vals = (:before, :at, :inside, :after)
binary_vals = (:yes, :no)
rival_aggsv_vals = (:cautious, :normal, :aggressive)
S_vals = [vehicle_pos_vals, vehicle_pos_vals, binary_vals, rival_aggsv_vals]
State_Space = create_State_Space(S_ids, S_vals)

# O-Space
O_ids = (:ego_pos, :rival_pos, :rival_blocking, :rival_vel)
pos_max = length(vehicle_pos_vals)
pos_increment = 1
vel_min = 1
vel_max = 3
vel_increment = 1
O_ran = [range(1, pos_max, step=pos_increment), range(1, pos_max, step=pos_increment), range(1, 2, step=pos_increment), range(vel_min, vel_max, step=vel_increment)]
Obs_Space = create_Obs_Space(O_ids, O_ran)

# T-Func
TF_params = (pos_stays=0.80, pos_stays_edge = 0.97, blocking_changes=0.20, aggresiveness_changes=0.00, aggresiveness_stays=0.70)
Trans_Func = define_Trans_Func(State_Space, Action_Space, TF_params)

# Z-Func
OF_params = (pos_guess = 0.99, blocking_guess = 0.99, aggr_guess = 0.99)    # TODO: This is fully observable. Change this.
Obs_Func = define_Obs_Func(Obs_Space, Action_Space, State_Space, OF_params)

# R-Func
RF_params = (final_reward = 2500, crash_reward = -10000, taken_over_reward = -1000)
Reward_Func = define_Reward_Func(State_Space, Action_Space, RF_params)

StopUncontrolledDP = StopUncontrolled(Action_Space,
                                      State_Space,
                                      Obs_Space,
                                      Trans_Func,
                                      Obs_Func,
                                      Reward_Func)


# for (s, a, r) in stepthrough(StopUncontrolledDP.pomdp, StopUncontrolledDP.policy, "s,a,r", max_steps=10)
#     printt(get_state_desc(State_Space, s))
#     @show a
#     @show r
#     println()
# end

# function uniform_belief(pomdp)
#     state_list = ordered_states(pomdp)
#     ns = length(state_list)
#     return DiscreteBelief(pomdp, state_list, ones(ns) / ns)
# end

# Policy can be used to map belief to actions
b = uniform_belief(StopUncontrolledDP.Pomdp);
a = get_action_from_belief(StopUncontrolledDP.policy, b);
bu = DiscreteUpdater(StopUncontrolledDP.Pomdp);
# bp = update_belief(bu, b, a_idx, o_idx);