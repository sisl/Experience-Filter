module MODIA

@info "Loading Stop Uncontrolled POMDP. \nThis will take a while..."

include("stop_uncontrolled_pomdp.jl")
export
	create_Action_Space,
	create_State_Space,
	create_Obs_Space,
	define_Trans_Func,
	define_Obs_Func,
	define_Reward_Func,
	get_transitions,
	get_observations

include("instantiate_stop_uncontrolled_pomdp.jl")
export
	StopUncontrolledDP,
	uniform_belief,
	get_action_from_belief,
	DiscreteUpdater,
	update_belief

# include("plot_funcs.jl")
# export
# 	get_policy_map

include("helper_funcs.jl")
export
	printt,
	py_construct_obs,
	py_tabulate_belief

include("history_learner.jl")
export
	learn_from_data,
	concat_observations,
	most_likely_state_from_obs,
	load_saved_data

end  # module