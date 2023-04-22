# Experience Filter: Using Past Experiences on Unseen Tasks or Environments

This is the official repo for the paper:
    
    Yildiz, A., Yel, E., Corso, A., Wray, K., Witwicki, S., Kochenderfer, M., "Experience Filter: Using Past Experiences on Unseen Tasks or Environments", in IEEE Intelligent Vechicles Symposium (2023).

## Defining an Experience Filter

This is done using the class defined in `ExperienceFilter.py`. A few test scripts can be found in the `test` folder.

## Defining the POMDP in Julia

This is done using the `MODIA.jl` module. To instantiate an untrained POMDP, run the script `instantiate_stop_uncontrolled_pomdp.jl`.

## Importing a POMDP Policy to CARLA

The behavior of the ego vehicle in CARLA can be controlled using the `MODIAAgent` defined in `modia_agent.py`, which at each timestep, queries the policy solved for the defined POMDP. A few test scripts can be found in the `test` folder.

## Running Scenarios in CARLA

After [installing](https://carla.readthedocs.io/en/stable/getting_started/) CARLA, [run](https://carla.readthedocs.io/en/stable/running_simulator_standalone/) it. Then, different scenario types can be run and recorded using the `record_scenario_histories.py` script.
This operation takes some time. 

## Learning/Updating POMDP Policies

Using the seen data, POMDP policies are learned through the `history_learner.jl` script.


## Benchmarking Learned Policies

After [installing](https://carla.readthedocs.io/en/stable/getting_started/) CARLA, [run](https://carla.readthedocs.io/en/stable/running_simulator_standalone/) it. Then, using the `benchmark_policy.py`, a POMDP object instantiated in Julia is automatically imported into the Python script. To make it easier to benchmark, the data required for the Experience Filter have been stored in the `filter_data_All.pkl`, which is loaded automatically. This operation takes some time, but its results have already been stored in `benchmark_records.csv`.

## Plotting Results

To plot policy maps, the script `plot_funcs.jl` can be used. To plot scoring metrics, the script `plot_funcs.py` can be used.

## Nodes

The `nodes` folder contains scripts that act standalone to navigate easier inside CARLA. The purpose of each node is explained in its preamble.

## Troubleshooting

### I'm getting a `RuntimeError` saying that the simulator timed out.

The CARLA simulator is not installed properly; it is not being detected by the API, check the installation manual for CARLA. Or you forgot to run CARLA before running one of the Python scripts.

### I'm getting the `ImportError: MODIA not found` error from a Python script. 

You need to have the `julia.api` installed correctly in order to call Julia within Python; [this article](https://blog.esciencecenter.nl/how-to-call-julia-code-from-python-8589a56a98f2) might help. You also need to have the Julia dependencies installed for the `MODIA.jl` module to work. Try running `julia MODIA.jl` in a new terminal to check if the module itself can be compiled. Then, try running `julia instantiate_stop_uncontrolled_pomdp.jl` to check if the POMDP can be created and solved properly.