using PrettyTables

concat_syms(a::Symbol, b::Symbol) = Symbol(string(a) * "_is_" * string(b))
contains(b::Symbol, a::Symbol) = occursin(string(a), string(b))

reverse_dict(space::Dict) = Dict(vl=>ky for (ky,vl) in space)
get_dict_desc(space::Dict, s::Int) = reverse_dict(space)[s]
ordered_dict_keys(space::Dict) = [reverse_dict(space)[ky] for ky in 1:space.count]

"""Print a state as a table."""
printt(s::NamedTuple) = PrettyTables.pretty_table(reshape(collect(s),(1,length(s))); header = collect(string.(fieldnames(s))))

"""Python Wrapper: Create observation objects as NamedTuples, from strings and values."""
py_construct_obs(names::Tuple, vals::Tuple) = NamedTuple{Symbol.(names)}(vals)

"""Python Wrapper: Display the most likely states in beliefs as a table."""
function py_tabulate_belief(State_Space::AbstractDict, beliefs::AbstractDict; prob_threshold=0.35, sort_by=-1, sort_rev=true)

    data_all = nothing
    L_table = nothing
    fieldnames_table = nothing

    for (rv_id, bel) in beliefs

        likely_states_idx = findall(x->x>=prob_threshold, bel.b)
        if isempty(likely_states_idx) continue end
        likely_states_probs = bel.b[likely_states_idx]
        likely_states = ordered_dict_keys(State_Space)[likely_states_idx]

        L = length(likely_states_probs)
        data = permutedims(vcat(repeat([rv_id], L)', hcat(collect.(likely_states)...), likely_states_probs'))
        if sort_by == -1 sort_by = size(data, 2) end
        data = sortslices(data, dims=1, by=x->x[sort_by], rev=sort_rev)

        if isnothing(data_all)
            L_table = L
            fieldnames_table = fieldnames(likely_states[1])
            data_all = data
        else
            L_table += L
            data_all = vcat(data_all, data)
        end
    end

    if isnothing(L_table)
        return
    else
        return pretty_table(data_all; header = vcat("Rival ID", collect(string.(fieldnames_table)), "Prob"), hlines=0:L_table+1)
    end
end