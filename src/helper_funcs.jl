using PrettyTables

concat_syms(a::Symbol, b::Symbol) = Symbol(string(a) * "_is_" * string(b))
contains(b::Symbol, a::Symbol) = occursin(string(a), string(b))

reverse_dict(space::Dict) = Dict(vl=>ky for (ky,vl) in space)
get_dict_desc(space::Dict, s::Int) = reverse_dict(space)[s]
ordered_dict_keys(space::Dict) = [reverse_dict(space)[ky] for ky in 1:space.count]

printt(s::NamedTuple) = PrettyTables.pretty_table(reshape(collect(s),(1,length(s))); header = collect(string.(fieldnames(s))))
