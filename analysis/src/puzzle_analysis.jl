include("data_analysis.jl")
include("solvers.jl")
using FileIO

data = load("analysis/processed_data/filtered_data.jld2")["data"]

puzzle_level_data = DataFrame(
    prb=String[], 
    Ls=Array{Int, 1}[], 
    optimal_len=Int[], 
    av_branch=Float64[], 
    nr_sol=Int[], 
    nr_opt_path=Int[], 
    state_space_size=Int[], 
    av_state_space_width=Float64[], 
    min_state_space_width=Int[], 
    max_depth=Int[], 
    av_rand=Float64[], 
    var_rand=Float64[])


"""
    get_Ls(data)

Return dictionary where keys are problem ids and values are arrays of lengths of people's solution.
"""
function get_Ls(data::Dict{String, DataFrame})::DefaultDict{String, Array{Int64}, Vector{Any}}
    dict = DefaultDict{String, Array{Int}}([])
    # Loop over all subjects
    for subj in keys(data)
        # Get L for each problem subject attempted
        _, _, attempts = analyse_subject(data[subj]);
        for prb in keys(attempts)
            # Push L to array for each problem
            push!(dict[prb], attempts[prb])
        end
    end
    return dict
end


"""
    get_optimal_lens(prbs)

Return dictionary where keys are problem ids and values are optimal lengths of solving that puzzle
"""
function get_optimal_lens(prbs::Vector{String})::Dict{String, Int}
    dict = Dict{String, Int}()
    #exps = 0
    for prb in prbs
        board = load_data(prb)
        _, past_moves, exp = a_star(board, h=red_distance)
        dict[prb] = length(past_moves)
        #exps += exp
    end
    #println(exps)
    return dict
end

function get_av_branchs(state_space)

end


    
Ls = get_Ls(data)
prbs = collect(keys(Ls))[sortperm([parse(Int, x[end-1] == '_' ? x[end] : x[end-1:end]) for x in keys(Ls)])]

optimal_lens = get_optimal_lens(prbs)

