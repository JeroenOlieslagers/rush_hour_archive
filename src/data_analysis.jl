include("engine.jl")
include("solvers.jl")
using CSV
using DataFrames
using DataStructures
using Plots
using PDFmerger
using StatsPlots


function load_raw_data()
    csv_reader = CSV.File("data/raw_data/trialdata_headers.csv");
    data = pre_process(csv_reader)
    return data
end

function pre_process(csv)
    """
    Pre-processes csv data into dictionary where keys are subject identifiers and values are trial dataframes
    """
    # Data dictionary has subject ID as key and dataframe object as value, containing all the data
    data = Dict{String, DataFrame}()
    # Load problem files
    jsons = readdir("data/raw_data/problems")
    # Loop over all data
    for row in csv
        # Remove backslashes
        info = replace(row.info, "\""=>"")
        # Split row by space to get individual data points
        info_list = split(info, " ")[2:end]
        # Check if row is valid data point (if doesnt contain "t:", it's an instruction row)
        if info_list[1][1:2] == "t:"
            # Create new data frame if subject has not appeared yet
            if !(row.subject in keys(data))
                data[row.subject] = DataFrame(event=String[], move=Int[], instance=String[], t=Int[], piece=Int[], target=Int[])
            end
            # Split key and value for data points
            info_list_split = [split(l, ":") for l in info_list]
            # Create a dict to store all data
            info_dict = Dict{Symbol, Any}()
            # Loop over data to rename some of the keys
            for info_row in info_list_split
                # Get and rename key
                key = info_row[1]
                if key == "move"
                    key = "target"
                elseif key == "move#"
                    key = "move"
                end
                # Get value and parse depending on type
                value = info_row[2][2:end-1]
                if key in ["event", "instance"]
                    if key == "instance"
                        for js in jsons
                            pro = split(js, "_")
                            if pro[1] == value
                                value = split(js, ".")[1]
                                break
                            end
                        end
                    end
                    info_dict[Symbol(key)] = value
                elseif key in ["move", "t", "target", "piece"]
                    # Set -1 as "NA" value
                    if value == "NA"
                        info_dict[Symbol(key)] = -1
                    else
                        # Set 'r' piece to 9
                        if key == "piece" && value == "r"
                            value = "8"
                        end
                        # If time has a . in it (BONUS_FAIL type of events), or move
                        # is undefined, set to -1
                        if occursin(".", value) || occursin("undefined", value)
                            info_dict[Symbol(key)] = -1
                        else
                            info_dict[Symbol(key)] = parse(Int, value)
                        end
                    end
                end
            end
            # Append to data object
            push!(data[row.subject], info_dict)
        end
    end
    return data
end

function time_hists(data)
    """
    Plots histogram of total time subject spent interacting with the application
    as well as intervals between interactions
    """
    times = Array{Float64, 1}()
    intervals = Array{Float64, 1}()
    for subject in keys(data)
        df = data[subject]
        ts = df.t
        # Get total time of interaction
        time = round((maximum(ts)-minimum(ts))/(60*1000), digits=3)
        if time > 300
            time = 150
        elseif time == 0
            time = -75
        end
        # Get intervals between interactions
        interval = round.((ts[2:end] - ts[1:end-1])/(1000), digits=3)
        long_int = interval .> 30*60
        neg_int = interval .< -30*60
        if length(interval[neg_int]) > 0
            println(subject * ": " * string(maximum((interval))))
        end
        interval[long_int] = 30*60*ones(length(interval[long_int]))
        interval[neg_int] = -30*60*ones(length(interval[neg_int]))
        push!(times, time)
        push!(intervals, interval...)
    end
    #p = histogram(times, bins=50, xlabel="Interaction time (min)", ylabel="Histogram count")
    p = histogram(intervals, bins=150, xlabel="Interval time (s)", ylabel="Histogram count", yscale=:log10, legend="")
    #savefig(p,"duration_plot.png")
end

function time_plot(data)
    """
    Line plot of times of interactions
    """
    ts = []
    tts = []
    ttts = []
    tots = []
    for subject in keys(data)
        tt = deepcopy(data[subject].t)./(1000*60)
        negs = findall(x->x<0,tt)
        deleteat!(tt, negs)
        int = tt[2:end] - tt[1:end-1]
        neg_int = findall(x->x<0, int)
        tt .-= first(tt)
        if length(tt) > 1
            # if maximum(abs.(tt[2:end] - tt[1:end-1])) < 5
            #     push!(ts, tt)
            # elseif 5 <= maximum(abs.(tt[2:end] - tt[1:end-1])) < 30
            #     push!(tts, tt)
            # elseif maximum(abs.(tt[2:end] - tt[1:end-1])) >= 30
            #     push!(ttts, tt)
            # end
            if length(neg_int) > 0
                println(subject)
                push!(ttts, tt)
            else
                push!(tts, tt)
            end
            push!(tots, last(tt))
        end
    end
    p1 = plot()
    map(x->plot!(LinRange(0, 1, length(x)), x, legend=false, color=:black, alpha=0.3, xticks=[], ylims=(0, 100), yticks=[0, 30, 60, 90], grid=false, ylabel="Timestamp (min)", title="Does not have negative interval: "*string(length(ts))), ts)
    p2 = plot()
    map(x->plot!(LinRange(0, 1, length(x)), x, legend=false, color=:black, alpha=0.3, xticks=[], ylims=(0, 100), yticks=[0, 30, 60, 90], grid=false, ylabel="Timestamp (min)", title="Does not have negative interval: "*string(length(tts))), tts)
    p3 = plot()
    map(x->plot!(LinRange(0, 1, length(x)), x, legend=false, color=:black, alpha=0.3, grid=false, ylims=(0, 100), yticks=[0, 30, 60, 90], ylabel="Timestamp (min)", xlabel="Experiment progress (normalised)", title="Has negative interval: "*string(length(ttts))), ttts)
    p4 = histogram(tots, bins=50, legend=false, grid=false, xlabel="Final timestamp (min)", ylabel="Histogram count")
    p = plot(p2, p3, p4, layout=(3, 1), size=(500, 700), left_margin=5Plots.mm)
    #savefig(p, "timestamp_zero_cat.png")
end

function problem_plot(data)
    """
    Plots what problem is being attempted and when
    """
    p = plot()
    for subject in keys(data)
        # Get problem sets
        probs = data[subject].instance
        uniq = unique(probs)
        # Get timestamps
        tt = (data[subject].t)./(1000*60)
        tt .-= first(tt)
        # Get negative time points (BONUS_FAIL/BONUS_SUCCESS)
        negs = findall(x->x<0, tt)
        # Remove negative time points
        deleteat!(tt, negs)
        # Calculate intervals between interactions
        intervals = tt[2:end] - tt[1:end-1]
        # Exclude subjects based on total length and breaks
        if last(tt) - first(tt) < 30
            println(subject)
            continue
        elseif (last(tt) - first(tt) < 45) && maximum(intervals) > 5
            println(subject)
            continue
        elseif (last(tt) - first(tt) < 60) && maximum(intervals) > 10
            println(subject)
            continue
        elseif maximum(intervals) > 15
            println(subject)
            continue
        end
        # Excluse negative interval subjects
        if minimum(intervals) < 0
            println(subject)
            continue
        end
        # Exclude subject that complete less than 10 puzzles
        if length(uniq) < 10
            println(subject)
            continue
        end
        # Initialise
        x = Array{Float64, 1}()
        y = Array{Int, 1}()
        t = Array{Float64, 1}()
        for (n, uni) in enumerate(uniq)
            # Get indices that correspond to the current problem instance
            idx = findall(x->x==uni, probs)
            # Remove indices that are BONUS_FAIL/BONUS_SUCCESS
            filter!(x->x∉negs, idx)
            # Normalise and append
            push!(x, (idx ./ length(probs))...)
            push!(y, n*ones(length(idx))...)
            push!(t, tt[idx]...)
        end
        # Set default marker color to black and size to 1
        c = [:black for _ in 1:length(x)]
        size = [2 for _ in 1:length(x)]
        # Get events
        events = data[subject].event
        filter!(x->x∉["BONUS_FAIL", "BONUS_SUCCESS"], events)
        # Get wins
        wins = findall(x->x=="win", events)
        c[wins] = [:green for _ in 1:length(wins)]
        size[wins] = [4 for _ in 1:length(wins)]
        # Get surrenders
        surrenders = findall(x->x=="surrender", events) 
        c[surrenders] = [:red for _ in 1:length(surrenders)]
        size[surrenders] = [4 for _ in 1:length(surrenders)]
        # Get restarts
        restarts = findall(x->x=="restart", events) 
        c[restarts] = [:blue for _ in 1:length(restarts)]
        size[restarts] = [4 for _ in 1:length(restarts)]

        p1 = scatter(x, y, legend=false, grid=false, ylabel="Problem index", xlabel="Experiment progress (normalised)", color = c, markerstrokewidth=0, markersize = size, title=split(subject, ":")[1])
        p2 = scatter(t, y, legend=false, grid=false, ylabel="Problem index", xlabel="Timestamp (min)", color = c, markerstrokewidth=0, markersize = size)
        p = plot(p1, p2, layout=(2, 1), size=(700, 500), left_margin=5Plots.mm, label="Move")
        scatter!([0], [0], legend=true, color = :red, label="Surrender", markerstrokewidth=0, markersize = 0.1)
        scatter!([0], [0], legend=true, color = :green, label="Win", markerstrokewidth=0, markersize = 0.1)
        scatter!([0], [0], legend=true, color = :blue, label="Restart", markerstrokewidth=0, markersize = 0.1)
        #break
        #savefig(p, "temp.pdf")
        #append_pdf!("problem_index_filtered.pdf", "temp.pdf", cleanup=true)
    end
    #scatter(p)
end

function analyse_subject(subject_data)
    """
    Check if moves subject does are sensible
    """
    # Get problem sets
    probs = subject_data.instance
    uniq = unique(probs)
    # Get timestamps
    tt = (subject_data.t)./(1000*60)
    tt .-= first(tt)
    # Get negative time points (BONUS_FAIL/BONUS_SUCCESS)
    negs = findall(x->x<0, tt)
    # Get moves data
    moves = subject_data.move
    targets = subject_data.target
    pieces = subject_data.piece
    events = subject_data.event
    times = subject_data.t
    # Only include move end and restart data (tells us everything we need)
    excl_events = findall(x->x∉["drag_end", "restart"], events)
    tot_arrs = Dict{String, Array{Matrix{Int}, 1}}()
    tot_move_tuples = Dict{String, Array{Tuple{Int, Int}, 1}}()
    tot_states_visited = Dict{String, Array{BigInt, 1}}()
    tot_RTs = Dict{String, Array{Int, 1}}()
    # Record number of moves per puzzle per attempt
    #attempts = DefaultDict{String, Dict{Int, Int}}(Dict{Int, Int})
    attempts = Dict{String, Int}()
    for (n, uni) in enumerate(uniq)
        # Get indices that correspond to the current problem instance
        idx = findall(x->x==uni, probs)
        idx_restart = findall(x->x==uni, probs)
        iddx = findfirst(x->x=="start", events[idx])
        if iddx !== nothing
            start_time = times[idx[iddx]]
        end
        # Remove indices that are BONUS_FAIL/BONUS_SUCCESS and not drag_end
        filter!(x->x∉negs, idx)
        filter!(x->x∉excl_events, idx)
        # Get moves of current problem
        all_moves = moves[idx]
        # If empty, continue
        if length(all_moves) == 0
            continue
        end
        # Remove moves that do not move any cars
        excl_moves = Array{Int, 1}()
        for (index, move) in enumerate(all_moves)
            # If move number does not change, exclude it
            if index < length(all_moves) && move == all_moves[index+1]
                push!(excl_moves, idx[index])
            end
        end
        filter!(x->x∉excl_moves, idx)
        # Get final move data
        prob_moves = moves[idx]
        prob_targets = targets[idx]
        prob_pieces = pieces[idx]
        prob_events = events[idx]
        prob_times = times[idx]
        if idx[end]+1 <= length(events)
            #println(events[idx[end]+1])
            if events[idx[end]+1] != "win"
                continue
            end
        else
            #continue
            if events[idx[end]] != "win"
                continue
            end
        end
        # Add to attempts dictionary to keep track of total number of moves per attempt
        # INCLUDES RESTARTING AS A MOVE
        attempts[uni] = length(idx)
        # Find restarts
        restarts = findall(x->x=="restart", prob_events)
        # Add restart at beginning and end to create correct intervals
        pushfirst!(restarts, 0)
        push!(restarts, length(prob_moves))
        # Collect data across restarts
        arrs = Array{Matrix{Int}, 1}()
        move_tuples = Array{Tuple{Int, Int}, 1}()
        states_visited = Array{BigInt, 1}()
        RTs = Array{Int, 1}()
        init_time = start_time
        for i in 1:length(restarts)-1
            # Get indices for intervals between restarts
            idx = (restarts[i]+1):(restarts[i+1]-1)
            # # Add to attempts dictionary to keep track of total number of moves per attempt
            # attempts[uni][i] = length(idx)
            # Initialise board
            board = load_data(uni)
            if i > 1
                init_time = prob_times[idx[1]-1]
            end
            # Loop over moves
            for j in idx
                # If start or restart, reload problem
                move, target, piece, time = prob_moves[j], prob_targets[j], prob_pieces[j], prob_times[j]
                if move == 0
                    # Load problem
                    board = load_data(uni)
                    arr = get_board_arr(board)
                    push!(arrs, arr)
                    push!(move_tuples, (-1, 0))
                    push!(states_visited, board_to_int(arr, BigInt))
                    push!(RTs, 0)
                end
                # Get moving piece
                move_piece = piece + 1
                # Get move amount
                m = get_move_amount(move_piece, target, board)
                # Make move
                make_move!(board, (move_piece, m))
                arr = get_board_arr(board)
                push!(arrs, arr)
                push!(move_tuples, (move_piece, m))
                push!(states_visited, board_to_int(arr, BigInt))
                push!(RTs, time - init_time)
            end
        end
        tot_arrs[uni] = arrs
        tot_move_tuples[uni] = move_tuples
        tot_states_visited[uni] = states_visited
        tot_RTs[uni] = RTs
        #p = plot(prob_moves)
        #display(p)
    end
    return tot_arrs, tot_move_tuples, tot_states_visited, attempts, tot_RTs
end

function filter_subjects(data)
    """
    Filters weird subjects
    """
    # New data object
    filtered_data = Dict{String, DataFrame}()
    # Count how many subjects get rejected
    counter = 0
    for subject in keys(data)
        # Get problem sets
        probs = data[subject].instance
        uniq = unique(probs)
        # Get timestamps
        tt = (data[subject].t)./(1000*60)
        tt .-= first(tt)
        # Get negative time points (BONUS_FAIL/BONUS_SUCCESS)
        negs = findall(x -> x < 0, tt)
        # Remove negative time points
        deleteat!(tt, negs)
        # Calculate intervals between interactions
        intervals = tt[2:end] - tt[1:end-1]
        # Exclude subjects based on total length and breaks
        if last(tt) - first(tt) < 30
            counter += 1
            continue
        # elseif (last(tt) - first(tt) < 45) && maximum(intervals) > 5
        #     counter += 1
        #     continue
        elseif (last(tt) - first(tt) < 60) && maximum(intervals) > 10
            counter += 1
            continue
        elseif maximum(intervals) > 15
            counter += 1
            continue
        end
        # Excluse negative interval subjects
        if minimum(intervals) < 0
            counter += 1
            continue
        end
        # Exclude subject that complete less than 10 puzzles
        # if length(uniq) < 10
        #     counter += 1
        #     continue
        # end
        # Add subject that meets criteria
        filtered_data[subject] = deepcopy(data[subject])
    end
    println("Rejection ratio: " * string(counter) * "/" * string(length(data)))
    return filtered_data
end
dd = filter_subjects(d);

"""
    boxplot_figure(data)

Plots the number of moves for each puzzle across all subjects,
where moves are considered if they result in a win (moves before
a restart or a surrender are not considered)
"""
function boxplot_figure(data)
    # Concatenate all subject's data
    y = DefaultDict{String, Array{Int}}([])
    for subj in keys(data)
        arrs, move_tuples, _, attempts = analyse_subject(data[subj]);
        for prob in keys(attempts)
            push!(y[prob], attempts[prob])
            # for restart in values(attempts[prob])
            #     push!(y[prob], restart)
            # end
        end
    end
    xx = [[k] for k in keys(y)]
    yy = collect(values(y))
    # Get optimal length
    n = DefaultDict{Int, Dict{String, Array{Int}}}(Dict{String, Array{Int}})
    for i in 1:length(xx)
        prob_str = xx[i][1]
        opt = parse(Int, split(prob_str, "_")[2])-1
        #push!(n, [opt])
        n[opt][prob_str] = yy[i]
    end

    # Box plot everything sorted by optimal move length
    ps = []
    for opt in sort(collect(keys(n)))
        xx = [[k] for k in keys(n[opt])]
        yy = collect(values(n[opt]))
        p = dotplot(xx, yy, label=nothing, marker=(:black, Plots.stroke(0)), ylim=(0, 200), xticks=[], grid=false, markersize=1.5, foreground_color_legend=nothing)
        if opt == 6
            scatter!([], [], color=:black, label="One subject", markersize=1)
            plot!([l[1] for l in xx], [opt for _ in xx], color=:red, label="Optimal solution length")
        elseif opt == 15
            plot!([l[1] for l in xx], [opt for _ in xx], color=:red, label=nothing, xlabel="Problem index")
        else
            plot!([l[1] for l in xx], [opt for _ in xx], color=:red, label=nothing)
        end
        push!(ps, p)
    end

    p = plot(ps..., layout=(4, 1), ylabel="Moves", legend=:topright, size=(400, 400), dpi=200)
    #savefig(p,"boxplot_high_ylim.png")
end

function random_agent_boxplot(data)
    y = DefaultDict{String, Array{Int}}([])
    for subj in keys(data)
        arrs, move_tuples, attempts = analyse_subject(data[subj]);
        for prob in keys(attempts)
            board = load_data(prob)
            push!(y[prob], random_agent_no_undo(board))
        end
    end
    xx = [[k] for k in keys(y)]
    yy = collect(values(y))
    # Get optimal length
    n = DefaultDict{Int, Dict{String, Array{Int}}}(Dict{String, Array{Int}})
    for i in 1:length(xx)
        prob_str = xx[i][1]
        opt = parse(Int, split(prob_str, "_")[2])-1
        #push!(n, [opt])
        n[opt][prob_str] = yy[i]
    end
    # Box plot everything sorted by optimal move length
    ps = []
    for opt in sort(collect(keys(n)))
        xx = [[k] for k in keys(n[opt])]
        yy = collect(values(n[opt]))
        p = boxplot(xx, yy, label=nothing, marker=(:black, Plots.stroke(0)), ylim=(0, 5000), xticks=[], grid=false)
        if opt == 15
            plot!([l[1] for l in xx], [opt for _ in xx], color=:red, label="Optimal move", xlabel="Problem")
        else
            plot!([l[1] for l in xx], [opt for _ in xx], color=:red, label=nothing)
        end
        push!(ps, p)
    end

    p = plot(ps..., layout=(4, 1), ylabel="Moves", legend=:topright)
    #savefig(p,"random_agent_no_undo_high_ylim.png")
end

"""
    A, opt = plot_optimal_action_fraction(IDV, optimal_a)

Returns the size of action space as well as size of optimal action space
in each state.
"""
function plot_optimal_action_fraction(IDV, optimal_a; visited_states=nothing)
    A = Int[]
    opt_L = Int[]
    opt = Int[]
    opt_dict = Dict{String, Array{Array{Int, 1}, 1}}()
    if visited_states === nothing
        opt_av = [[] for _ in 1:25]
        opt_L_av = [[] for _ in 1:25]
        joint = zeros(25, 34)
        cnt = zeros(25, 34)
        visits = zeros(25, 34)
        opt_dict["1"] = [[] for _ in 1:25]
        for prb in keys(IDV)
            prb_ss = IDV[prb]
            opt_a = optimal_a[prb]
            for state in keys(prb_ss)
                # Don't calculate on solved positions
                if prb_ss[state][1] == 0
                    continue
                end
                joint[prb_ss[state][2], prb_ss[state][1]] += length(opt_a[state])
                cnt[prb_ss[state][2], prb_ss[state][1]] += prb_ss[state][2]
                visits[prb_ss[state][2], prb_ss[state][1]] += 1
                push!(A, prb_ss[state][2])
                push!(opt_L, prb_ss[state][1])
                push!(opt_L_av[prb_ss[state][2]], prb_ss[state][1])
                push!(opt, length(opt_a[state]))
                push!(opt_av[prb_ss[state][2]], length(opt_a[state]))
                push!(opt_dict["1"][prb_ss[state][2]], length(opt_a[state]))
            end
        end
    else
        opt_av = [[] for _ in 1:24]
        opt_L_av = [[] for _ in 1:24]
        joint = zeros(24, 33)
        cnt = zeros(24, 33)
        visits = zeros(24, 33)
        # Only calculate visited states if provided
        for subj in keys(visited_states)
            opt_dict[subj] = [[] for _ in 1:24]
            for prb in keys(visited_states[subj])
                for r in eachindex(visited_states[subj][prb])
                    states = visited_states[subj][prb][r]
                    # Loop over states (ignore last state as it has no action)
                    for i in 1:length(states)-1
                        s = states[i]
                        # Don't calculate on solved positions
                        if IDV[prb][s][1] == 0
                            continue
                        end
                        joint[IDV[prb][s][2], IDV[prb][s][1]] += length(optimal_a[prb][s])
                        cnt[IDV[prb][s][2], IDV[prb][s][1]] += IDV[prb][s][2]
                        visits[IDV[prb][s][2], IDV[prb][s][1]] += 1
                        push!(A, IDV[prb][s][2])
                        push!(opt_L, IDV[prb][s][1])
                        push!(opt_L_av[IDV[prb][s][2]], IDV[prb][s][1])
                        push!(opt, length(optimal_a[prb][s]))
                        push!(opt_av[IDV[prb][s][2]], length(optimal_a[prb][s]))
                        push!(opt_dict[subj][IDV[prb][s][2]], length(optimal_a[prb][s]))
                    end
                end
            end
        end
    end
    joint ./= cnt
    joint .*= 0.999
    joint += visits
    replace!(joint, NaN=>0)
    return A, opt_L, opt, [mean(item) for item in opt_L_av], [mean(item) for item in opt_av], opt_dict, joint
end

function trajectory(states, IDV)
    x = []
    y = []
    for prb in keys(states)
        for r in eachindex(states[prb])
            push!(x, [])
            push!(y, [])
            for state in states[prb][r]
                push!(x[end], IDV[prb][state][2])
                push!(y[end], IDV[prb][state][1])
            end
        end
    end
    return x, y
end

function trajectory_rand(states, IDV, prb)
    x = []
    y = []
    for s in states
        push!(x, IDV[prb][s][2])
        push!(y, IDV[prb][s][1])
    end
    return x, y
end

function plot_puzzle_timeline(puzzle, visited_states, IDV; sp=nothing)
    for subj in keys(visited_states)
        for prb in keys(visited_states[subj])
            if prb == puzzle
                for r in eachindex(visited_states[subj][prb])
                    x = 1:length(visited_states[subj][prb][r])
                    y = []
                    for state in visited_states[subj][prb][r]
                        push!(y, IDV[prb][state][1])
                    end
                    if sp === nothing
                        plot!(x, y, label=nothing, c=:black, alpha=0.2)
                    else
                        if sp % 2 == 1
                            if sp == 1
                                plot!(x, y, sp=sp, label=nothing, c=:black, alpha=0.2, ylim=(0, 16), xlim=(0, 140), yticks=[0, 5, 10, 15], xticks=[0, 50, 100], xlabel="Move number")
                            else
                                plot!(x, y, sp=sp, label=nothing, c=:black, alpha=0.2, ylim=(0, 16), xlim=(0, 140), yticks=[0, 5, 10, 15], xticks=([0, 50, 100], ["", "", ""]))
                            end
                        else
                            if sp == 2
                                plot!(x, y, sp=sp, xticks=[0, 50, 100], xlabel="Move number", label=nothing, c=:black, alpha=0.2, ylim=(0, 16), xlim=(0, 140), yticks=[0, 5, 10, 15])
                            else
                                plot!(x, y, sp=sp, label=nothing, c=:black, alpha=0.2, ylim=(0, 16), xlim=(0, 140), yticks=[0, 5, 10, 15], xticks=([0, 50, 100], ["", "", ""]))
                            end
                        end
                    end
                end
            end
        end
    end
end

# states = BigInt[]
# push!(states, visited_states[subj][prb][1][1])
# for i in 1:10000
#     push!(states, sample(graphs_prb[prb][states[end]]))
#     if states[end] in solutions_prb[prb]
#         break
#     end
# end


# x, y = trajectory(visited_states[subj], IDV);
# x_rand, y_rand = trajectory_rand(states, IDV, prb);

# plot!(x_rand, y_rand, c=:red, alpha=0.5, label=nothing)
# plot!(x[1], y[1], c=:blue, alpha=0.5,label=nothing)

# for i in eachindex(x)
#     plot!(x[i] .+ 0.5, y[i] .+ 0.5, c=:blue, alpha=0.2, label=nothing)
# end
# plot!()


# graphs_prb = load("data/processed_data/graphs_prb.jld2")["data"];
# solutions_prb = load("data/processed_data/solutions_prb.jld2")["data"];


optimal_a = load("data/processed_data/optimal_a.jld2");
IDV = load("data/processed_data/IDV.jld2");
# A, opt_L, opt, opt_L_av, opt_av, opt_dict, joint = plot_optimal_action_fraction(IDV, optimal_a; visited_states=visited_states);


# lim_A = maximum(A)+1
# lim_opt = maximum(opt)+1
# lim_opt_L = maximum(opt_L)+1
# lim = maximum([lim_A, lim_opt])
# histogram2d(A, opt, bins=(lim_A, lim_opt), xlim=(1, lim), ylim=(1, lim), grid=false, color=cgrad(:grays, rev=true), ylabel=latexstring("|A_\\texttt{opt}|"), xlabel=latexstring("|A|"), colorbar_title="\nCounts", left_margin = 3Plots.mm, right_margin = 7Plots.mm, size=(500, 400))

# plot!([1, lim], [1, lim], c=:red, label="Diagonal")

# plot!((1:lim_A-1) .+ 0.5, opt_av, label="Average "*latexstring("|A_\\texttt{opt}|"), c=:blue)

# histogram2d(A, opt_L, bins=(lim_A, lim_opt_L), xlim=(1, lim_A), ylim=(1, lim_opt_L), grid=false, color=cgrad(:grays, rev=true), ylabel=latexstring("L_\\texttt{opt}"), xlabel=latexstring("|A|"), colorbar_title="\nCounts", left_margin = 3Plots.mm, right_margin = 7Plots.mm, size=(400, 400))
# plot!((1:lim_A-1) .+ 0.5, opt_L_av, label="Average "*latexstring("L_\\texttt{opt}"), c=:blue)

# prb_states = Dict{String, Array}()
# for prbb in prbbs
#     prb_states[prbb] = [visited_states[subjs[9]][prbb][end]]
# end

# xx = []
# yy = []
# for prb in prbs
#     for subj in subjs
#         if prb in keys(visited_states[subj])
#             push!(xx, IDV[prb][visited_states[subj][prb][1][1]][2])
#             push!(yy, IDV[prb][visited_states[subj][prb][1][1]][1])
#             break
#         end
#     end
# end

# x, y = trajectory(prb_states, IDV)

# marginal_histogram(A, opt_L, bins=(lim_A, lim_opt_L), xlim=(1, lim_A), ylim=(1, lim_opt_L), ylabel=latexstring("L_\\texttt{opt}"), xlabel=latexstring("|A|"), size=(400, 400))
# scatter!(xx, yy, sp=2, c=:gray, markersize=1, label=nothing)
# plot!(x[1], y[1], label="Puzzle 1", sp=2, c=palette(:default)[1],foreground_color_legend = nothing)
# scatter!([x[1][1]], [y[1][1]], sp=2, c=palette(:default)[1], label=nothing)
# plot!(x[4], y[4], label="Puzzle 2", sp=2, c=palette(:default)[2])
# scatter!([x[4][1]], [y[4][1]], sp=2, c=palette(:default)[2], label=nothing)
# plot!(x[2], y[2], label="Puzzle 3", sp=2, c=palette(:default)[3])
# scatter!([x[2][1]], [y[2][1]], sp=2, c=palette(:default)[3], label=nothing)
# plot!(x[3], y[3], label="Puzzle 4", sp=2, c=palette(:default)[4])
# scatter!([x[3][1]], [y[3][1]], sp=2, c=palette(:default)[4], label=nothing)
# scatter!([],[],sp=2,c=:gray,label="Initial states",mc=:white, msc=:black)


# heatmap((joint .% 1.0)', color=cgrad(:grays, rev=true), xlim=(0, lim_A), ylim=(0, lim_opt_L), ylabel=latexstring("L_\\texttt{opt}"), xlabel=latexstring("|A|"), size=(400, 400))

function twod_colormap(X)
    flat = vcat(X...)
    flat_sorted_unique = sort(unique(flat))
    all_visits_ = flat .÷ 1.0
    all_visits = all_visits_[all_visits_ .> 0]
    all_proportion_ = (flat .% 1.0) ./ 0.999
    all_proportion = all_proportion_[all_visits_ .> 0]
    unique_visits = sort(unique(flat .÷ 1.0))
    unique_proportion = sort(unique((flat .% 1.0) ./ 0.999))
    cs = LCHab{Float64}[]
    for x in flat_sorted_unique
        visits = x ÷ 1.0
        proportion = (x % 1.0) / 0.999
        visits_idx = findfirst(item->item==visits, unique_visits)
        proportion_idx = findfirst(item->item==proportion, unique_proportion)
        if isnan(x) || x == 0.0
            push!(cs, LCHab{Float64}(100.0, 0.0, 0.0))
        else
            # transformed_visits = visits_idx / length(unique_visits_joint)
            transformed_visits = mean(all_visits .< visits)
            # transformed_proportion = 120*(proportion_idx / length(unique_proportion)) - 80
            transformed_proportion = 120*mean(all_proportion .< proportion) - 80
            push!(cs, LCHab{Float64}(98 - transformed_visits*47,transformed_visits*72, transformed_proportion))
        end
    end
    cbar = zeros(length(unique_visits), length(unique_proportion))
    cbar_cs = Luv{Float64}[]
    for i in eachindex(unique_visits)
        for j in eachindex(unique_proportion)
            # transformed_visits = i / length(unique_visits_joint)
            # transformed_proportion = 120*(j / length(unique_proportion)) - 80
            transformed_visits = mean(all_visits .< unique_visits[i])
            transformed_proportion = 120*mean(all_proportion .< unique_proportion[j]) - 80
            push!(cbar_cs, LCHab{Float64}(98 - transformed_visits*47,transformed_visits*72, transformed_proportion))
            cbar[i, j] = length(unique_proportion)*(i-1) + (j-1)
        end
    end
    return cs, cbar_cs, cbar, unique_visits, unique_proportion
end

# cs, cbar_cs, cbar, unique_visits, unique_proportion = twod_colormap(joint);

# # xt = round.(Int, collect(range(1, length(unique_proportion), length=5)));
# # yt = round.(Int, collect(range(1, length(unique_visits), length=5)));
# xs = [0.0, 0.15, 0.2, 0.25, 0.4, 1.0];
# xt = [findfirst(x->x==item, round.(unique_proportion, digits=4)) for item in xs];
# ys = [0, 10, 20, 50, 100, 200, 500];
# yt = [findfirst(x->x==item, floor.(unique_visits, sigdigits=1)) for item in ys];


# plot(layout=(1, 2), size=(600, 300))
# heatmap!(joint', sp=1, color=cs, colorbar=false, xlabel=latexstring("|A|"), ylabel="opt_L", framestyle = :box)

# heatmap!(cbar, sp=2, color=cbar_cs, colorbar=false, xlabel=latexstring("f_\\texttt{opt}"), ylabel="Visits", xticks=(xt, string.(round.(unique_proportion[xt], digits=2))), yticks=(yt, string.(Int.(floor.(unique_visits[yt], sigdigits=1)))), framestyle = :box)
# plot!([10], [10], sp=2, label=nothing, tick_direction=:out, xlim=(0, length(unique_proportion)), ylim=(0, length(unique_visits)))


# prbbs = ["prb21272_7","prb20059_7","prb45893_16","prb24227_16"];
# prbbbs = [prbs[46], prbs[44]];
# plot(layout=(1, 2), size=(900, 360), bottom_margin = 8Plots.mm, left_margin = 8Plots.mm, legendfont=font(18), 
# xtickfont=font(16), 
# ytickfont=font(16), 
# guidefont=font(32), grid=false, ylabel=latexstring("d_\\texttt{goal}"))
# for i in eachindex(prbbbs)
#     plot_puzzle_timeline(prbbbs[i], visited_states, IDV, sp=i)
# end
# plot!()
# flat_joint = vcat(joint...);
# flat_joint_sorted_unique = sort(unique(flat_joint));
# cs = twod_colormap.(flat_joint_sorted_unique);
# new_joint = zeros(24, 33);
# for i in 1:24
#     for j in 1:33
#         new_joint[i, j] = findfirst(x->x==joint[i, j], flat_joint_sorted_unique)
#     end
# end
# plot(layout=(1, 2), size=(600, 300))
# heatmap!(new_joint', sp=1, color=cs, colorbar=false, xlabel=latexstring("|A|"), ylabel="opt_L")

# NN = 100
# limm = 1#floor(Int, -log(exp(log(700)/NN)-1)/(log(700)/NN));
# show_c = zeros(NN, NN-limm+1);
# x = zeros(NN);
# y = zeros(NN-limm+1);
# for i in 1:NN
#     for j in limm:NN
#         #show_c[i, j-limm+1] = floor(Int, exp(j*log(700)/NN)) + ((i-1)/(NN-1))*0.999
#         show_c[i, j-limm+1] = floor(Int, 700/NN)*j + ((i-1)/(NN-1))*0.999
#         x[i] = (i-1)/(NN-1)
#         #y[j-limm+1] = floor(Int, exp(j*log(700)/NN))
#         y[j-limm+1] = floor(Int, 700/NN)*j
#     end
# end
# flat_show_c = vcat(show_c...);
# flat_show_c_sorted_unique = sort(unique(flat_show_c));
# cs = twod_colormap.(flat_show_c_sorted_unique);
# new_joint_show_c = zeros(NN, NN-limm+1);
# for i in 1:NN
#     for j in 1:NN-limm+1
#         new_joint_show_c[i, j] = findfirst(x->x==show_c[i, j], flat_show_c_sorted_unique)
#     end
# end
# heatmap!(x, y, new_joint_show_c', sp=2, color=cs, colorbar=false, xlabel=latexstring("f_\\texttt{opt}"), ylabel="Visits")

#scatter(A, opt_L, markershape=:rect, color=:black, markersize=5, xlim=(0, lim_A), ylim=(0, lim_opt_L), ylabel=latexstring("L_\\texttt{opt}"), xlabel=latexstring("|A|"), size=(350, 500))

# histogram2d(A, opt ./ A, grid=false, color=cgrad(:grays, rev=true), ylabel=latexstring("\\frac{|A_\\texttt{opt}|}{|A|}"), xlabel=latexstring("|A|"), colorbar_title="\nCounts", left_margin = 3Plots.mm, right_margin = 7Plots.mm, size=(500, 400))
# plot!(1:lim-1, 1 ./ collect(1:lim-1), c=:red, label=latexstring("1/|A|"))
# plot!(2:lim-1, 2 ./ collect(2:lim-1), c=:red, label=latexstring("2/|A|"))
# plot!(3:lim-1, 3 ./ collect(3:lim-1), c=:red, label=latexstring("3/|A|"))
# plot!(4:lim-1, 4 ./ collect(4:lim-1), c=:red, label=latexstring("4/|A|"))
# plot!(1:lim-1, opt_av ./ collect(1:lim-1), label=latexstring("\\mathbb{E}[|A_\\texttt{opt}|/|A|]"), c=:blue)

# av_subj, av_bins = quantile_binning(opt_dict; bins=6, bounds=true, lim=lim-1);
# a_dict = quantile_binning(opt_dict; bins=6, lim=lim-1);

# plot(ylim=(0.5, 1), grid=false, xlabel=latexstring("\\mathbb{E}[|A_\\texttt{bin}|]"), xticks=round.(av_bins, digits=1), legend=:bottomright)
# plot!(av_bins, 1 .- (av_subj ./ av_bins), label=latexstring("1 - \\mathbb{E}[|A_\\texttt{opt}|/|A_\\texttt{bin}|]"))
# scatter!(av_bins, 1 .- (1 ./ av_bins), label=latexstring("1 - \\mathbb{E}[1/|A_\\texttt{bin}|]"))
# for i in 1:42
#     if i == 42
#         plot!(av_bins_a[i, :], 1 .- (av_subj_a[i, :] ./ av_bins_a[i, :]), label=latexstring("1 - |A_\\texttt{opt}|_i/|A_\\texttt{bin}|_i"), c=:gray, alpha=0.1)
#         scatter!(av_bins_a[i, :], 1 .- (1 ./ av_bins_a[i, :]), label=latexstring("1 - 1/|A_\\texttt{bin}|_i"), c=:gray, alpha=0.1)
#     else
#         plot!(av_bins_a[i, :], 1 .- (av_subj_a[i, :] ./ av_bins_a[i, :]), label=nothing, c=:gray, alpha=0.1)
#         scatter!(av_bins_a[i, :], 1 .- (1 ./ av_bins_a[i, :]), label=nothing, c=:gray, alpha=0.1)
#     end
# end


# plot(ylim=(0.5, 1), grid=false, xlabel=latexstring("\\mathbb{E}[|A_\\texttt{bin}|]")* " visited states")
# plot!(av_bins, 1 .- (av_subj ./ av_bins), xlim=(5, 18), label=nothing, xticks=round.(av_bins, digits=1), xtickfontcolor=:red, color=:red, legend=:bottomright)
# scatter!(av_bins, 1 .- (1 ./ av_bins), xlim=(5, 18), label=nothing, color=:red)

# p = twiny()
# plot!(p, av_bins_all, 1 .- (av_subj_all ./ av_bins_all), xlim=(5, 18), ylim=(0.5, 1), label=nothing, xticks=round.(av_bins_all, digits=1), xtickfontcolor=:blue, color=:blue, legend=:topleft, xlabel=latexstring("\\mathbb{E}[|A_\\texttt{bin}|]")* " all states")
# scatter!(p, av_bins_all, 1 .- (1 ./ av_bins_all), xlim=(5, 18), label=nothing, color=:blue)


# raw_data = load_raw_data();
# data = filter_subjects(raw_data);

# time_plot(data)
# problem_plot(data)

#boxplot_figure(data)

#subj = "A3CTXNQ2GXIQSP:34HJIJKLP64Z5YMIU6IKNXSH7PDV4I"
