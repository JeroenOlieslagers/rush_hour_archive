include("rushhour.jl")
include("solvers.jl")
using CSV
using DataFrames
using DataStructures
using Plots
using PDFmerger


function load_raw_data()
    csv_reader = CSV.File("experiment/raw_data/trialdata_headers.csv");
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
    jsons = readdir("experiment/raw_data/problems")
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
    # Only include move end and restart data (tells us everything we need)
    excl_events = findall(x->x∉["drag_end", "restart"], events)
    tot_arrs = Array{Array{Matrix{Int}, 1}, 1}()
    tot_move_tuples = Array{Array{Tuple{Int, Int}, 1}, 1}()
    # Record number of moves per puzzle per attempt
    #attempts = DefaultDict{String, Dict{Int, Int}}(Dict{Int, Int})
    attempts = Dict{String, Int}()
    for (n, uni) in enumerate(uniq)
        # Get indices that correspond to the current problem instance
        idx = findall(x->x==uni, probs)
        idx_restart = findall(x->x==uni, probs)
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
        attempts[uni] = length(idx)
        # Find restarts
        restarts = findall(x->x=="restart", prob_events)
        # Add restart at beginning and end to create correct intervals
        pushfirst!(restarts, 0)
        push!(restarts, length(prob_moves))
        for i in 1:length(restarts)-1
            # Get indices for intervals between restarts
            idx = (restarts[i]+1):(restarts[i+1]-1)
            # # Add to attempts dictionary to keep track of total number of moves per attempt
            # attempts[uni][i] = length(idx)
            # Initialise board
            board = load_data(uni)
            arrs = Array{Matrix{Int}, 1}()
            move_tuples = Array{Tuple{Int, Int}, 1}()
            # Loop over moves
            for (move, target, piece) in zip(prob_moves[idx], prob_targets[idx], prob_pieces[idx])
                # If start or restart, reload problem
                if move == 0
                    # Load problem
                    board = load_data(uni)
                    arr = get_board_arr(board)
                    push!(arrs, arr)
                end
                # Get moving piece
                move_piece = piece + 1
                # Get move amount
                m = get_move_amount(move_piece, target, board)
                # Make move
                make_move(board, (move_piece, m))
                arr = get_board_arr(board)
                push!(arrs, arr)
                push!(move_tuples, (move_piece, m))
            end
            push!(tot_arrs, arrs)
            push!(tot_move_tuples, move_tuples)
        end
        #p = plot(prob_moves)
        #display(p)
    end
    return tot_arrs, tot_move_tuples, attempts
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
        negs = findall(x->x<0, tt)
        # Remove negative time points
        deleteat!(tt, negs)
        # Calculate intervals between interactions
        intervals = tt[2:end] - tt[1:end-1]
        # Exclude subjects based on total length and breaks
        if last(tt) - first(tt) < 30
            counter += 1
            continue
        elseif (last(tt) - first(tt) < 45) && maximum(intervals) > 5
            counter += 1
            continue
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
        if length(uniq) < 10
            counter += 1
            continue
        end
        # Add subject that meets criteria
        filtered_data[subject] = deepcopy(data[subject])
    end
    println("Rejection ratio: " * string(counter) * "/" * string(length(data)))
    return filtered_data
end

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
        arrs, move_tuples, attempts = analyse_subject(data[subj]);
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
        p = boxplot(xx, yy, label=nothing, marker=(:black, Plots.stroke(0)), ylim=(0, 150), xticks=[], grid=false)
        if opt == 15
            plot!([l[1] for l in xx], [opt for _ in xx], color=:red, label="Optimal move", xlabel="Problem")
        else
            plot!([l[1] for l in xx], [opt for _ in xx], color=:red, label=nothing)
        end
        push!(ps, p)
    end

    p = plot(ps..., layout=(4, 1), ylabel="Moves", legend=:topright)
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

# total_data = load_raw_data();
# data = filter_subjects(total_data);

# time_plot(data)
# problem_plot(data)

# random_agent_boxplot(data)

#subj = "A3CTXNQ2GXIQSP:34HJIJKLP64Z5YMIU6IKNXSH7PDV4I"
