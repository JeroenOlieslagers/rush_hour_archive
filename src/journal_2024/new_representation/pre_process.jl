
# LOAD RAW DATA
data = load_messy_raw_data()
df = load_raw_data();
ndf = filter_subjects(df);
nndf = pre_process(ndf);

a = ndf[ndf.subject .== subj .&& ndf.instance .== "prb12715_11" .&& ndf.event .∈ Ref(["start", "drag_end", "win"]), :]
b = ndf[ndf.instance .== "prb42959_11", :]
c = a[a.instance .== unique(a.instance)[2], :]

nndf[nndf.subject .== subj .&& nndf.puzzle .== "prb12715_11", :]
for subj in unique(nndf.subject)
    a = nndf[nndf.subject .== subj, :]
    if length(states[subj]) != sum(a.event .== "move")
        println(subj)
    end
end

subj = "A1N1EF0MIRSEZZ:3R5F3LQFV3SKIB1AENMWM1BICT5OZB"
prb = "prb14898_11"
sum(problems[subj] .== prb)
c = a[a.puzzle .== prb, :]
ndf[ndf.subject .== subj .&& ndf.instance .== prb, :]
aa = []
for subj in unique(ndf.subject)
    ss = ndf[ndf.subject .== subj, :]
    for prb in unique(ss.instance)
        a = ss[ss.instance .== prb, :][end, :event]
        push!(aa, a)
    end
end
a = nndf[nndf.event .== "move", :]
a = ndf[ndf.subject .== subj, :]
c = a[a.instance .== prb, :]


function load_raw_data()
    df = CSV.read("data/raw_data/all_subjects.csv", DataFrame)
    return df
end

"""
    filter_subjects(df)

Remove subjects according to excludion criteria
"""
function filter_subjects(df)
    # New data object
    #filtered_data = Dict{String, DataFrame}()
    filtered_data = DataFrame()
    # Count how many subjects get rejected
    counter = 0
    for subject in keys(data)
        # Get problem sets
        subj_data = df[df.subject .== subject, :]#data[subject]
        probs = subj_data.instance
        uniq = unique(probs)
        # Get timestamps
        tt = (subj_data.t)./(1000*60)
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
        #filtered_data[subject] = deepcopy(data[subject])
        filtered_data = vcat(filtered_data, subj_data)
    end
    println("Rejection ratio: " * string(counter) * "/" * string(length(data)))
    return filtered_data
end

"""
    pre_process(df)

Take in DataFrame of subject data, filter it and return DataFrame where colums are 
subject ID, puzzle ID, move, state, time stamp, attempt
"""
function pre_process(df)
    new_df = DataFrame(subject=String[], puzzle=String[], event=String[], move=a_type[], state=s_type[], attempt=Int[], RT=Int[], timestamp=Int[])
    for subj in unique(df.subject)
        subj_df = df[df.subject .== subj .&& df.event .∈ Ref(["start", "restart", "drag_end", "win"]), :]
        for prb in unique(subj_df.instance)
            prb_df = subj_df[subj_df.instance .== prb, :]
            s_free, s_fixed = load_data(prb)
            attempt = 0
            prev_t = first(prb_df).t
            # Don't consider puzzles that are not solved
            # if "win" ∉ prb_df.event
            #     continue
            # end
            if prb_df[end, :event] != "win"
                continue
            end
            for (n, row) in enumerate(eachrow(prb_df))
                # Start of trial / after restart
                if row.event == "start"
                    attempt += 1
                    s_free, _ = load_data(prb)
                    if attempt == 1
                        push!(new_df, [subj, prb, "start", (0, 0), (copy(s_free), s_fixed), attempt, row.t - prev_t, row.t])
                    else
                        push!(new_df, [subj, prb, "restart", (0, 0), (copy(s_free), s_fixed), attempt, row.t - prev_t, row.t])
                    end
                    prev_t = row.t
                    continue
                elseif row.event == "win" || row.event == "restart"
                    continue
                end
                # End if puzzle solved
                if check_solved((s_free, s_fixed))
                    push!(new_df, [subj, prb, "win", (0, 0), (copy(s_free), s_fixed), attempt, 0, prev_t])
                    break
                end
                # Skip if car is not moved
                if n < size(prb_df, 1) && prb_df[n+1, :move] == row.move
                    continue
                end
                car_id = row.piece == 8 ? 1 : row.piece + 2
                # Get move amount
                m = 0
                if s_fixed[car_id].dir == :x
                    m = 1 + (row.target % 6) - s_free[car_id]
                else
                    m = 1 + (row.target ÷ 6) - s_free[car_id]
                end
                move = (Int8(car_id), Int8(m))
                push!(new_df, [subj, prb, "move", move, (copy(s_free), s_fixed), attempt, row.t - prev_t, row.t])
                make_move!(s_free, move)
                prev_t = row.t
            end
        end
    end
    return new_df
end

function load_messy_raw_data()
    csv_reader = CSV.File("data/raw_data/trialdata_headers.csv");
    data = process_messy_data(csv_reader)
    return data
end

"""
    process_messy_data(csv)

Pre-process csv DataFrame into dictionary where keys are subject identifiers and values are dataframes containing subject data
"""
function process_messy_data(csv)
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
                data[row.subject] = DataFrame(subject=String[], event=String[], move=Int[], instance=String[], t=Int[], piece=Int[], target=Int[])
            end
            # Split key and value for data points
            info_list_split = [split(l, ":") for l in info_list]
            # Create a dict to store all data
            info_dict = Dict{Symbol, Any}()
            info_dict[:subject] = row.subject
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

"""
    get_all_subj_moves(data)

Return four dictionaries with structure 
{subj:
  {prb: 
    [
      [values for first attempt]
      [values for second attempt]
      ...
    ]
  }
}
where values are lists of: 
- moves (Tuple{Int, Int})
- states (BigInt)
- times (Int)
"""
function get_all_subj_moves(data)
    all_subj_moves = Dict{String, Dict{String, Vector{Tuple{Int, Int}}}}()
    all_subj_states = Dict{String, Dict{String, Vector{BigInt}}}()
    all_subj_times = Dict{String, Dict{String, Vector{Int}}}()
    Ls = DefaultDict{String, Array{Int}}([])
    for subj in collect(keys(data))
        tot_moves, tot_states_visited, tot_times, attempts = analyse_subject(data[subj]);
        all_subj_moves[subj] = tot_moves
        all_subj_states[subj] = tot_states_visited
        all_subj_times[subj] = tot_times
        for prb in keys(attempts)
            push!(Ls[prb], attempts[prb])
        end
    end
    return all_subj_moves, all_subj_states, all_subj_times, Ls
end





all_subj_moves, all_subj_states, all_subj_times, Ls = get_all_subj_moves(data);
prbs = collect(keys(Ls))[sortperm([parse(Int, x[end-1] == '_' ? x[end] : x[end-1:end]) for x in keys(Ls)])];
subjs = collect(keys(data));

d_goals = load("d_goals.jld2")["d_goals"];

board = load_data(prbs[41])
make_move!(board, (6, -1))

mvs = [(4, -2), (2, -2), (3, 2), (5, 3), (9, 1), (6, -3), (9, -1), (5, -3), (3, -3), (5, 3), (2, 2), (4, 3)];

for (n, mv) in enumerate(mvs)
    make_move!(board, mv)
    draw_board(get_board_arr(board))
    savefig("move$(n).svg")
end



attempted = DefaultDict{String, Int64}(0)
as = []

for subj in subjs
    a = 0
    for prb in unique(data[subj].instance)
        if prb in prbs
            attempted[prb] += 1
            a += 1
        end
    end
    push!(as, length(keys(all_subj_moves[subj]))/a)
end

y1 = [[], [], [], []];
y2 = [[], [], [], []];
y3 = [[], [], [], []];
diffs = Dict(6 => 1, 10 => 2, 13 => 3, 15 => 4)

for prb in prbs
    diff = parse(Int64, split(prb, "_")[2])-1
    push!(y1[diffs[diff]], mean(Ls[prb] .== diff))
    push!(y2[diffs[diff]], attempted[prb]/length(subjs))
    push!(y3[diffs[diff]], length(Ls[prb])/attempted[prb])
end

plot(layout=grid(1, 3), size=(372*2, 300), grid=false, dpi=300,         
    legendfont=font(14, "helvetica"), 
    xtickfont=font(12, "helvetica"), 
    ytickfont=font(12, "helvetica"), 
    titlefont=font(14, "helvetica"), 
    guidefont=font(14, "helvetica"),
    right_margin=0Plots.mm, top_margin=1Plots.mm, bottom_margin=6Plots.mm, left_margin=4Plots.mm, 
    fontfamily="helvetica", tick_direction=:out)
bar!(string.(sort(collect(keys(diffs)))), mean.(values(y1)), yerr=sem.(values(y1)), sp=1, label=nothing, xlabel="", ylabel="Proportion optimal", ylim=(0, 0.32), c=:transparent, ms=10)
bar!(string.(sort(collect(keys(diffs)))), mean.(values(y2)), yerr=sem.(values(y2)), sp=2, label=nothing, xlabel="Length", ylabel="Attempt rate", ylim=(0, 1), c=:transparent, ms=10)
bar!(string.(sort(collect(keys(diffs)))), mean.(values(y3)), yerr=sem.(values(y3)), sp=3, label=nothing, xlabel="", ylabel="Completion rate", ylim=(0, 1), c=:transparent, ms=10)
