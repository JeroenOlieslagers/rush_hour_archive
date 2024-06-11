
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
    for subject in unique(df.subject)
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
        ##
        if (!(last(tt) - first(tt) >= 60 || length(unique(subj_data.instance)) == 70)) || ("win" ∉ subj_data.event)
            counter += 1
            continue
        end
        ##
        # Exclude subjects based on total length and breaks
        # if last(tt) - first(tt) < 30
        #     counter += 1
        #     continue
        # # elseif (last(tt) - first(tt) < 45) && maximum(intervals) > 5
        # #     counter += 1
        # #     continue
        # elseif (last(tt) - first(tt) < 60) && maximum(intervals) > 10
        #     counter += 1
        #     continue
        # elseif maximum(intervals) > 15
        #     counter += 1
        #     continue
        # end
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
    println("Rejection ratio: " * string(counter) * "/" * string(length(unique(df.subject))))
    return filtered_data
end

"""
    pre_process(df)

Take in DataFrame of subject data, filter it and return DataFrame where colums are 
subject ID, puzzle ID, move, state, distance to goal, difficulty of puzzle, time stamp, attempt
"""
function pre_process(df, d_goals_prbs)
    new_df = DataFrame(subject=String[], puzzle=String[], event=String[], move=a_type[], prev_move=a_type[], s_free=s_free_type[], s_fixed=s_fixed_type[], d_goal=Int[], Lopt=Int[], attempt=Int[], RT=Int[], timestamp=Int[])
    for subj in unique(df.subject)
        subj_df = df[df.subject .== subj .&& df.event .∈ Ref(["start", "restart", "drag_end", "win"]), :]
        for prb in unique(subj_df.instance)
            prb_df = subj_df[subj_df.instance .== prb, :]
            s_free, s_fixed = load_data(prb)
            attempt = 0
            prev_t = first(prb_df).t
            prev_move = (Int8(0), Int8(0))
            Lopt = parse(Int, prb[end-1] == '_' ? prb[end] : prb[end-1:end]) - 2
            # Don't consider puzzles that are not solved
            # if "win" ∉ prb_df.event
            #     continue
            # end
            if prb_df[end, :event] != "win"
                continue
            end
            for (n, row) in enumerate(eachrow(prb_df))
                # Start of trial / after restart
                d_goal = d_goals_prbs[prb][board_to_int32((s_free, s_fixed))[2]]
                if row.event == "start"
                    attempt += 1
                    if attempt == 1
                        s_free, _ = load_data(prb)
                        push!(new_df, [subj, prb, "start", (0, 0), prev_move, copy(s_free), s_fixed, d_goal, Lopt, attempt, row.t - prev_t, row.t])
                    else
                        if last(new_df).event !== "restart" && last(new_df).event !== "start"
                            push!(new_df, [subj, prb, "restart", (0, 0), prev_move, copy(s_free), s_fixed, d_goal, Lopt, attempt-1, row.t - prev_t, row.t])
                        else
                            attempt -= 1
                        end
                        s_free, _ = load_data(prb)
                    end
                    prev_t = row.t
                    prev_move = (Int8(0), Int8(0))
                    continue
                elseif row.event == "win" || row.event == "restart"
                    continue
                end
                # End if puzzle solved
                if check_solved((s_free, s_fixed))
                    push!(new_df, [subj, prb, "win", (0, 0), prev_move, copy(s_free), s_fixed, d_goal, Lopt, attempt, 0, prev_t])
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
                push!(new_df, [subj, prb, "move", move, prev_move, copy(s_free), s_fixed, d_goal, Lopt, attempt, row.t - prev_t, row.t])
                make_move!(s_free, move)
                prev_t = row.t
                prev_move = move
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
