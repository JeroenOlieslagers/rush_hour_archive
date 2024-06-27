
function max_20(col)
    col[col .> 20] .= 20
    return col
end

function normalize_hist_counts(df, v1, v2, idv, lims)
    df[!, :norm_counts] = zeros(Float64, size(df, 1))
    for a in unique(df[!, v1])
        for b in unique(df[!, v2])
            dummy = df[df[!, v1] .== a .&& df[!, v2] .== b, :].hist_counts
            df[df[!, v1] .== a .&& df[!, v2] .== b, :norm_counts] .= dummy ./ sum(dummy)
            for d in lims
                df_ = df[df[!, v1] .== a .&& df[!, v2] .== b, :]
                if d ∉ df_[!, idv]
                    push!(df, [a, b, d, 0, 0])
                end
            end
        end
    end
    return df
end

function puzzle_statistics(df)
    puzzle_df = DataFrame(subject=String[], puzzle=String[], Lopt=Int[], L=Int[], T=Int[])
    for subj in unique(df.subject)
        subj_df = df[df.subject .== subj .&& df.event .== "move", :]
        for prb in unique(subj_df.puzzle)
            prb_df = subj_df[subj_df.puzzle .== prb, :]
            Lopt = parse(Int, prb[end-1] == '_' ? prb[end] : prb[end-1:end]) - 2
            push!(puzzle_df, [subj, prb, Lopt, size(prb_df, 1), sum(prb_df.RT)])
        end
    end
    return puzzle_df
end

function add_completion_data!(opt_df, messy_data, prbs)
    opt_df[!, :attempted] = zeros(Int64, 167)
    opt_df[!, :total] = zeros(Int64, 167)

    for subj in unique(opt_df.subject)
        for prb in prbs
            diff = parse(Int64, split(prb, "_")[2])-2
            _df = opt_df[opt_df.subject .== subj, :]
            if diff ∉ _df.Lopt
                push!(opt_df, [subj, diff, 0, 0, 0, 0])
            end
            if prb in unique(messy_data[subj].instance)
                opt_df[opt_df.subject .== subj .&& opt_df.Lopt .== diff, :attempted] .+= 1
            end
            opt_df[opt_df.subject .== subj .&& opt_df.Lopt .== diff, :total] .+= 1
        end
    end
    return nothing
end

function difficulty_stats(opt, attempted, completed, total)
    opt_rate = opt[completed .> 0] ./ completed[completed .> 0]
    attempt_rate = attempted ./ total
    completion_rate = completed ./ attempted
    return [(mean(opt_rate), sem(opt_rate), mean(attempt_rate), sem(attempt_rate), mean(completion_rate), sem(completion_rate))]
end