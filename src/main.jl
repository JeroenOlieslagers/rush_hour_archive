using RushHour

d_goals_prbs = load("data/processed_data/d_goals_prbs.jld2")

# LOAD RAW DATA
#messy_data = load_messy_raw_data();
df_raw = load_raw_data();
df_filtered = filter_subjects(df_raw);
df = pre_process(df_filtered, d_goals_prbs);
#prbs = ["prb29232_7", "prb10206_7", "prb11647_7", "prb2834_7", "prb14651_7", "prb32695_7", "prb12604_7", "prb21272_7", "prb26567_7", "prb32795_7", "prb20059_7", "prb1707_7", "prb14047_7", "prb15290_7", "prb13171_7", "prb28111_7", "prb8786_7", "prb23259_7", "prb79230_11", "prb54081_11", "prb3217_11", "prb29414_11", "prb33509_11", "prb31907_11", "prb42959_11", "prb68910_11", "prb62015_11", "prb14898_11", "prb9718_11", "prb38526_11", "prb717_11", "prb62222_11", "prb34092_11", "prb12715_11", "prb22436_11", "prb46224_11", "prb23404_14", "prb34551_14", "prb19279_14", "prb55384_14", "prb6671_14", "prb20888_14", "prb343_14", "prb29585_14", "prb65535_14", "prb3203_14", "prb47495_14", "prb29600_14", "prb14485_14", "prb68514_14", "prb33117_14", "prb72800_14", "prb38725_14", "prb44171_16", "prb58853_16", "prb15595_16", "prb48146_16", "prb45893_16", "prb78361_16", "prb57223_16", "prb24227_16", "prb1267_16", "prb25861_16", "prb10166_16", "prb24406_16", "prb25604_16", "prb46580_16", "prb29027_16", "prb46639_16", "prb54506_16"]
prbs = unique(df.puzzle)[sortperm([parse(Int, x[end-1] == '_' ? x[end] : x[end-1:end]) for x in unique(df.puzzle)])];
subjs = unique(df.subject);


#stuff = first_pass(df);
#save("data/processed_data/stuff.jld2", stuff)

stuff = load("data/processed_data/stuff.jld2");

df[!, :tree] = stuff["trees"];
df[!, :dict] = stuff["dicts"];
df[!, :all_moves] = stuff["all_moves"];
df[!, :neighs] = stuff["neighs"];
df[!, :features] = stuff["features"];


#df_models, params = fit_all_models(df, d_goals_prbs)
#CSV.write("data/processed_data/df_models.csv", df_models)
#save("data/processed_data/params.jld2", "params", params)

df_models = CSV.read("data/processed_data/df_models.csv", DataFrame);
params = load("data/processed_data/params.jld2")["params"];

df_models[!, :params] = params;


#df_stats = calculate_summary_statistics(df, df_models, d_goals_prbs)
#CSV.write("data/processed_data/df_stats.csv", df_stats)

#binned_stats = bin_stats(df_stats, :X_d_goal)
#CSV.write("data/processed_data/binned_stats.csv", binned_stats)

df_stats = CSV.read("data/processed_data/df_stats.csv", DataFrame)
binned_stats = CSV.read("data/processed_data/binned_stats.csv", DataFrame)


fig2D(df)
fig4(df_models)
fig5(binned_stats)
fig6(df_stats)
fig7(binned_stats)

supplement_fig2(df, messy_data)
supplement_fig4(df)
supplement_fig5(df)



#include("markov_chains.jl")

# dict = get_QR_dict(prbs)
# save("data/processed_data/dict.jld2", dict)

dict = load("data/processed_data/dict.jld2")




