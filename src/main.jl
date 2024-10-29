include("load_scripts.jl")

df, messy_data, d_goals_prbs, df_models, df_stats, binned_stats = load_processed_data();
subjs, prbs = get_subjs_prbs(df);

fig2D(df, prbs)
fig4(df_models)
fig5_(binned_stats)
fig6_(df_stats)
fig7_(binned_stats)

supplement_fig2(df, messy_data)
supplement_fig4(df)
supplement_fig5(df, d_goals_prbs)
supplement_fig6(binned_stats)
supplement_fig7(df_stats)
supplement_fig8(binned_stats)

mc_dict = load("data/processed_data/mc_dict.jld2")
dict = load("data/processed_data/dict.jld2")
df_stats = calculate_summary_statistics(df, df_models, d_goals_prbs, mc_dict, dict)
binned_stats = bin_stats(df_stats, :X_d_goal)



