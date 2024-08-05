include("load_scripts.jl")

df, messy_data, d_goals_prbs, df_models, df_stats, binned_stats = load_processed_data();
subjs, prbs = get_subjs_prbs(df);

fig2D(df, prbs)
fig4(df_models)
fig5(binned_stats)
fig6(df_stats)
fig7(binned_stats)

supplement_fig2(df, messy_data)
supplement_fig4(df)
supplement_fig5(df, d_goals_prbs)
supplement_fig6(df)
supplement_fig7(df)
supplement_fig8(df)
