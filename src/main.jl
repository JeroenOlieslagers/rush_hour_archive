using Pkg

Pkg.activate(".")
Pkg.instantiate()

include("load_scripts.jl")

df, messy_data, d_goals_prbs, df_models, df_stats, binned_stats = load_processed_data();
subjs, prbs = get_subjs_prbs(df);

fig2A(prbs)
fig2C(df, prbs)
fig2D(df, prbs)
fig4(binned_stats)
fig5(df_stats)
fig6AD(binned_stats)
fig6EF(df_stats)
fig6GI(binned_stats)
fig7(df_models)

# this will print all mixed effect model stats
all_mixed_models(df_stats)

fig_ext1(prbs)
fig_ext2(df, messy_data)
fig_ext3(df)
fig_ext4(df, d_goals_prbs) # this will take a minute or two to run
fig_ext6(binned_stats)
fig_ext7(df_stats)
fig_ext8(binned_stats)

xs, ys = standard_error_simulations();
true_params = load("data/processed_data/true_params.jld2")["true_params"];
fitted_params = load("data/processed_data/fitted_params.jld2")["fitted_params"];

fig_supp1(xs, ys)
fig_supp2(true_params, fitted_params)


# mc_dict = load("data/processed_data/mc_dict.jld2")
# dict = load("data/processed_data/dict.jld2")



