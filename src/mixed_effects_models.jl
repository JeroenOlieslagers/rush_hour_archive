
function all_mixed_models(df_stats)
    # only participant data
    df_ = df_stats[df_stats.model .== "data", :]
    # only random model data
    dfr_ = df_stats[df_stats.model .== "random_model", :]
    #dfm_ = df_stats[df_stats.model .== "gamma_only_model", :]
    # only participant data where the participant took a sensible action
    dff_ = df_[df_.y_p_in_tree .== 1, :]
    # only random model data where the the random action was sensible action
    dffr_ = dfr_[dfr_.y_p_in_tree .== 1, :]
    # only participat data where the first move is excluded
    ddf_ = df_[df_.X_first_move .== 0, :]
    # figure 4A
    display(fit(MixedModel, @formula(y_p_in_tree ~ 1 + (1|subject) + X_d_goal), df_))
    # figure 4B
    display(fit(MixedModel, @formula(y_p_in_tree ~ 1 + (1|subject) + X_d_goal), dfr_))
    # figure 4C
    display(fit(MixedModel, @formula(y_d_tree ~ 1 + (1|subject) + X_d_goal), dffr_))
    # figure 5A
    display(fit(MixedModel, @formula(log10(X_rt) ~ 1 + (1|subject) + X_first_move), df_))
    # figure 5B
    display(fit(MixedModel, @formula(log10(X_rt) ~ 1 + (1|subject) + X_d_goal), ddf_))
    # figure 5C
    display(fit(MixedModel, @formula(log10(X_rt) ~ 1 + (1|subject) + y_d_tree), dff_))
    # fit(MixedModel, @formula(y_p_in_tree ~ 1 + (1|subject) + X_d_goal), dfm_)
end