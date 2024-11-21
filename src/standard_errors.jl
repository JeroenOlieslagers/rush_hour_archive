
function generate_mu(M, N_is, sigma, sigma_p)
    # true means of individual participants
    mu_is = randn(M).*sigma
    # generate trial level data
    x = [mu_is[i] .+ randn(N_is[i])*sigma_p for i in 1:M]
    # calculate estimate of true mean of each participant based on trial data
    mu_bar_is = [mean(x[i]) for i in 1:M]
    # estimate of population average based on true averages
    mu_bar_1 = mean(mu_is)
    # estimate of population average based on estimated averages
    mu_bar_2 = mean(mu_bar_is)
    # # estimate of sigma based on true averages
    # sigma_bar = std(mu_is)
    # # estimate of of sigma' based on trial data
    # sigma_p_bar = std(reduce(vcat, x))
    return mu_bar_1, mu_bar_2#, sigma_bar, sigma_p_bar
end

function standard_error_simulations()
    # coarseness of x-axis
    n_X = 20
    # number of simulations to average across
    T = 1000
    # x-axis values
    Ms = round.(10 .^(range(1,stop=3,length=n_X)))
    b_Ns = round.(10 .^(range(1,stop=3,length=n_X)))
    sigmas = 10 .^(range(-1,stop=1,length=n_X))
    sigma_ps = 10 .^(range(-1,stop=1,length=n_X))
    xs = [Ms, b_Ns, sigmas, sigma_ps]
    # fixed values
    M = 100
    b_N = 10
    sigma = 1
    sigma_p = 10
    fixed = [M, b_N, sigma, sigma_p]
    # these will be filled in. first row is 1 layer empirical, second row is 1 layer analytical, third row is 2 layer empirical, fourth row is 2 layer analytical
    ys = zeros(n_X, 4, 4)

    for i in 1:4
        for j in ProgressBar(1:n_X)
            if i == 1
                M = Int(xs[1][j])
                b_N = fixed[2]
                sigma = fixed[3]
                sigma_p = fixed[4]
            elseif i == 2
                M = fixed[1]
                b_N = Int(xs[2][j])
                sigma = fixed[3]
                sigma_p = fixed[4]
            elseif i == 3
                M = fixed[1]
                b_N = fixed[2]
                sigma = xs[3][j]
                sigma_p = fixed[4]
            elseif i == 4
                M = fixed[1]
                b_N = fixed[2]
                sigma = fixed[3]
                sigma_p = xs[4][j]
            end
            mu_bar_1s = zeros(T)
            mu_bar_2s = zeros(T)
            # number of trials per participant
            N_is = Int.(b_N .+ round.(b_N .* rand(M)));
            for k in 1:T
                mu_bar_1, mu_bar_2 = generate_mu(M, N_is, sigma, sigma_p)
                mu_bar_1s[k] = mu_bar_1
                mu_bar_2s[k] = mu_bar_2
            end
            ys[j, i, 1] = std(mu_bar_1s)
            ys[j, i, 2] = sigma/sqrt(M)
            ys[j, i, 3] = std(mu_bar_2s)
            ys[j, i, 4] = sqrt(sigma^2/M + sigma_p^2*sum(1 ./ N_is)/M^2)
        end
    end
    return xs, ys
end