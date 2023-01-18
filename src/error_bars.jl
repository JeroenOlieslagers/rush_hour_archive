using Plots

function generate_mu(M, Nis, sigma, SIGMA)
    #Random.seed!(0)
    mus = randn(M).*SIGMA # pop std of sigma/sqrt(M)
    
    x = [mus[i] .+ randn(Nis[i])*sigma for i in 1:M]
    mu_hat = [mean(x[i]) for i in 1:M]

    return mean(mu_hat), mean(mus), std(mus), std(reduce(vcat, x))
end

tt = 50;
T = 1000;
Ms = Int.(round.(10 .^(range(1,stop=3,length=tt))));
sigmas = 10 .^(range(-1,stop=1,length=tt));
SIGMAs = 10 .^(range(-1,stop=1,length=tt));
b_Ns = Int.(round.(10 .^(range(1,stop=3,length=tt))));

#M = 100;
sigma = 20;
SIGMA = 10;

one_layer_emp = zeros(tt);
one_layer_ana = zeros(tt);
two_layer_emp = zeros(tt);
two_layer_ana = zeros(tt);

error_bar_1 = zeros(tt, T);
error_bar_2 = zeros(tt, T);

for j in ProgressBar(1:tt)
    MUs = zeros(T)
    mus = zeros(T)
    M = Ms[j]
    #sigma = sigmas[j]
    #SIGMA = SIGMAs[j]
    #Nis = Int.(b_Ns[j] .+ round.(b_Ns[j] .* rand(M)));
    #Nis = Int.(20 .+ round.(30 .* rand(M)));
    Nis = Int.(ones(M) .* 30);
    #Nis = Int.(round.(20 .+ randn(M) .* 5));
    for i in 1:T
        MU, mu, SIG, sig = generate_mu(M, Nis, sigma, SIGMA)
        MUs[i] = MU
        mus[i] = mu
        error_bar_1[j, i] = SIG/sqrt(M)
        error_bar_2[j, i] = sqrt(SIG*SIG/M + sig*sig*sum(1 ./ Nis)/(M*M))
    end
    one_layer_emp[j] = std(mus)
    one_layer_ana[j] = SIGMA/sqrt(M)
    two_layer_emp[j] = std(MUs)
    two_layer_ana[j] = sqrt(SIGMA*SIGMA/M + sigma*sigma*sum(1 ./ Nis)/(M*M))
end

dotplot([[i] for i in 1:tt], [error_bar_1[i, :] for i in 1:tt], label=nothing, xticks=(1:tt, [M for M in Ms]), alpha=0.1, markersize=1, c=:blue)
scatter!(1:tt, one_layer_ana, c=:blue)
dotplot!([[i] for i in 1:tt], [error_bar_2[i, :] for i in 1:tt], label=nothing, xticks=(1:tt, [M for M in Ms]), alpha=0.1, markersize=1, c=:red)
scatter!(1:tt, two_layer_ana, c=:red)

Plots.scatter(Ms, one_layer_emp, xscale=:log10, label="1 layer analytical", alpha=0.2, color=:blue, xlabel=latexstring("b_N"), ylabel=latexstring("\\hat\\sigma"), xticks=[10, 100, 1000], title=latexstring("\\bar\\sigma=2, \\sigma=10, M=100, N_i \\sim\\mathcal{U}(b_N, 2b_N)"), size=(450, 250), foreground_color_legend = nothing, background_color_legend = nothing, legend=:topright, dpi=300)
plot!(Ms, one_layer_ana, xscale=:log10, label="1 layer empirical", color=:blue, linewidth=2)
scatter!(Ms, two_layer_emp, xscale=:log10, label="2 layer empirical", alpha=0.2, color=:red)
plot!(Ms, two_layer_ana, xscale=:log10, label="2 layer analytical", color=:red, linewidth=2)

Plots.plot(Ms, two_layer_ana, label="1 layer analytical", alpha=0.2, color=:blue, xlabel=latexstring("b_N"), ylabel=latexstring("\\hat\\sigma"), xticks=[10, 100, 1000], title=latexstring("\\bar\\sigma=2, \\sigma=10, M=100, N_i \\sim\\mathcal{U}(b_N, 2b_N)"), size=(450, 250), foreground_color_legend = nothing, background_color_legend = nothing, legend=:topright, dpi=300)

println("STD OF TRUE MUS")
println("EMPIRICAL: $(std(mus))")
println("ANALYTICAL: $(SIGMA/sqrt(M))")

println("STD OF ESTIMATED MUS")
println("EMPIRICAL: $(std(MUs))")
println("ANALYTICAL: $(sqrt(SIGMA*SIGMA/M + sigma*sigma*sum(1 ./ Nis)/(M*M)))")

MU, mu, SIG, sig = generate_mu(M, Nis, sigma, SIGMA)
println("TRUE SIGMA: $(SIGMA)")
println("ESTIMATED SIGMA: $(SIG)")

println("TRUE sigma: $(sigma)")
println("ESTIMATED sigma: $(sig)")


alpha = zeros(1000)
Ns = Int.(round.(10 .^range(1, 4, 50)))
stds = []
for N in ProgressBar(Ns)
    for i in 1:1000
        x = rand(Beta(2.0, 1.0), N)

        m = mean(x)
        v = var(x)
        alpha[i] = m*(m*(1-m)/v) - m
        beta[i] = (1-m)*(m*(1-m)/v) - (1-m)
    end
    x = rand(Beta(2.0, 1.0), N)
    m = mean(x)
    v = var(x)
    a = m*(m*(1-m)/v) - m
    b = (1-m)*(m*(1-m)/v) - (1-m)
    s = zeros(1000)
    for i in 1:1000
        s[i] = mean(rand(Beta(a, b), N))
    end
    scatter!(N, m, yerr=([percentile(s, 5)], [percentile(s, 95)]), label=nothing)
    dotplot!(N, s)

    push!(stds, std(alpha))
end

println(mean(alpha))
println(std(alpha))



Ns = Int.(round.(10 .^range(1, 4, 50)))
res = []
for N in ProgressBar(Ns)
    T = 10000
    nom = zeros(T)
    denom = zeros(T)
    frac = zeros(T)
    for i in 1:T
        x = rand(Beta(2.0, 1.0), N)
        sm = mean(x)
        v = var(x)
        nom[i] = sm*sm*(1 - sm)
        denom[i] = v
        frac[i] = nom[i]/denom[i]
    end
    push!(res, abs(mean(frac) - mean(nom)/mean(denom)))
end