function marginal_histogram(x, y; bins=[], xlim=(), ylim=(), xlabel="", ylabel="", size=(400, 400))
    l = @layout [
        tophist _
        hist2d{0.9w,0.9h} righthist
    ]
    plot(layout=l, link=:both, grid=false, size=size)
    histogram2d!(A, opt_L, sp=2, xlabel=xlabel, ylabel=ylabel, bins=bins, xlim=xlim, ylim=ylim, color=cgrad(:grays, rev=true), colorbar=false)
    histogram!(A, sp=1, xlim=xlim, ylim=(0, Inf), c=:black, bins=bins[1], xticks=nothing, yticks=nothing, label=nothing)
    histogram!(opt_L, sp=3, ylim=ylim, xlim=(0, Inf), c=:black, bins=bins[2], orientation=:h, xticks=nothing, yticks=nothing, label=nothing)
    display(plot!())
end