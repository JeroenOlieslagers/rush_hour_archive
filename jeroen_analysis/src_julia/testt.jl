using BenchmarkTools

function brownian_path()
    x0 = randn()
    x = zeros(100000)
    x[1] = x0
    a = 0.9999
    b = 0.1
    for i in 1:(100000-1)
        x[i+1] = a*x[i] + b*randn()
    end
    return x
end

@btime brownian_path()


