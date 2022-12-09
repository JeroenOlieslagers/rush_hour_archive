# include("engine.jl")
# include("solvers.jl")
# include("data_analysis.jl")


"""
    softmax(β, a, A)

``\\frac{e^{βa}}{\\sum_{a_i∈A}e^{βa_i}}``
"""
function softmax(β, a, A)
    return exp(β*a)/sum(exp.(β*A))
end