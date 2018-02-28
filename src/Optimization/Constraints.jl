export boundproject

function boundproject(x,UB,LB)
    x[x .< LB] = LB
    x[x .> UB] = UB
    return x
end
