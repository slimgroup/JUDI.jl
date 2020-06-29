using JUDI.TimeModeling, PyCall, LinearAlgebra, PyPlot, Test

model= Model((51, 51), (10., 10.), (0., 0.), rand(Float32, 51, 51);nb=10)
modelPy = devito_model(model, Options())

v0 = sqrt.(1f0 ./model.m)
cut = remove_padding(deepcopy(modelPy.vp.data[:, :]), modelPy.padsizes; true_adjoint=true)
vpdata = deepcopy(modelPy.vp.data)

@show dot(v0, cut)/dot(vpdata, vpdata)
@test isapprox(dot(v0, cut), dot(vpdata, vpdata))


model= Model((51, 51, 51), (10., 10., 10.), (0., 0., 0.), rand(Float32, 51, 51, 51);nb=10)
modelPy = devito_model(model, Options())

v0 = sqrt.(1f0 ./model.m)
cut = remove_padding(deepcopy(modelPy.vp.data[:, :, :]), modelPy.padsizes; true_adjoint=true)
vpdata = deepcopy(modelPy.vp.data)

@show dot(v0, cut)/dot(vpdata, vpdata)
@test isapprox(dot(v0, cut), dot(vpdata, vpdata))