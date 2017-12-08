
export fwi_objective

function fwi_objective(model_full::Model, source::judiVector, dObs::judiVector, srcnum::Int64; options=Options(), frequencies=[])
# Setup time-domain linear or nonlinear foward and adjoint modeling and interface to OPESCI/devito 
 	
	# Load full geometry for out-of-core geometry containers
	typeof(dObs.geometry) == GeometryOOC && (dObs.geometry = Geometry(dObs.geometry))
	typeof(source.geometry) == GeometryOOC && (source.geometry = Geometry(source.geometry))
    length(model_full.n) == 3 ? dims = (3,2,1) : dims = (2,1)   # model dimensions for Python are (z,y,x) and (z,x)

	# for 3D modeling, limit model to area with sources/receivers
	if options.limit_m == true && model_full.n[3] > 1	# only supported for 3D
		model = deepcopy(model_full)
		model = limit_model_to_receiver_area(source.geometry,dObs.geometry,model,options.buffer_size)
	else
		model = model_full
	end
	
	# Source/receiver parameters
	tmaxSrc = source.geometry.t[1]
	tmaxRec = dObs.geometry.t[1]

	# Extrapolate input data to computational grid
	dtComp = calculate_dt(model.n,model.d,model.o,sqrt.(1f0./model.m))
	qIn = time_resample(source.data[1],source.geometry,dtComp)[1]
	if typeof(dObs.data[1]) == SeisIO.SeisCon
		data = convert(Array{Float32,2},dObs.data[1][1].data)
		dObs = judiVector(dObs.geometry,data)
	end
	dObserved = time_resample(dObs.data[1],dObs.geometry,dtComp)[1]
    ntComp = size(dObserved,2)
	ntSrc = Int(trunc(tmaxSrc/dtComp+1))
	ntRec = Int(trunc(tmaxRec/dtComp+1))

	# Set up coordinates
	src_coords = setup_grid(source.geometry, model.n, model.o)
	rec_coords = setup_grid(dObs.geometry, model.n, model.o)

	# Forward modeling to generate synthetic data and background wavefields
    if isempty(frequencies)
        dPredicted, u0 = pycall(ac.forward_modeling, PyObject, model.n, model.d, model.o, PyReverseDims(permutedims(model.m,dims)), 
                                PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'), save=true)
    else
        dPredicted, ufr, ufi = pycall(ac.forward_freq_modeling, PyObject, model.n, model.d, model.o, PyReverseDims(permutedims(model.m,dims)), 
            PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'), length(frequencies), frequencies)
    end


	# Data misfit
	argout1 = .5f0*norm(vec(dPredicted) - vec(dObserved),2)^2.f0

	# Backpropagation of data residual
    if isempty(frequencies)
    	argout2 = pycall(ac.adjoint_born, Array{Float32,length(model.n)}, model.n, model.d, model.o, PyReverseDims(permutedims(model.m,dims)), 
                         PyReverseDims(src_coords'), PyReverseDims(rec_coords'), PyReverseDims((dPredicted  - dObserved)'), u0)
    else
	    argout2 = pycall(ac.adjoint_freq_born, Array{Float32,length(model.n)}, model.n, model.d, model.o, PyReverseDims(permutedims(model.m,dims)), 
                         PyReverseDims(src_coords'), PyReverseDims(rec_coords'), PyReverseDims((dPredicted  - dObserved)'), frequencies, ufr, ufi)
    end
    argout2 = remove_padding(argout2,model.nb,true_adjoint=options.sum_padding)
	if options.limit_m==true && length(model_full.n) == 3
		argout2 = extend_gradient(model_full,model,argout2)
	end
	return [argout1; vec(argout2)]
end


