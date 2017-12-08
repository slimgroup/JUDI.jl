using PyPlot

export time_modeling

# Setup time-domain linear or nonlinear foward and adjoint modeling and interface to OPESCI/devito 
function time_modeling(model_full::Model, srcGeometry::Geometry, srcData, recGeometry::Geometry, recData, dm, srcnum::Int64, op::Char, mode::Int64, options)

	# Load full geometry for out-of-core geometry containers
	typeof(recGeometry) == GeometryOOC && (recGeometry = Geometry(recGeometry))
	typeof(srcGeometry) == GeometryOOC && (srcGeometry = Geometry(srcGeometry))
    length(model_full.n) == 3 ? dims = (3,2,1) : dims = (2,1)   # model dimensions for Python are (z,y,x) and (z,x)

	# for 3D modeling, limit model to area with sources/receivers
	if options.limit_m == true && length(model_full.n) == 2	# only supported for 3D
		model = deepcopy(model_full)
		if op=='J' && mode==1
			model,dm = limit_model_to_receiver_area(srcGeometry,recGeometry,model,options.buffer_size;pert=dm)
		else
			model = limit_model_to_receiver_area(srcGeometry,recGeometry,model,options.buffer_size)
		end
	else
		model = model_full
	end

	# Computational time step
	dtComp = calculate_dt(model.n,model.d,model.o,sqrt.(1f0./model.m))

	# Source/receiver parameters
	tmaxSrc = srcGeometry.t[1]
	tmaxRec = recGeometry.t[1]

	# Extrapolate input data to computational grid
	if mode==1
		qIn = time_resample(srcData[1],srcGeometry,dtComp)[1]
		ntComp = size(qIn,1)
	elseif op=='F' &&  mode==-1
		if typeof(recData[1]) == SeisIO.SeisCon
			recDataCell = Array{Any}(1); recDataCell[1] = convert(Array{Float32,2},recData[1][1].data); recData = recDataCell
		elseif typeof(recData[1]) == String
			recData = load(recData[1])["d"].data
		end
		dIn = time_resample(recData[1],recGeometry,dtComp)[1]
		ntComp = size(dIn,1)
	elseif op=='J' && mode==-1
		if typeof(recData[1]) == SeisIO.SeisCon
			recDataCell = Array{Any}(1); recDataCell[1] = convert(Array{Float32,2},recData[1][1].data); recData = recDataCell
		elseif typeof(recData[1]) == String
			recData = load(recData[1])["d"].data
		end
		qIn = time_resample(srcData[1],srcGeometry,dtComp)[1]
		dIn = time_resample(recData[1],recGeometry,dtComp)[1]
		ntComp = size(dIn,1)
	end
	ntSrc = Int(trunc(tmaxSrc/dtComp + 1))
	ntRec = Int(trunc(tmaxRec/dtComp + 1))
	
	# Set up coordinates
	src_coords = setup_grid(srcGeometry, model.n, model.o)
	rec_coords = setup_grid(recGeometry, model.n, model.o)

	if op=='F'
		if mode==1
            # forward modeling
            #println("Nonlinear forward modeling (source no. ",srcnum,")")
            dOut = pycall(ac.forward_modeling, PyObject, model.n, model.d, model.o, PyReverseDims(permutedims(model.m,dims)), 
                          PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'))[1]
            ntRec > ntComp && (dOut = [dOut zeros(size(dOut,1), ntRec - ntComp)])
            dOut = time_resample(dOut,dtComp,recGeometry)
            if options.save_data_to_disk
                container = write_shot_record(srcGeometry,srcData,recGeometry,dOut,options)
                return judiVector(container)
            else 
                return judiVector(recGeometry,dOut)
            end
		else
            # adjoint modeling
            #println("Nonlinear adjoint modeling (source no. ",srcnum,")")
            qOut = pycall(ac.adjoint_modeling, Array{Float32,2}, model.n, model.d, model.o, PyReverseDims(permutedims(model.m,dims)), 
                          PyReverseDims(src_coords'), PyReverseDims(rec_coords'), PyReverseDims(dIn'))
            ntSrc > ntComp && (qOut = [qOut zeros(size(qOut), ntSrc - ntComp)])
            qOut = time_resample(qOut,dtComp,srcGeometry)
            return judiVector(srcGeometry,qOut)
		end	
	elseif op=='J'
		if mode==1
            # forward linearized modeling
            #println("Linearized forward modeling (source no. ",srcnum,")")
            dOut = pycall(ac.forward_born, Array{Float32,2}, model.n, model.d, model.o, PyReverseDims(permutedims(model.m,dims)), 
                          PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'), 
                          PyReverseDims(permutedims(reshape(dm,model.n),dims)))
            ntRec > ntComp && (dOut = [dOut zeros(size(dOut,1), ntRec - ntComp)])
            dOut = time_resample(dOut,dtComp,recGeometry)
            if options.save_data_to_disk
                container = write_shot_record(srcGeometry,srcData,recGeometry,dOut,options)
                return judiVector(container)
            else 
                return judiVector(recGeometry,dOut)
            end
		else
            # adjoint linearized modeling
            #println("Linearized adjoint modeling (source no. ",srcnum,")")
            u0 = pycall(ac.forward_modeling, PyObject, model.n, model.d, model.o, PyReverseDims(permutedims(model.m,dims)), 
                        PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'), save=true)[2]
            grad = pycall(ac.adjoint_born, Array{Float32, length(model.n)}, model.n, model.d, model.o, PyReverseDims(permutedims(model.m,dims)), 
                          PyReverseDims(src_coords'), PyReverseDims(rec_coords'), PyReverseDims(dIn'), u0)
            grad = remove_padding(grad,model.nb,true_adjoint=options.sum_padding)
            if options.limit_m == true && length(model_full.n) == 3
                grad = extend_gradient(model_full,model,grad)
            end
            return vec(grad)
		end	
	end
end

# Function instance without options
time_modeling(model::Model, srcGeometry::Geometry, srcData, recGeometry::Geometry, recData, perturbation, srcnum::Int64, op::Char, mode::Int64) = 
	time_modeling(model, srcGeometry, srcData, recGeometry, recData, perturbation, srcnum, op, mode, Options())


