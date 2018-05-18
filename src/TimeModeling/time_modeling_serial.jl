
export time_modeling

# Setup time-domain linear or nonlinear foward and adjoint modeling and interface to OPESCI/devito
function time_modeling(model_full::Model, srcGeometry::Geometry, srcData, recGeometry::Geometry, recData, dm, srcnum::Int64, op::Char, mode::Int64, options)

    # Load full geometry for out-of-core geometry containers
    typeof(recGeometry) == GeometryOOC && (recGeometry = Geometry(recGeometry))
    typeof(srcGeometry) == GeometryOOC && (srcGeometry = Geometry(srcGeometry))
    length(model_full.n) == 3 ? dims = [3,2,1] : dims = [2,1]   # model dimensions for Python are (z,y,x) and (z,x)

    # for 3D modeling, limit model to area with sources/receivers
    if options.limit_m == true
        model = deepcopy(model_full)
        if op=='J' && mode==1
            model,dm = limit_model_to_receiver_area(srcGeometry,recGeometry,model,options.buffer_size;pert=dm)
        else
            model = limit_model_to_receiver_area(srcGeometry,recGeometry,model,options.buffer_size)
        end
    else
        model = model_full
    end

    # Set up Python model structure
    if op=='J' && mode == 1
        modelPy = pm.Model(origin=(0.,0.,0.), spacing=model.d, shape=model.n, vp=process_physical_parameter(sqrt.(1f0./model.m), dims), nbpml=model.nb,
                           rho=process_physical_parameter(model.rho, dims), dm=process_physical_parameter(reshape(dm,model.n), dims))
    else
        modelPy = pm.Model(origin=(0.,0.,0.), spacing=model.d, shape=model.n, vp=process_physical_parameter(sqrt.(1f0./model.m), dims), nbpml=model.nb,
                           rho=process_physical_parameter(model.rho, dims))
    end
    dtComp = modelPy[:critical_dt]

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
            dOut = pycall(ac.forward_modeling, PyObject, modelPy, PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'),
                          space_order=options.space_order, nb=model.nb)[1]
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
            qOut = pycall(ac.adjoint_modeling, Array{Float32,2}, modelPy, PyReverseDims(src_coords'), PyReverseDims(rec_coords'), PyReverseDims(dIn'),
                          space_order=options.space_order, nb=model.nb)
            ntSrc > ntComp && (qOut = [qOut zeros(size(qOut), ntSrc - ntComp)])
            qOut = time_resample(qOut,dtComp,srcGeometry)
            return judiVector(srcGeometry,qOut)
        end
    elseif op=='J'
        if mode==1
            # forward linearized modeling
            #println("Linearized forward modeling (source no. ",srcnum,")")
            dOut = pycall(ac.forward_born, Array{Float32,2}, modelPy, PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'),
                          space_order=options.space_order, nb=model.nb, isic=options.isic)
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
            if options.optimal_checkpointing == true
                op_F = pycall(ac.forward_modeling, PyObject, modelPy, PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'),
                              op_return=true, space_order=options.space_order, nb=model.nb)
                grad = pycall(ac.adjoint_born, Array{Float32, length(model.n)}, modelPy, PyReverseDims(rec_coords'), PyReverseDims(dIn'), op_forward=op_F,
                              space_order=options.space_order, nb=model.nb, is_residual=true, isic=options.isic)
            elseif ~isempty(options.frequencies)
                typeof(options.frequencies) == Array{Any,1} && (options.frequencies = options.frequencies[srcnum])
                d_pred, uf_real, uf_imag = pycall(ac.forward_freq_modeling, PyObject, modelPy, PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'),
                                          options.frequencies, space_order=options.space_order, nb=model.nb)
                grad = pycall(ac.adjoint_freq_born, Array{Float32, length(model.n)}, modelPy, PyReverseDims(rec_coords'), PyReverseDims(dIn'),
                              options.frequencies, uf_real, uf_imag, space_order=options.space_order, nb=model.nb)
            else
                u0 = pycall(ac.forward_modeling, PyObject, modelPy, PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'),
                            space_order=options.space_order, nb=model.nb, save=true)[2]
                grad = pycall(ac.adjoint_born, Array{Float32, length(model.n)}, modelPy, PyReverseDims(rec_coords'), PyReverseDims(dIn'), u=u0,
                              space_order=options.space_order, nb=model.nb, isic=options.isic)
            end

            grad = remove_padding(grad,model.nb,true_adjoint=options.sum_padding)
            if options.limit_m == true
                grad = extend_gradient(model_full,model,grad)
            end
            return vec(grad)
        end
    end
end

# Setup time-domain linear or nonlinear foward and adjoint modeling and interface to OPESCI/devito
function time_modeling(model_full::Model_TTI, srcGeometry::Geometry, srcData, recGeometry::Geometry, recData, dm, srcnum::Int64, op::Char, mode::Int64, options)

    # Load full geometry for out-of-core geometry containers
    typeof(recGeometry) == GeometryOOC && (recGeometry = Geometry(recGeometry))
    typeof(srcGeometry) == GeometryOOC && (srcGeometry = Geometry(srcGeometry))
    length(model_full.n) == 3 ? dims = (3,2,1) : dims = (2,1)   # model dimensions for Python are (z,y,x) and (z,x)

    # for 3D modeling, limit model to area with sources/receivers
    if options.limit_m == true
        model = deepcopy(model_full)
        if op=='J' && mode==1
            model,dm = limit_model_to_receiver_area(srcGeometry,recGeometry,model,options.buffer_size;pert=dm)
        else
            model = limit_model_to_receiver_area(srcGeometry,recGeometry,model,options.buffer_size)
        end
    else
        model = model_full
    end

    # Set up Python model structure
    if op=='J' && mode == 1
        # Set up Python model structure (force origin to be zero due to current devito bug)
        modelPy = pm.Model(origin=(0.,0.,0.), spacing=model.d, shape=model.n, vp=process_physical_parameter(sqrt.(1f0./model.m), dims),
                           epsilon=process_physical_parameter(model.epsilon, dims),
                           delta=process_physical_parameter(model.delta, dims),
                           theta=process_physical_parameter(model.theta, dims),
                           phi=process_physical_parameter(model.phi, dims), nbpml=model.nb,
                           dm=process_physical_parameter(reshape(dm,model.n), dims),
                           space_order=12)
    else
        # Set up Python model structure (force origin to be zero due to current devito bug)
        modelPy = pm.Model(origin=(0., 0., 0.), spacing=model.d, shape=model.n, vp=process_physical_parameter(sqrt.(1f0./model.m), dims),
                           epsilon=process_physical_parameter(model.epsilon, dims),
                           delta=process_physical_parameter(model.delta, dims),
                           theta=process_physical_parameter(model.theta, dims),
                           phi=process_physical_parameter(model.phi, dims), nbpml=model.nb,
                           space_order=12)
    end
    dtComp = modelPy[:critical_dt]

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
            dOut = pycall(tti.forward_modeling, PyObject, modelPy, PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'),
                          space_order=options.space_order, nb=model.nb)[1]
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
            qOut = pycall(tti.adjoint_modeling, PyObject, modelPy, PyReverseDims(src_coords'), PyReverseDims(rec_coords'), PyReverseDims(dIn'),
                                space_order=options.space_order, nb=model.nb)[1]
            ntSrc > ntComp && (qOut = [qOut zeros(size(qOut), ntSrc - ntComp)])
            qOut = time_resample(qOut,dtComp,srcGeometry)
            return judiVector(srcGeometry,qOut)
        end
    elseif op=='J'
        if mode==1
            # forward linearized modeling
            #println("Linearized forward modeling (source no. ",srcnum,")")
            dOut = pycall(tti.forward_born, PyObject, modelPy, PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'),
                          space_order=options.space_order, nb=model.nb, isiciso=options.isic, h_sub_factor=options.h_sub)[1]
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
            if options.optimal_checkpointing == true
                op_F = pycall(tti.forward_modeling, PyObject, modelPy, PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'),
                              op_return=true, space_order=options.space_order, nb=model.nb)
                grad = pycall(tti.adjoint_born, Array{Float32, length(model.n)}, modelPy, PyReverseDims(rec_coords'), PyReverseDims(dIn'), op_forward=op_F,
                              space_order=options.space_order, nb=model.nb, is_residual=true, isiciso=options.isic)
            else
                d, u0, v0 = pycall(tti.forward_modeling, PyObject, modelPy, PyReverseDims(src_coords'), PyReverseDims(qIn'), PyReverseDims(rec_coords'),
                                   space_order=options.space_order, nb=model.nb, save=true, t_sub_factor=options.t_sub, h_sub_factor=options.h_sub)
                grad = pycall(tti.adjoint_born, Array{Float32, length(model.n)}, modelPy, PyReverseDims(rec_coords'), PyReverseDims(dIn'), u=u0, v=v0,
                              space_order=options.space_order, nb=model.nb, isiciso=options.isic)
            end
            grad = remove_padding(grad,model.nb,true_adjoint=options.sum_padding)
            if options.limit_m == true
                grad = extend_gradient(model_full,model,grad)
            end
            return vec(grad)
        end
    end
end

# Function instance without options
time_modeling(model::Modelall, srcGeometry::Geometry, srcData, recGeometry::Geometry, recData, perturbation, srcnum::Int64, op::Char, mode::Int64) =
    time_modeling(model, srcGeometry, srcData, recGeometry, recData, perturbation, srcnum, op, mode, Options())
