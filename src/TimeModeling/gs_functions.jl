using ECOS

export gs_residual, gs_residual_trace, gs_residual_shot

function gs_residual_trace(maxshift, dtComp, d1_in::Array{Float32, 2}, d2_in::Array{Float32, 2}, normalized)
	#shifts
	nshift = round(Int64, maxshift/dtComp)
	nSamples = 4*nshift + 1
	nSQP = 2*nshift + 1
	# Initialze tmp for speed
	adj_src = similar(d1_in)
	nt, nrec = size(d1_in)
	global obs = zeros(Float32, nt+nSamples-1)
	global syn = zeros(Float32, nt+nSamples-1)
	global aux = similar(obs)
	global center = zeros(Float32, 4*nshift + 1)
	global H = zeros(nSQP, nSQP)
	# QP setup
	# get coefficientsn
	x = Convex.Variable(nSQP)
	A = Variable(nSQP, nSQP)
	A.value = zeros(nSQP, nSQP)
	p = Convex.minimize(Convex.quadform(x, A.value))
	un = ones(1, nSQP)
	p.constraints += un * x == 1
	p.constraints += [x >= 0; x <= 1]

	if normalized == "shot"
		d1 = d1_in/norm(d1_in)
		d2 = d1_in/norm(d2_in)
	else
		d1 = view(d1_in, :, :)
		d2 = view(d2_in, :, :)
	end

    indnz = [i for i in 1:size(d1, 2) if (norm(d2[:,i])>0 && norm(d1[:,i])>0)]

	for (i, rr) in enumerate(indnz)
		# start1 = time()
		# println("Trace ", i, " out of ", nrec)
		global syn[2*nshift+1:end-2*nshift] .= d1[:, rr]
		global obs[2*nshift+1:end-2*nshift] .= d2[:, rr]

		normalized == "trace" ? weight = norm(obs) : weight = 1
		if normalized == "trace"
			global syn ./= norm(syn)
			global obs ./= norm(obs)
		end

		@inbounds for i = 1:(4*nshift + 1)
			circshift!(aux, syn, i - 2*nshift - 1)
			broadcast!(-, aux, aux, obs)
			global center[i] = dot(aux, syn - obs)/norm(aux)
		end
		center[center .< 1f-5] .= 1f-5
		@inbounds for i = 1:2*nshift + 1
			start = 2*nshift + 1 - i + 1
			lastind = 2*nshift + 1 + start - 1
			global H[i, :] .+= .5f0 .* center[start:lastind]
		end
		global H .+= H'
		# Make posef, bit costly though
		if ~isposdef(H)
			v = eigvals(H)[1]
			println("Non positive definite matrix, adding ", real(v)," I to make it posdef")
			global H .+= (-1.01*real(v))*Matrix(I, nSQP, nSQP)
		end

		A.value = H
		fix!(A)
		solve!(p, ECOS.Optimizer(verbose=0, max_iters=100))
		alphas = Array{Float32}(x.value)
		# Data misfit
		for i = 1:2*nshift+1
			shift = i - nshift - 1
			adj_src[:, rr] += alphas[i] * circshift(circshift(syn, shift) - obs, -2*shift)[2*nshift+1:(end-2*nshift)]
		end
        adj_src[:, rr] = weight * adj_src[:, rr]
		# println("Trace time ", time() - start1)
	end
	return adj_src
end

function gs_residual_shot(maxshift, dtComp, d1_in::Array{Float32, 2}, d2_in::Array{Float32, 2}, normalized)
	#shifts
	nshift = round(Int64, maxshift/dtComp)
	# println(nshift, " ", dtComp)
	nSQP = 2*nshift + 1

	adj_src = similar(d1_in)
	nt, nrec = size(d1_in)

	d1 = [zeros(Float32, 2*nshift, nrec); d1_in; zeros(Float32, 2*nshift, nrec)]
	d2 = [zeros(Float32, 2*nshift, nrec); d2_in; zeros(Float32, 2*nshift, nrec)]
	if normalized == "shot"
		d1 /= norm(vec(d1))
		d2 /= norm(vec(d2))
	elseif normalized == "trace"
		for i = 1:size(d1,2)
			norm(d1[:, i]) > 0 ? n1 = norm(d1[:, i]) : n1 = 1
			norm(d2[:, i]) > 0 ? n2 = norm(d2[:, i]) : n2 = 1
			d1[:, i] /= n1
			d2[:, i] /= n1
		end
	end
	aux = zeros(Float32, size(d1))

	# QP setup
	global H = zeros(nSQP, nSQP)

	# Build H
	center = zeros(Float32, 4*nshift + 1)
	for i = 1:(4*nshift + 1)
		circshift!(aux, d1, (i - 2*nshift - 1, 0))
		broadcast!(-, aux, aux, d2)
		global center[i] = dot(aux, d1 - d2)/norm(aux)
	end
	center[center .< 1f-5] .= 1f-5

	for i = 1:nSQP
		start = 2*nshift + 1 - i + 1
		lastind = 2*nshift + 1 + start - 1
		H[i, :] .+= .5f0 .* center[start:lastind]
	end
	H .+= H'
	# Make posef, bit costly though
	if ~isposdef(H)
		v = eigvals(H)[1]
		println("Non positive definite matrix, adding ", real(v)," I to make it posdef")
		H .+= (-1.01*real(v))*Matrix(I, nSQP, nSQP)
	end


	# get coefficientsn
	x = Convex.Variable(nSQP)
	p = Convex.minimize(Convex.quadform(x, H))
	un = ones(1, nSQP)
	p.constraints += un * x == 1
	p.constraints += [x >= 0; x <= 1]

	solve!(p, ECOS.Optimizer(verbose=0, max_iters=100))

	alphas = Array{Float32}(x.value)
	# Data misfit
	for i = 1:nSQP
		shift = i - nshift - 1
		adj_src += alphas[i] * circshift(abs.(circshift(d1, (shift, 0))).*(circshift(d1, (shift, 0)) - d2), (-2*shift, 0))[2*nshift+1:(end-2*nshift), :]
	end

	return adj_src
end

function gs_residual(gs::Dict, dtComp, d1::Array{Float32, 2}, d2::Array{Float32, 2}, normalized)
	if gs["strategy"] == "shot"
		adj_src = gs_residual_shot(gs["maxshift"], dtComp, d1, d2, normalized)
	else
		adj_src = gs_residual_trace(gs["maxshift"], dtComp, d1, d2, normalized)
	end
	return adj_src
end
