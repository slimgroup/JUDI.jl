# pysource package

## Submodules

## FD_utils module


### FD_utils.angles_to_trig(model)
Tile and asymut angles trigonometric functions


### FD_utils.gx(field, model)
Rotated first derivative in x
:param u: TTI field
:type u: TimeFunction or Expr
:param model: Model structure
:type model: Model


* **Returns**

    du/dx in rotated coordinates



* **Return type**

    Expr



### FD_utils.gx_T(field, model)
Rotated first derivative in x


* **Parameters**

    
    * **u** (*TimeFunction** or **Expr*) – TTI field


    * **model** (*Model*) – Model structure



* **Returns**

    du/dx.T in rotated coordinates



* **Return type**

    Expr



### FD_utils.gy(field, model)
Rotated first derivative in y


* **Parameters**

    
    * **u** (*TimeFunction** or **Expr*) – TTI field


    * **model** (*Model*) – Model structure



* **Returns**

    du/dy in rotated coordinates



* **Return type**

    Expr



### FD_utils.gy_T(field, model)
Rotated first derivative in y


* **Parameters**

    
    * **u** (*TimeFunction** or **Expr*) – TTI field


    * **model** (*Model*) – Model structure



* **Returns**

    du/dy.T in rotated coordinates



* **Return type**

    Expr



### FD_utils.gz(field, model)
Rotated first derivative in z


* **Parameters**

    
    * **u** (*TimeFunction** or **Expr*) – TTI field


    * **model** (*Model*) – Model structure



* **Returns**

    du/dz in rotated coordinates



* **Return type**

    Expr



### FD_utils.gz_T(field, model)
Rotated first derivative in z


* **Parameters**

    
    * **u** (*TimeFunction** or **Expr*) – TTI field


    * **model** (*Model*) – Model structure



* **Returns**

    du/dz.T in rotated coordinates



* **Return type**

    Expr



### FD_utils.laplacian(v, irho)
Laplacian with density div( 1/rho grad) (u)


### FD_utils.ssa_1(u, v, model)
First row of
gx_t(A \* gx(P)) + gy_t( A1 \* gy(P)) + gz_T( A2 \* gz(P))


### FD_utils.ssa_2(u, v, model)
Second row of
gx_t(A \* gx(P)) + gy_t( A1 \* gy(P)) + gz_T( A2 \* gz(P))


### FD_utils.ssa_tti(u, v, model)
TTI finite difference kernel.


* **Parameters**

    
    * **u** (*TimeFunction*) – first TTI field


    * **v** (*TimeFunction*) – second TTI field


    * **model** (*Model*) – Model structure


## checkpoint module


### class checkpoint.CheckpointOperator(op, \*\*kwargs)
Devito’s concrete implementation of the ABC pyrevolve.Operator. This class wraps
devito.Operator so it conforms to the pyRevolve API. pyRevolve will call apply
with arguments t_start and t_end. Devito calls these arguments t_s and t_e so
the following dict is used to perform the translations between different names.


* **Parameters**

    
    * **op** (*Operator*) – devito.Operator object that this object will wrap.


    * **args** (*dict*) – If devito.Operator.apply() expects any arguments, they can be provided
    here to be cached. Any calls to CheckpointOperator.apply() will
    automatically include these cached arguments in the call to the
    underlying devito.Operator.apply().



#### apply(t_start, t_end)
If the devito operator requires some extra arguments in the call to apply
they can be stored in the args property of this object so pyRevolve calls
pyRevolve.Operator.apply() without caring about these extra arguments while
this method passes them on correctly to devito.Operator


#### t_arg_names( = {'t_end': 'time_M', 't_start': 'time_m'})

### class checkpoint.DevitoCheckpoint(objects)
Devito’s concrete implementation of the Checkpoint abstract base class provided by
pyRevolve. Holds a list of symbol objects that hold data.


#### property dtype()
data type


#### get_data(timestep)
returns the data (wavefield) for the time-step timestep


#### get_data_location(timestep)
returns the data (wavefield) for the time-step timestep


#### load()
NotImplementedError


#### save()
NotImplementedError


#### property size()
The memory consumption of the data contained in a checkpoint.


### checkpoint.get_symbol_data(symbol, timestep)
Return the symbol corresponding to the data at time-step timestep

## geom_utils module


### class geom_utils.AcquisitionGeometry(model, rec_positions, src_positions, t0, tn, \*\*kwargs)
Encapsulate the geometry of an acquisition:
- receiver positions and number
- source positions and number
In practice this would only point to a segy file with the
necessary information


#### property dt()
Time step value


#### property dtype()
Data type of the array (shot record, …)


#### property f0()
Peak frequency of the source


#### property grid()
Grid associated with the geometry


#### property model()
Model associated with the geometry


#### property nrec()
Number ov receivers


#### property nsrc()
Number of (simultenaous) sources


#### property nt()
Number of time sample equal to (self.tn - self.t0)/self.dt


#### property rec()
Construct a new Receiver object from the geometry informations


#### resample(dt)
Resamples in time at input dt


#### property src()
Construct a new PointSource object from the geometry informations


#### property src_type()
Type of source


#### property t0()
Start time of the source/receiver


#### property time_axis()
Time Axis


#### property tn()
End time of the source/receiver


### geom_utils.src_rec(model, u, src_coords=None, rec_coords=None, wavelet=None, fw=True, nt=None)
Generates the source injection and receiver interpolation.
This function is fully abstracted and does not care whether this is a
forward or adjoint wave-equation.
The source is the source term of the equation
The receiver is the measurment term

Therefore, for the adjoint, this function has to be called as:
src_rec(model, v, src_coords=rec_coords, …)
because the data is the sources


* **Parameters**

    
    * **model** (*Model*) – Physical model


    * **u** (*TimeFunction** or **tuple*) – Wavefield to inject into and read from


    * **src_coords** (*Array*) – Physical coordinates of the sources


    * **rec_coords** (*Array*) – Physical coordinates of the receivers


    * **wavelet** (*Array*) – Data for the source


    * **fw=True** – Whether the direction is forward or backward in time


    * **nt** (*int*) – Number of time steps


## interface module


### interface.J_adjoint(model, src_coords, wavelet, rec_coords, recin, space_order=8, checkpointing=False, free_surface=False, n_checkpoints=None, maxmem=None, freq_list=[], dft_sub=None, isic=False, ws=None)
Jacobian (adjoint fo born modeling operator) operator on a shot record
as a source (i.e data residual). Supports three modes:
\* Checkpinting
\* Frequency compression (on-the-fly DFT)
\* Standard zero lag cross correlation over time


* **Parameters**

    
    * **model** (*Model*) – Physical model


    * **src_coords** (*Array*) – Coordiantes of the source(s)


    * **wavelet** (*Array*) – Source signature


    * **rec_coords** (*Array*) – Coordiantes of the receiver(s)


    * **recin** (*Array*) – Receiver data


    * **space_order** (*Int** (**optional**)*) – Spatial discretization order, defaults to 8


    * **free_surface** (*Bool** (**optional**)*) – Whether or not to use a free surface


    * **checkpointing** (*Bool*) – Whether or not to use checkpointing


    * **n_checkpoints** (*Int*) – Number of checkpoints for checkpointing


    * **maxmem** (*Float*) – Maximum memory to use for checkpointing


    * **freq_list** (*List*) – List of frequencies for on-the-fly DFT


    * **dft_sub** (*Int*) – Subsampling factor for on-the-fly DFT


    * **isic** (*Bool*) – Whether or not to use ISIC imaging condition


    * **ws** (*Array*) – Extended source spatial distribution



* **Returns**

    Adjoint jacobian on the input data (gradient)



* **Return type**

    Array



### interface.J_adjoint_checkpointing(model, src_coords, wavelet, rec_coords, recin, space_order=8, free_surface=False, is_residual=False, n_checkpoints=None, maxmem=None, return_obj=False, isic=False, ws=None)
Jacobian (adjoint fo born modeling operator) operator on a shot record
as a source (i.e data residual). Outputs the gradient with Checkpointing.


* **Parameters**

    
    * **model** (*Model*) – Physical model


    * **src_coords** (*Array*) – Coordiantes of the source(s)


    * **wavelet** (*Array*) – Source signature


    * **rec_coords** (*Array*) – Coordiantes of the receiver(s)


    * **recin** (*Array*) – Receiver data


    * **space_order** (*Int** (**optional**)*) – Spatial discretization order, defaults to 8


    * **free_surface** (*Bool** (**optional**)*) – Whether or not to use a free surface


    * **checkpointing** (*Bool*) – Whether or not to use checkpointing


    * **n_checkpoints** (*Int*) – Number of checkpoints for checkpointing


    * **maxmem** (*Float*) – Maximum memory to use for checkpointing


    * **isic** (*Bool*) – Whether or not to use ISIC imaging condition


    * **ws** (*Array*) – Extended source spatial distribution


    * **is_residual** (*Bool*) – Whether to treat the input as the residual or as the observed data



* **Returns**

    Adjoint jacobian on the input data (gradient)



* **Return type**

    Array



### interface.J_adjoint_freq(model, src_coords, wavelet, rec_coords, recin, space_order=8, free_surface=False, freq_list=[], is_residual=False, return_obj=False, dft_sub=None, isic=False, ws=None)
Jacobian (adjoint fo born modeling operator) operator on a shot record
as a source (i.e data residual). Outputs the gradient with Frequency
compression (on-the-fly DFT).


* **Parameters**

    
    * **model** (*Model*) – Physical model


    * **src_coords** (*Array*) – Coordiantes of the source(s)


    * **wavelet** (*Array*) – Source signature


    * **rec_coords** (*Array*) – Coordiantes of the receiver(s)


    * **recin** (*Array*) – Receiver data


    * **space_order** (*Int** (**optional**)*) – Spatial discretization order, defaults to 8


    * **free_surface** (*Bool** (**optional**)*) – Whether or not to use a free surface


    * **freq_list** (*List*) – List of frequencies for on-the-fly DFT


    * **dft_sub** (*Int*) – Subsampling factor for on-the-fly DFT


    * **isic** (*Bool*) – Whether or not to use ISIC imaging condition


    * **ws** (*Array*) – Extended source spatial distribution


    * **is_residual** (*Bool*) – Whether to treat the input as the residual or as the observed data



* **Returns**

    Adjoint jacobian on the input data (gradient)



* **Return type**

    Array



### interface.J_adjoint_standard(model, src_coords, wavelet, rec_coords, recin, space_order=8, free_surface=False, is_residual=False, return_obj=False, isic=False, ws=None)
Adjoint Jacobian (adjoint fo born modeling operator) operator on a shot record
as a source (i.e data residual). Outputs the gradient with standard
zero lag cross correlation over time.


* **Parameters**

    
    * **model** (*Model*) – Physical model


    * **src_coords** (*Array*) – Coordiantes of the source(s)


    * **wavelet** (*Array*) – Source signature


    * **rec_coords** (*Array*) – Coordiantes of the receiver(s)


    * **recin** (*Array*) – Receiver data


    * **space_order** (*Int** (**optional**)*) – Spatial discretization order, defaults to 8


    * **free_surface** (*Bool** (**optional**)*) – Whether or not to use a free surface


    * **isic** (*Bool*) – Whether or not to use ISIC imaging condition


    * **ws** (*Array*) – Extended source spatial distribution


    * **is_residual** (*Bool*) – Whether to treat the input as the residual or as the observed data



* **Returns**

    Adjoint jacobian on the input data (gradient)



* **Return type**

    Array



### interface.adjoint_no_rec(model, rec_coords, data, space_order=8, free_surface=False)
Adjoint/backward modeling of a shot record (receivers as source)
without source sampling.


* **Parameters**

    
    * **model** (*Model*) – Physical model


    * **rec_coords** (*Array*) – Coordiantes of the receiver(s)


    * **data** (*Array*) – Shot gather


    * **space_order** (*Int** (**optional**)*) – Spatial discretization order, defaults to 8


    * **free_surface** (*Bool** (**optional**)*) – Whether or not to use a free surface



* **Returns**

    Adjoint wavefield



* **Return type**

    Array



### interface.adjoint_rec(model, src_coords, rec_coords, data, space_order=8, free_surface=False)
Adjoint/backward modeling of a shot record (receivers as source).


* **Parameters**

    
    * **model** (*Model*) – Physical model


    * **src_coords** (*Array*) – Coordiantes of the source(s)


    * **rec_coords** (*Array*) – Coordiantes of the receiver(s)


    * **data** (*Array*) – Shot gather


    * **space_order** (*Int** (**optional**)*) – Spatial discretization order, defaults to 8


    * **free_surface** (*Bool** (**optional**)*) – Whether or not to use a free surface



* **Returns**

    Shot record (adjoint wavefield at source position(s))



* **Return type**

    Array



### interface.adjoint_w(model, rec_coords, data, wavelet, space_order=8, free_surface=False)
Adjoint/backward modeling of a shot record (receivers as source) for an
extended source setup


* **Parameters**

    
    * **model** (*Model*) – Physical model


    * **rec_coords** (*Array*) – Coordiantes of the receiver(s)


    * **data** (*Array*) – Shot gather


    * **wavelet** (*Array*) – Time signature of the forward source for stacking along time


    * **space_order** (*Int** (**optional**)*) – Spatial discretization order, defaults to 8


    * **free_surface** (*Bool** (**optional**)*) – Whether or not to use a free surface



* **Returns**

    spatial distribution



* **Return type**

    Array



### interface.adjoint_wf_src(model, u, src_coords, space_order=8, free_surface=False)
Adjoint/backward modeling of a full wavefield (full wavefield as adjoint source).


* **Parameters**

    
    * **model** (*Model*) – Physical model


    * **u** (*Array** or **TimeFunction*) – Time-space dependent source


    * **src_coords** (*Array*) – Source coordinates


    * **space_order** (*Int** (**optional**)*) – Spatial discretization order, defaults to 8


    * **free_surface** (*Bool** (**optional**)*) – Whether or not to use a free surface



* **Returns**

    Shot record (sampled at source position(s))



* **Return type**

    Array



### interface.adjoint_wf_src_norec(model, u, space_order=8, free_surface=False)
Adjoint/backward modeling of a full wavefield (full wavefield as adjoint source).


* **Parameters**

    
    * **model** (*Model*) – Physical model


    * **u** (*Array** or **TimeFunction*) – Time-space dependent source


    * **space_order** (*Int** (**optional**)*) – Spatial discretization order, defaults to 8


    * **free_surface** (*Bool** (**optional**)*) – Whether or not to use a free surface



* **Returns**

    Adjoint wavefield



* **Return type**

    Array



### interface.born_rec(model, src_coords, wavelet, rec_coords, space_order=8, free_surface=False, isic=False)
Linearized (Born) modeling of a point source for a model perturbation
(square slowness) dm.


* **Parameters**

    
    * **model** (*Model*) – Physical model


    * **src_coords** (*Array*) – Coordiantes of the source(s)


    * **wavelet** (*Array*) – Source signature


    * **rec_coords** (*Array*) – Coordiantes of the receiver(s)


    * **space_order** (*Int** (**optional**)*) – Spatial discretization order, defaults to 8


    * **free_surface** (*Bool** (**optional**)*) – Whether or not to use a free surface


    * **isic** (*Bool*) – Whether or not to use ISIC imaging condition



* **Returns**

    Shot record



* **Return type**

    Array



### interface.born_rec_w(model, weight, wavelet, rec_coords, space_order=8, free_surface=False, isic=False)
Linearized (Born) modeling of a point source for a model
perturbation (square slowness) dm with an extended source


* **Parameters**

    
    * **model** (*Model*) – Physical model


    * **weight** (*Array*) – Spatial distriubtion of the extended source


    * **wavelet** (*Array*) – Source signature


    * **rec_coords** (*Array*) – Coordiantes of the receiver(s)


    * **space_order** (*Int** (**optional**)*) – Spatial discretization order, defaults to 8


    * **free_surface** (*Bool** (**optional**)*) – Whether or not to use a free surface


    * **isic** (*Bool*) – Whether or not to use ISIC imaging condition



* **Returns**

    Shot record



* **Return type**

    Array



### interface.forward_no_rec(model, src_coords, wavelet, space_order=8, free_surface=False)
Forward modeling of a point source without receiver.


* **Parameters**

    
    * **model** (*Model*) – Physical model


    * **src_coords** (*Array*) – Coordiantes of the source(s)


    * **wavelet** (*Array*) – Source signature


    * **space_order** (*Int** (**optional**)*) – Spatial discretization order, defaults to 8


    * **free_surface** (*Bool** (**optional**)*) – Whether or not to use a free surface



* **Returns**

    Wavefield



* **Return type**

    Array



### interface.forward_rec(model, src_coords, wavelet, rec_coords, space_order=8, free_surface=False)
Forward modeling of a point source with receivers.


* **Parameters**

    
    * **model** (*Model*) – Physical model


    * **src_coords** (*Array*) – Coordiantes of the source(s)


    * **wavelet** (*Array*) – Source signature


    * **rec_coords** (*Array*) – Coordiantes of the receiver(s)


    * **space_order** (*Int** (**optional**)*) – Spatial discretization order, defaults to 8


    * **free_surface** (*Bool** (**optional**)*) – Whether or not to use a free surface



* **Returns**

    Shot record



* **Return type**

    Array



### interface.forward_rec_w(model, weight, wavelet, rec_coords, space_order=8, free_surface=False)
Forward modeling of an extended source with receivers.


* **Parameters**

    
    * **model** (*Model*) – Physical model


    * **weights** (*Array*) – Spatial distribution of the extended source.


    * **wavelet** (*Array*) – Source signature


    * **rec_coords** (*Array*) – Coordiantes of the receiver(s)


    * **space_order** (*Int** (**optional**)*) – Spatial discretization order, defaults to 8


    * **free_surface** (*Bool** (**optional**)*) – Whether or not to use a free surface



* **Returns**

    Shot record



* **Return type**

    Array



### interface.forward_rec_wf(model, src_coords, wavelet, rec_coords, space_order=8, free_surface=False)
Forward modeling of a point source.


* **Parameters**

    
    * **model** (*Model*) – Physical model


    * **src_coords** (*Array*) – Coordiantes of the source(s)


    * **wavelet** (*Array*) – Source signature


    * **rec_coords** (*Array*) – Coordiantes of the receiver(s)


    * **space_order** (*Int** (**optional**)*) – Spatial discretization order, defaults to 8


    * **free_surface** (*Bool** (**optional**)*) – Whether or not to use a free surface



* **Returns**

    
    * *Array* – Shot record


    * *TimeFunction* – Wavefield




### interface.forward_wf_src(model, u, rec_coords, space_order=8, free_surface=False)
Forward modeling of a full wavefield source.


* **Parameters**

    
    * **model** (*Model*) – Physical model


    * **u** (*TimeFunction** or **Array*) – Time-space dependent wavefield


    * **rec_coords** (*Array*) – Coordiantes of the receiver(s)


    * **space_order** (*Int** (**optional**)*) – Spatial discretization order, defaults to 8


    * **free_surface** (*Bool** (**optional**)*) – Whether or not to use a free surface



* **Returns**

    Shot record



* **Return type**

    Array



### interface.forward_wf_src_norec(model, u, space_order=8, free_surface=False)
Forward modeling of a full wavefield source without receiver.


* **Parameters**

    
    * **model** (*Model*) – Physical model


    * **u** (*TimeFunction** or **Array*) – Time-space dependent wavefield


    * **space_order** (*Int** (**optional**)*) – Spatial discretization order, defaults to 8


    * **free_surface** (*Bool** (**optional**)*) – Whether or not to use a free surface



* **Returns**

    Wavefield



* **Return type**

    Array



### interface.grad_fwi(model, recin, rec_coords, u, space_order=8, free_surface=False)
FWI gradient, i.e adjoint Jacobian on a data residual.


* **Parameters**

    
    * **model** (*Model*) – Physical model


    * **recin** (*Array*) – Data residual


    * **rec_coords** (*Array*) – Receivers coordinates


    * **u** (*TimeFunction*) – Forward wavefield


    * **space_order** (*Int** (**optional**)*) – Spatial discretization order, defaults to 8


    * **free_surface** (*Bool** (**optional**)*) – Whether or not to use a free surface



* **Returns**

    FWI gradient



* **Return type**

    Array


## kernels module


### kernels.acoustic_kernel(model, u, fw=True, q=None)
Acoustic wave equation time stepper


* **Parameters**

    
    * **model** (*Model*) – Physical model


    * **u** (*TimeFunction** or **tuple*) – wavefield (tuple if TTI)


    * **fw** (*Bool*) – Whether forward or backward in time propagation


    * **q** (*TimeFunction** or **Expr*) – Full time-space source



### kernels.tti_kernel(model, u1, u2, fw=True, q=None)
TTI wave equation (one from my paper) time stepper


* **Parameters**

    
    * **model** (*Model*) – Physical model


    * **u1** (*TimeFunction*) – First component (pseudo-P) of the wavefield


    * **u2** (*TimeFunction*) – First component (pseudo-P) of the wavefield


    * **fw** (*Bool*) – Whether forward or backward in time propagation


    * **q** (*TimeFunction** or **Expr*) – Full time-space source as a tuple (one value for each component)



### kernels.wave_kernel(model, u, fw=True, q=None, fs=False)
Pde kernel corresponding the the model for the input wavefield


* **Parameters**

    
    * **model** (*Model*) – Physical model


    * **u** (*TimeFunction** or **tuple*) – wavefield (tuple if TTI)


    * **fw** (*Bool*) – Whether forward or backward in time propagation


    * **q** (*TimeFunction** or **Expr*) – Full time-space source


    * **fs** (*Bool*) – Freesurface flag


## models module


### class models.Model(origin, spacing, shape, vp, space_order=2, nbl=40, dtype=<class 'numpy.float32'>, epsilon=None, delta=None, theta=None, phi=None, rho=1, dm=None, subdomains=(), \*\*kwargs)
The physical model used in seismic inversion processes.


* **Parameters**

    
    * **origin** (*tuple of floats*) – Origin of the model in m as a tuple in (x,y,z) order.


    * **spacing** (*tuple of floats*) – Grid size in m as a Tuple in (x,y,z) order.


    * **shape** (*tuple of int*) – Number of grid points size in (x,y,z) order.


    * **space_order** (*int*) – Order of the spatial stencil discretisation.


    * **vp** (*array_like** or **float*) – Velocity in km/s.


    * **nbl** (*int**, **optional*) – The number of absorbin layers for boundary damping.


    * **dtype** (*np.float32** or **np.float64*) – Defaults to 32.


    * **epsilon** (*array_like** or **float**, **optional*) – Thomsen epsilon parameter (0<epsilon<1).


    * **delta** (*array_like** or **float*) – Thomsen delta parameter (0<delta<1), delta<epsilon.


    * **theta** (*array_like** or **float*) – Tilt angle in radian.


    * **phi** (*array_like** or **float*) – Asymuth angle in radian.


    * **dt** (*Float*) – User provided computational time-step



#### property critical_dt()
Critical computational time step value from the CFL condition.


#### property dm()
Model perturbation for linearized modeling


#### property dt()
User provided dt


#### property is_tti()
Whether the model is TTI or isotopic


#### property m()
Symbolic representation of the squared slowness
m = 1/vp^2


#### property space_order()
Spatial discretization order


#### property spacing_map()
Map between spacing symbols and their values for each SpaceDimension.


#### property vp()
Function holding the model velocity in km/s.

## propagators module


### propagators.adjoint(model, y, src_coords, rcv_coords, space_order=8, q=0, save=False, free_surface=False, ws=None)
Low level propagator, to be used through interface.py
Compute adjoint wavefield v = adjoint(F(m))\*y
and related quantities (||v||_w, v(xsrc))


### propagators.born(model, src_coords, rcv_coords, wavelet, space_order=8, save=False, free_surface=False, isic=False, ws=None)
Low level propagator, to be used through interface.py
Compute adjoint wavefield v = adjoint(F(m))\*y
and related quantities (||v||_w, v(xsrc))


### propagators.forward(model, src_coords, rcv_coords, wavelet, space_order=8, save=False, q=0, free_surface=False, return_op=False, freq_list=None, dft_sub=None, ws=None)
Low level propagator, to be used through interface.py
Compute forward wavefield u = A(m)^{-1}\*f and related quantities (u(xrcv))


### propagators.gradient(model, residual, rcv_coords, u, return_op=False, space_order=8, w=None, free_surface=False, freq=None, dft_sub=None, isic=True)
Low level propagator, to be used through interface.py
Compute adjoint wavefield v = adjoint(F(m))\*y
and related quantities (||v||_w, v(xsrc))


### propagators.name(model)

### propagators.op_kwargs(model, fs=False)
## sensitivity module


### sensitivity.basic_src(model, u, \*\*kwargs)
Basic source for linearized modeling


* **Parameters**

    
    * **u** (*TimeFunction** or **Tuple*) – Forward wavefield (tuple of fields for TTI or dft)


    * **model** (*Model*) – Model containing the perturbation dm



### sensitivity.corr_fields(u, v, model, \*\*kwargs)
Cross correlation of forward and adjoint wavefield


* **Parameters**

    
    * **u** (*TimeFunction** or **Tuple*) – Forward wavefield (tuple of fields for TTI or dft)


    * **v** (*TimeFunction** or **Tuple*) – Adjoint wavefield (tuple of fields for TTI)


    * **model** (*Model*) – Model structure



### sensitivity.corr_freq(u, v, model, freq=None, dft_sub=None, \*\*kwargs)
Standard cross-correlation imaging condition with on-th-fly-dft


* **Parameters**

    
    * **u** (*TimeFunction** or **Tuple*) – Forward wavefield (tuple of fields for TTI or dft)


    * **v** (*TimeFunction** or **Tuple*) – Adjoint wavefield (tuple of fields for TTI)


    * **model** (*Model*) – Model structure


    * **freq** (*Array*) – Array of frequencies for on-the-fly DFT


    * **factor** (*int*) – Subsampling factor for DFT



### sensitivity.func_name(freq=None, isic=False)
Get key for imaging condition/linearized source function


### sensitivity.grad_expr(gradm, u, v, model, w=1, freq=None, dft_sub=None, isic=False)
Gradient update stencil


* **Parameters**

    
    * **u** (*TimeFunction** or **Tuple*) – Forward wavefield (tuple of fields for TTI or dft)


    * **v** (*TimeFunction** or **Tuple*) – Adjoint wavefield (tuple of fields for TTI)


    * **model** (*Model*) – Model structure


    * **w** (*Float** or **Expr** (**optional**)*) – Weight for the gradient expression (default=1)


    * **freq** (*Array*) – Array of frequencies for on-the-fly DFT


    * **factor** (*int*) – Subsampling factor for DFT


    * **isic** (*Bool*) – Whether or not to use inverse scattering imaging condition (not supported yet)



### sensitivity.isic_freq_g(u, v, model, \*\*kwargs)
Inverse scattering imaging condition


* **Parameters**

    
    * **u** (*TimeFunction** or **Tuple*) – Forward wavefield (tuple of fields for TTI or dft)


    * **v** (*TimeFunction** or **Tuple*) – Adjoint wavefield (tuple of fields for TTI)


    * **model** (*Model*) – Model structure



### sensitivity.isic_g(u, v, model, \*\*kwargs)
Inverse scattering imaging condition


* **Parameters**

    
    * **u** (*TimeFunction** or **Tuple*) – Forward wavefield (tuple of fields for TTI or dft)


    * **v** (*TimeFunction** or **Tuple*) – Adjoint wavefield (tuple of fields for TTI)


    * **model** (*Model*) – Model structure



### sensitivity.isic_s(model, u, \*\*kwargs)
ISIC source for linearized modeling


* **Parameters**

    
    * **u** (*TimeFunction** or **Tuple*) – Forward wavefield (tuple of fields for TTI or dft)


    * **model** (*Model*) – Model containing the perturbation dm



### sensitivity.lin_src(model, u, isic=False)
Source for linearized modeling


* **Parameters**

    
    * **u** (*TimeFunction** or **Tuple*) – Forward wavefield (tuple of fields for TTI or dft)


    * **model** (*Model*) – Model containing the perturbation dm


## sources module


### class sources.PointSource(\*args, \*\*kwargs)
Symbolic data object for a set of sparse point sources


* **Parameters**

    
    * **name** (*String*) – Name of the symbol representing this source


    * **grid** (*Grid*) – Grid object defining the computational domain.


    * **coordinates** (*Array*) – Point coordinates for this source


    * **data** (*(**Optional**) **Data*) – values to initialise point data


    * **ntime** (*Int** (**Optional**)*) – Number of timesteps for which to allocate data


    * **npoint** (*Int** (**Optional**)*) – 


    * **of sparse points represented by this source** (*Number*) – 


    * **dimension** (*Dimension** (**Optional**)*) – object for representing the number of points in this source


    * **either the dimensions ntime and npoint**** or ****the fully** (*Note**,*) – 


    * **data array need to be provided.** (*initialised*) – 



#### default_assumptions( = {'commutative': True, 'complex': True, 'extended_real': True, 'finite': True, 'hermitian': True, 'imaginary': False, 'infinite': False, 'real': True})

#### is_commutative( = True)

#### is_complex( = True)

#### is_extended_real( = True)

#### is_finite( = True)

#### is_hermitian( = True)

#### is_imaginary( = False)

#### is_infinite( = False)

#### is_real( = True)

### sources.Receiver()
alias of `sources.PointSource`


### sources.Shot()
alias of `sources.PointSource`


### class sources.RickerSource(\*args, \*\*kwargs)
Symbolic object that encapsulate a set of sources with a
pre-defined Ricker wavelet:
[http://subsurfwiki.org/wiki/Ricker_wavelet](http://subsurfwiki.org/wiki/Ricker_wavelet)
name: Name for the resulting symbol
grid: `Grid` object defining the computational domain.
f0: Peak frequency for Ricker wavelet in kHz
time: Discretized values of time in ms


#### default_assumptions( = {'commutative': True, 'complex': True, 'extended_real': True, 'finite': True, 'hermitian': True, 'imaginary': False, 'infinite': False, 'real': True})

#### is_commutative( = True)

#### is_complex( = True)

#### is_extended_real( = True)

#### is_finite( = True)

#### is_hermitian( = True)

#### is_imaginary( = False)

#### is_infinite( = False)

#### is_real( = True)

#### wavelet(f0, t)
Defines a Ricker wavelet with a peak frequency f0 at time t.
f0: Peak frequency in kHz
t: Discretized values of time in ms


### class sources.GaborSource(\*args, \*\*kwargs)
Symbolic object that encapsulate a set of sources with a
pre-defined Gabor wavelet:
[https://en.wikipedia.org/wiki/Gabor_wavelet](https://en.wikipedia.org/wiki/Gabor_wavelet)
name: Name for the resulting symbol
grid: `Grid` object defining the computational domain.
f0: Peak frequency for Ricker wavelet in kHz
time: Discretized values of time in ms


#### default_assumptions( = {'commutative': True, 'complex': True, 'extended_real': True, 'finite': True, 'hermitian': True, 'imaginary': False, 'infinite': False, 'real': True})

#### is_commutative( = True)

#### is_complex( = True)

#### is_extended_real( = True)

#### is_finite( = True)

#### is_hermitian( = True)

#### is_imaginary( = False)

#### is_infinite( = False)

#### is_real( = True)

#### wavelet(f0, t)
Defines a Gabor wavelet with a peak frequency f0 at time t.
f0: Peak frequency in kHz
t: Discretized values of time in ms


### class sources.TimeAxis(start=None, step=None, num=None, stop=None)
Data object to store the TimeAxis. Exactly three of the four key arguments
must be prescribed. Because of remainder values it is not possible to create
a TimeAxis that exactly adhears to the inputs therefore start, stop, step
and num values should be taken from the TimeAxis object rather than relying
upon the input values.
The four possible cases are:
\* start is None: start = step\*(1 - num) + stop
\* step is None: step = (stop - start)/(num - 1)
\* num is None: num = ceil((stop - start + step)/step) and
because of remainder stop = step\*(num - 1) + start
\* stop is None: stop = step\*(num - 1) + start


* **Parameters**

    
    * **start** (*float**, **optional*) – Start of time axis.


    * **step** (*float**, **optional*) – Time interval.


    * **num** (*int**, **optional*) – Number of values (Note: this is the number of intervals + 1).
    Stop value is reset to correct for remainder.


    * **stop** (*float**, **optional*) – End time.



#### time_values()
## wave_utils module


### wave_utils.extended_src_weights(model, wavelet, v)
Adjoint of extended source. This function returns the expression to obtain
the spatially varrying weights from the wavefield and time-dependent wavelet


* **Parameters**

    
    * **model** (*Model*) – Physical model structure


    * **wavelet** (*Array*) – Time-serie for the time-varying source


    * **v** (*TimeFunction*) – Wavefield to get the weights from



### wave_utils.extented_src(model, weight, wavelet, q=0)
Extended source for modelling where the source is the outer product of
a spatially varying weight and a time-dependent wavelet i.e.:
u.dt2 - u.laplace = w(x)\*q(t)
This function returns the extended source w(x)\*q(t)


* **Parameters**

    
    * **model** (*Model*) – Physical model structure


    * **weight** (*Array*) – Array of weight for the spatial Function


    * **wavelet** (*Array*) – Time-serie for the time-varying source


    * **q** (*Symbol** or **Expr** (**optional**)*) – Previously existing source to be added to (source will be q +  w(x)\*q(t))



### wave_utils.freesurface(field, npml, forward=True)
Generate the stencil that mirrors the field as a free surface modeling for
the acoustic wave equation


* **Parameters**

    
    * **field** (*TimeFunction** or **Tuple*) – Field for which to add a free surface


    * **npml** (*int*) – Number of ABC points


    * **forward** (*Bool*) – Whether it is forward or backward propagation (in time)



### wave_utils.otf_dft(u, freq, dt, factor=None)
On the fly DFT wavefield (frequency slices) and expression


* **Parameters**

    
    * **u** (*TimeFunction** or **Tuple*) – Forward wavefield


    * **freq** (*Array*) – Array of frequencies for on-the-fly DFT


    * **factor** (*int*) – Subsampling factor for DFT



### wave_utils.sub_time(time, factor, dt=1, freq=None)
Subsampled  time axis


* **Parameters**

    
    * **time** (*Dimension*) – time Dimension


    * **factor** (*int*) – Subsampling factor



### wave_utils.wavefield(model, space_order, save=False, nt=None, fw=True, name='')
Create the wavefield for the wave equation


* **Parameters**

    
    * **model** (*Model*) – Physical model


    * **space_order** (*int*) – Spatial discretization order


    * **save** (*Bool*) – Whether or not to save the time history


    * **nt** (*int** (**optional**)*) – Number of time steps if the wavefield is saved


    * **fw** (*Bool*) – Forward or backward (for naming)


    * **name** (*string*) – Custom name attached to default (u+name)



### wave_utils.wf_as_src(v, w=1)
Weighted source as a time-space wavefield


* **Parameters**

    
    * **u** (*TimeFunction** or **Tuple*) – Forward wavefield (tuple of fields for TTI or dft)


    * **w** (*Float** or **Expr** (**optional**)*) – Weight for the source expression (default=1)


## Module contents
