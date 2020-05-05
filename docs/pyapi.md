# Python API

We briefly describe python API JUDI relies on forth e wave-equation solves. The python code is in the subfolder `src/pysource` and is organized so that changing wave-euation is easy and code reusability is as god as possible.

## checkpoint

Interface with PyRevolve for checkpointing


### CheckpointOperator

```python
class CheckpointOperator(Operator)
```

Devito's concrete implementation of the ABC pyrevolve.Operator. This class wraps
devito.Operator so it conforms to the pyRevolve API. pyRevolve will call apply
with arguments t_start and t_end. Devito calls these arguments t_s and t_e so
the following dict is used to perform the translations between different names.

**Parameters:**


op : Operator
devito.Operator object that this object will wrap.
args : dict
If devito.Operator.apply() expects any arguments, they can be provided
here to be cached. Any calls to CheckpointOperator.apply() will
automatically include these cached arguments in the call to the
underlying devito.Operator.apply().

<a name=".pysource.checkpoint.CheckpointOperator.t_arg_names"></a>
#### t\_arg\_names

<a name=".pysource.checkpoint.CheckpointOperator.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(op, **kwargs)
```

<a name=".pysource.checkpoint.CheckpointOperator._prepare_args"></a>
#### \_prepare\_args

```python
 | _prepare_args(t_start, t_end)
```

<a name=".pysource.checkpoint.CheckpointOperator.apply"></a>
#### apply

```python
 | apply(t_start, t_end)
```

If the devito operator requires some extra arguments in the call to apply
they can be stored in the args property of this object so pyRevolve calls
pyRevolve.Operator.apply() without caring about these extra arguments while
this method passes them on correctly to devito.Operator

<a name=".pysource.checkpoint.DevitoCheckpoint"></a>
### DevitoCheckpoint

```python
class DevitoCheckpoint(Checkpoint)
```

Devito's concrete implementation of the Checkpoint abstract base class provided by
pyRevolve. Holds a list of symbol objects that hold data.

<a name=".pysource.checkpoint.DevitoCheckpoint.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(objects)
```

Intialise a checkpoint object. Upon initialisation, a checkpoint
stores only a reference to the objects that are passed into it.

<a name=".pysource.checkpoint.DevitoCheckpoint.dtype"></a>
#### dtype

```python
 | @property
 | dtype()
```

<a name=".pysource.checkpoint.DevitoCheckpoint.get_data"></a>
#### get\_data

```python
 | get_data(timestep)
```

<a name=".pysource.checkpoint.DevitoCheckpoint.get_data_location"></a>
#### get\_data\_location

```python
 | get_data_location(timestep)
```

<a name=".pysource.checkpoint.DevitoCheckpoint.size"></a>
#### size

```python
 | @property
 | size()
```

The memory consumption of the data contained in a checkpoint.

<a name=".pysource.checkpoint.DevitoCheckpoint.save"></a>
#### save

```python
 | save(*args)
```

<a name=".pysource.checkpoint.DevitoCheckpoint.load"></a>
#### load

```python
 | load(*args)
```

<a name=".pysource.checkpoint.get_symbol_data"></a>
#### get\_symbol\_data

```python
get_symbol_data(symbol, timestep)
```

<a name=".pysource.propagators"></a>
## pysource.propagators

<a name=".pysource.propagators.name"></a>
#### name

```python
name(model)
```

<a name=".pysource.propagators.op_kwargs"></a>
#### op\_kwargs

```python
op_kwargs(model, fs=False)
```

<a name=".pysource.propagators.forward"></a>
#### forward

```python
forward(model, src_coords, rcv_coords, wavelet, space_order=8, save=False, q=0, free_surface=False, return_op=False, freq_list=None, dft_sub=None, ws=None)
```

Compute forward wavefield u = A(m)^{-1}*f and related quantities (u(xrcv))

<a name=".pysource.propagators.adjoint"></a>
#### adjoint

```python
adjoint(model, y, src_coords, rcv_coords, space_order=8, q=0, save=False, free_surface=False, ws=None)
```

Compute adjoint wavefield v = adjoint(F(m))*y
and related quantities (||v||_w, v(xsrc))

<a name=".pysource.propagators.gradient"></a>
#### gradient

```python
gradient(model, residual, rcv_coords, u, return_op=False, space_order=8, w=None, free_surface=False, freq=None, dft_sub=None, isic=True)
```

Compute adjoint wavefield v = adjoint(F(m))*y
and related quantities (||v||_w, v(xsrc))

<a name=".pysource.propagators.born"></a>
#### born

```python
born(model, src_coords, rcv_coords, wavelet, space_order=8, save=False, free_surface=False, isic=False, ws=None)
```

Compute adjoint wavefield v = adjoint(F(m))*y
and related quantities (||v||_w, v(xsrc))

<a name=".pysource.models"></a>
## pysource.models

<a name=".pysource.models.__all__"></a>
#### \_\_all\_\_

<a name=".pysource.models.PhysicalDomain"></a>
### PhysicalDomain

```python
class PhysicalDomain(SubDomain)
```

<a name=".pysource.models.PhysicalDomain.name"></a>
#### name

<a name=".pysource.models.PhysicalDomain.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(nbl)
```

<a name=".pysource.models.PhysicalDomain.define"></a>
#### define

```python
 | define(dimensions)
```

<a name=".pysource.models.initialize_damp"></a>
#### initialize\_damp

```python
initialize_damp(damp, nbl, spacing, mask=False)
```

Initialise damping field with an absorbing boundary layer.


**Parameters:**


damp : Function
The damping field for absorbing boundary condition.
nbl : int
Number of points in the damping layer.
spacing :
Grid spacing coefficient.
mask : bool, optional
whether the dampening is a mask or layer.
mask => 1 inside the domain and decreases in the layer
not mask => 0 inside the domain and increase in the layer

<a name=".pysource.models.GenericModel"></a>
### GenericModel

```python
class GenericModel(object)
```

General model class with common properties

<a name=".pysource.models.GenericModel.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(origin, spacing, shape, space_order, nbl=20, dtype=np.float32, subdomains=(), damp_mask=True)
```

<a name=".pysource.models.GenericModel.physical_params"></a>
#### physical\_params

```python
 | physical_params(**kwargs)
```

Return all set physical parameters and update to input values if provided

<a name=".pysource.models.GenericModel._gen_phys_param"></a>
#### \_gen\_phys\_param

```python
 | _gen_phys_param(field, name, space_order, is_param=False, default_value=0, func=lambda x: x)
```

<a name=".pysource.models.GenericModel.physical_parameters"></a>
#### physical\_parameters

```python
 | @property
 | physical_parameters()
```

<a name=".pysource.models.GenericModel.dim"></a>
#### dim

```python
 | @property
 | dim()
```

Spatial dimension of the problem and model domain.

<a name=".pysource.models.GenericModel.spacing"></a>
#### spacing

```python
 | @property
 | spacing()
```

Grid spacing for all fields in the physical model.

<a name=".pysource.models.GenericModel.space_dimensions"></a>
#### space\_dimensions

```python
 | @property
 | space_dimensions()
```

Spatial dimensions of the grid

<a name=".pysource.models.GenericModel.spacing_map"></a>
#### spacing\_map

```python
 | @property
 | spacing_map()
```

Map between spacing symbols and their values for each `SpaceDimension`.

<a name=".pysource.models.GenericModel.dtype"></a>
#### dtype

```python
 | @property
 | dtype()
```

Data type for all assocaited data objects.

<a name=".pysource.models.GenericModel.domain_size"></a>
#### domain\_size

```python
 | @property
 | domain_size()
```

Physical size of the domain as determined by shape and spacing

<a name=".pysource.models.Model"></a>
### Model

```python
class Model(GenericModel)
```

The physical model used in seismic inversion processes.


**Parameters:**


origin : tuple of floats
Origin of the model in m as a tuple in (x,y,z) order.
spacing : tuple of floats
Grid size in m as a Tuple in (x,y,z) order.
shape : tuple of int
Number of grid points size in (x,y,z) order.
space_order : int
Order of the spatial stencil discretisation.
vp : array_like or float
Velocity in km/s.
nbl : int, optional
The number of absorbin layers for boundary damping.
dtype : np.float32 or np.float64
Defaults to 32.
epsilon : array_like or float, optional
Thomsen epsilon parameter (0<epsilon<1).
delta : array_like or float
Thomsen delta parameter (0<delta<1), delta<epsilon.
theta : array_like or float
Tilt angle in radian.
phi : array_like or float
Asymuth angle in radian.

The `Model` provides two symbolic data objects for the
creation of seismic wave propagation operators:

m : array_like or float
The square slowness of the wave.
damp : Function
The damping field for absorbing boundary condition.

<a name=".pysource.models.Model.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(origin, spacing, shape, vp, space_order=2, nbl=40, dtype=np.float32, epsilon=None, delta=None, theta=None, phi=None, rho=1, dm=None, subdomains=(), **kwargs)
```

<a name=".pysource.models.Model.space_order"></a>
#### space\_order

```python
 | @property
 | space_order()
```

<a name=".pysource.models.Model.dt"></a>
#### dt

```python
 | @dt.setter
 | dt(dt)
```

<a name=".pysource.models.Model.is_tti"></a>
#### is\_tti

```python
 | @property
 | is_tti()
```

<a name=".pysource.models.Model._max_vp"></a>
#### \_max\_vp

```python
 | @property
 | _max_vp()
```

<a name=".pysource.models.Model.critical_dt"></a>
#### critical\_dt

```python
 | @property
 | critical_dt()
```

Critical computational time step value from the CFL condition.

<a name=".pysource.models.Model.dm"></a>
#### dm

```python
 | @dm.setter
 | dm(dm)
```

Set a new model perturbation.


**Parameters:**


vp : float or array
New velocity in km/s.

<a name=".pysource.models.Model.vp"></a>
#### vp

```python
 | @vp.setter
 | vp(vp)
```

Set a new velocity model and update square slowness.


**Parameters:**


vp : float or array
New velocity in km/s.

<a name=".pysource.models.Model.m"></a>
#### m

```python
 | @property
 | m()
```

<a name=".pysource.models.Model.spacing_map"></a>
#### spacing\_map

```python
 | @property
 | spacing_map()
```

Map between spacing symbols and their values for each `SpaceDimension`.

<a name=".pysource.FD_utils"></a>
## pysource.FD\_utils

<a name=".pysource.FD_utils.laplacian"></a>
#### laplacian

```python
laplacian(v, irho)
```

Laplacian with density div( 1/rho grad) (u)

<a name=".pysource.FD_utils.ssa_tti"></a>
#### ssa\_tti

```python
ssa_tti(u, v, model)
```

TTI finite difference kernel.


**Parameters:**


u : TimeFunction
first TTI field
v : TimeFunction
second TTI field
model: Model
Model structure

<a name=".pysource.FD_utils.ssa_1"></a>
#### ssa\_1

```python
ssa_1(u, v, model)
```

First row of
gx_t(A * gx(P)) + gy_t( A1 * gy(P)) + gz_T( A2 * gz(P))

<a name=".pysource.FD_utils.ssa_2"></a>
#### ssa\_2

```python
ssa_2(u, v, model)
```

Second row of
gx_t(A * gx(P)) + gy_t( A1 * gy(P)) + gz_T( A2 * gz(P))

<a name=".pysource.FD_utils.angles_to_trig"></a>
#### angles\_to\_trig

```python
angles_to_trig(model)
```

<a name=".pysource.FD_utils.gx"></a>
#### gx

```python
gx(field, model)
```

Rotated first derivative in x
u: TTI field
model: Model structure
:return: du/dx in rotated coordinates

<a name=".pysource.FD_utils.gy"></a>
#### gy

```python
gy(field, model)
```

Rotated first derivative in y
u: TTI field
model: Model structure
:return: du/dy in rotated coordinates

<a name=".pysource.FD_utils.gz"></a>
#### gz

```python
gz(field, model)
```

Rotated first derivative in z
u: TI field
model: Model structure
:return: du/dz in rotated coordinates

<a name=".pysource.FD_utils.gx_T"></a>
#### gx\_T

```python
gx_T(field, model)
```

Rotated first derivative in x
u: TTI field
model: Model structure
:return: du/dx in rotated coordinates

<a name=".pysource.FD_utils.gy_T"></a>
#### gy\_T

```python
gy_T(field, model)
```

Rotated first derivative in y
u: TTI field
model: Model structure
:return: du/dy in rotated coordinates

<a name=".pysource.FD_utils.gz_T"></a>
#### gz\_T

```python
gz_T(field, model)
```

Rotated first derivative in z
u: TI field
model: Model structure
:return: du/dz in rotated coordinates

<a name=".pysource.interface"></a>
## pysource.interface

<a name=".pysource.interface.forward_rec"></a>
#### forward\_rec

```python
forward_rec(model, src_coords, wavelet, rec_coords, space_order=8, free_surface=False)
```

Forward modeling of a point source.
Outputs the shot record.

<a name=".pysource.interface.forward_rec_w"></a>
#### forward\_rec\_w

```python
forward_rec_w(model, weight, wavelet, rec_coords, space_order=8, free_surface=False)
```

Forward modeling of a point source.
Outputs the shot record.

<a name=".pysource.interface.forward_rec_wf"></a>
#### forward\_rec\_wf

```python
forward_rec_wf(model, src_coords, wavelet, rec_coords, space_order=8, free_surface=False)
```

Forward modeling of a point source.
Outputs the shot record.

<a name=".pysource.interface.forward_no_rec"></a>
#### forward\_no\_rec

```python
forward_no_rec(model, src_coords, wav, space_order=8, free_surface=False)
```

Forward modeling of a point source without receiver.
Outputs the full wavefield.

<a name=".pysource.interface.forward_wf_src"></a>
#### forward\_wf\_src

```python
forward_wf_src(model, u, rec_coords, space_order=8, free_surface=False)
```

Forward modeling of a full wavefield source.
Outputs the shot record.

<a name=".pysource.interface.forward_wf_src_norec"></a>
#### forward\_wf\_src\_norec

```python
forward_wf_src_norec(model, u, space_order=8, free_surface=False)
```

Forward modeling of a full wavefield source without receiver.
Outputs the full wavefield

<a name=".pysource.interface.adjoint_rec"></a>
#### adjoint\_rec

```python
adjoint_rec(model, src_coords, rec_coords, data, space_order=8, free_surface=False)
```

Adjoint/backward modeling of a shot record (receivers as source).
Outputs the adjoint wavefield sampled at the source location.

<a name=".pysource.interface.adjoint_w"></a>
#### adjoint\_w

```python
adjoint_w(model, rec_coords, data, wavelet, space_order=8, free_surface=False)
```

Adjoint/backward modeling of a shot record (receivers as source).
Outputs the adjoint wavefield sampled at the source location.

<a name=".pysource.interface.adjoint_no_rec"></a>
#### adjoint\_no\_rec

```python
adjoint_no_rec(model, rec_coords, data, space_order=8, free_surface=False)
```

Adjoint/backward modeling of a shot record (receivers as source).
Outputs the full adjoint wavefield.

<a name=".pysource.interface.adjoint_wf_src"></a>
#### adjoint\_wf\_src

```python
adjoint_wf_src(model, u, src_coords, space_order=8, free_surface=False)
```

Adjoint/backward modeling of a full wavefield (full wavefield as adjoint source).
Outputs the adjoint wavefield sampled at the source location.

<a name=".pysource.interface.adjoint_wf_src_norec"></a>
#### adjoint\_wf\_src\_norec

```python
adjoint_wf_src_norec(model, u, src_coords, space_order=8, free_surface=False)
```

Adjoint/backward modeling of a full wavefield (full wavefield as adjoint source).
Outputs the full adjoint wavefield.

<a name=".pysource.interface.grad_fwi"></a>
#### grad\_fwi

```python
grad_fwi(model, recin, rec_coords, u, space_order=8, free_surface=False)
```

<a name=".pysource.interface.born_rec"></a>
#### born\_rec

```python
born_rec(model, src_coords, wavelet, rec_coords, space_order=8, free_surface=False, isic=False)
```

Linearized (Born) modeling of a point source for a model perturbation (square slowness) dm.
Output the linearized data.

<a name=".pysource.interface.born_rec_w"></a>
#### born\_rec\_w

```python
born_rec_w(model, weight, wavelet, rec_coords, space_order=8, free_surface=False, isic=False)
```

Linearized (Born) modeling of a point source for a model perturbation (square slowness) dm.
Output the linearized data.

<a name=".pysource.interface.J_adjoint"></a>
#### J\_adjoint

```python
J_adjoint(model, src_coords, wavelet, rec_coords, recin, space_order=8, checkpointing=False, free_surface=False, n_checkpoints=None, maxmem=None, freq_list=[], dft_sub=None, isic=False, ws=None)
```

Jacobian (adjoint fo born modeling operator) iperator on a shot record as a source (i.e data residual).
Outputs the gradient.
Supports three modes:
* Checkpinting
* Frequency compression (on-the-fly DFT)
* Standard zero lag cross correlation over time

<a name=".pysource.interface.J_adjoint_freq"></a>
#### J\_adjoint\_freq

```python
J_adjoint_freq(model, src_coords, wavelet, rec_coords, recin, space_order=8, free_surface=False, freq_list=[], is_residual=False, return_obj=False, dft_sub=None, isic=False, ws=None)
```

Gradient (appication of Jacobian to a shot record) computed with on-the-fly
Fourier transform.
Outputs gradient, and objective function (least-square) if requested.

<a name=".pysource.interface.J_adjoint_standard"></a>
#### J\_adjoint\_standard

```python
J_adjoint_standard(model, src_coords, wavelet, rec_coords, recin, space_order=8, free_surface=False, is_residual=False, return_obj=False, isic=False, ws=None)
```

Gradient (appication of Jacobian to a shot record) computed with the standard sum over time.
Outputs gradient, and objective function (least-square) if requested.

<a name=".pysource.interface.J_adjoint_checkpointing"></a>
#### J\_adjoint\_checkpointing

```python
J_adjoint_checkpointing(model, src_coords, wavelet, rec_coords, recin, space_order=8, free_surface=False, is_residual=False, n_checkpoints=None, maxmem=None, return_obj=False, isic=False, ws=None)
```

Gradient (appication of Jacobian to a shot record) computed with (optimal?) checkpointing.
Outputs gradient, and objective function (least-square) if requested.

<a name=".pysource.adjoint_test_F"></a>
## pysource.adjoint\_test\_F

<a name=".pysource.adjoint_test_F.parser"></a>
#### parser

<a name=".pysource.adjoint_test_F.args"></a>
#### args

<a name=".pysource.adjoint_test_F.is_tti"></a>
#### is\_tti

<a name=".pysource.adjoint_test_F.shape"></a>
#### shape

<a name=".pysource.adjoint_test_F.spacing"></a>
#### spacing

<a name=".pysource.adjoint_test_F.origin"></a>
#### origin

<a name=".pysource.adjoint_test_F.v"></a>
#### v

<a name=".pysource.adjoint_test_F.rho"></a>
#### rho

<a name=".pysource.adjoint_test_F.v[:]"></a>
#### v[:]

<a name=".pysource.adjoint_test_F.rho[:]"></a>
#### rho[:]

<a name=".pysource.adjoint_test_F.vp_i"></a>
#### vp\_i

<a name=".pysource.adjoint_test_F.rho_i"></a>
#### rho\_i

<a name=".pysource.adjoint_test_F.t0"></a>
#### t0

<a name=".pysource.adjoint_test_F.tn"></a>
#### tn

<a name=".pysource.adjoint_test_F.dt"></a>
#### dt

<a name=".pysource.adjoint_test_F.nt"></a>
#### nt

<a name=".pysource.adjoint_test_F.time_axis"></a>
#### time\_axis

<a name=".pysource.adjoint_test_F.f1"></a>
#### f1

<a name=".pysource.adjoint_test_F.src1"></a>
#### src1

<a name=".pysource.adjoint_test_F.src1.coordinates.data[0, :]"></a>
#### src1.coordinates.data[0, :]

<a name=".pysource.adjoint_test_F.src1.coordinates.data[0, -1]"></a>
#### src1.coordinates.data[0, -1]

<a name=".pysource.adjoint_test_F.rec_t"></a>
#### rec\_t

<a name=".pysource.adjoint_test_F.rec_t.coordinates.data[:, 0]"></a>
#### rec\_t.coordinates.data[:, 0]

<a name=".pysource.adjoint_test_F.rec_t.coordinates.data[:, 1]"></a>
#### rec\_t.coordinates.data[:, 1]

<a name=".pysource.adjoint_test_F.d_hat, u1"></a>
#### d\_hat, u1

<a name=".pysource.adjoint_test_F.q0, _"></a>
#### q0, \_

<a name=".pysource.adjoint_test_F.a"></a>
#### a

<a name=".pysource.adjoint_test_F.b"></a>
#### b

<a name=".pysource.test"></a>
## pysource.test

<a name=".pysource.test.parser"></a>
#### parser

<a name=".pysource.test.args"></a>
#### args

<a name=".pysource.test.is_tti"></a>
#### is\_tti

<a name=".pysource.test.shape"></a>
#### shape

<a name=".pysource.test.spacing"></a>
#### spacing

<a name=".pysource.test.origin"></a>
#### origin

<a name=".pysource.test.v"></a>
#### v

<a name=".pysource.test.v0"></a>
#### v0

<a name=".pysource.test.rho"></a>
#### rho

<a name=".pysource.test.rho0"></a>
#### rho0

<a name=".pysource.test.v[:]"></a>
#### v[:]

<a name=".pysource.test.v0[:]"></a>
#### v0[:]

<a name=".pysource.test.rho[:]"></a>
#### rho[:]

<a name=".pysource.test.rho0[:]"></a>
#### rho0[:]

<a name=".pysource.test.vp_i"></a>
#### vp\_i

<a name=".pysource.test.rho_i"></a>
#### rho\_i

<a name=".pysource.test.dm"></a>
#### dm

<a name=".pysource.test.t0"></a>
#### t0

<a name=".pysource.test.tn"></a>
#### tn

<a name=".pysource.test.dt"></a>
#### dt

<a name=".pysource.test.nt"></a>
#### nt

<a name=".pysource.test.time_axis"></a>
#### time\_axis

<a name=".pysource.test.f1"></a>
#### f1

<a name=".pysource.test.src"></a>
#### src

<a name=".pysource.test.src.coordinates.data[0, :]"></a>
#### src.coordinates.data[0, :]

<a name=".pysource.test.src.coordinates.data[0, -1]"></a>
#### src.coordinates.data[0, -1]

<a name=".pysource.test.nrec"></a>
#### nrec

<a name=".pysource.test.rec_t"></a>
#### rec\_t

<a name=".pysource.test.rec_t.coordinates.data[:, 0]"></a>
#### rec\_t.coordinates.data[:, 0]

<a name=".pysource.test.rec_t.coordinates.data[:, 1]"></a>
#### rec\_t.coordinates.data[:, 1]

<a name=".pysource.test.N"></a>
#### N

<a name=".pysource.test.a"></a>
#### a

<a name=".pysource.test.b"></a>
#### b

<a name=".pysource.test.freq_list"></a>
#### freq\_list

<a name=".pysource.test.dft_sub"></a>
#### dft\_sub

<a name=".pysource.test.d_lin, _"></a>
#### d\_lin, \_

<a name=".pysource.test.d0, u0"></a>
#### d0, u0

<a name=".pysource.test.g"></a>
#### g

<a name=".pysource.test.g2"></a>
#### g2

<a name=".pysource.sensitivit"></a>
## pysource.sensitivit

<a name=".pysource.sensitivit.func_name"></a>
#### func\_name

```python
func_name(freq=None, isic=False)
```

Get key for imaging condition/linearized source function

<a name=".pysource.sensitivit.grad_expr"></a>
#### grad\_expr

```python
grad_expr(gradm, u, v, model, w=1, freq=None, dft_sub=None, isic=False)
```

Gradient update stencil


**Parameters:**


u: TimeFunction or Tuple
Forward wavefield (tuple of fields for TTI or dft)
v: TimeFunction or Tuple
Adjoint wavefield (tuple of fields for TTI)
model: Model
Model structure
w: Float or Expr (optional)
Weight for the gradient expression (default=1)
freq: Array
Array of frequencies for on-the-fly DFT
factor: int
Subsampling factor for DFT
isic: Bool
Whether or not to use inverse scattering imaging condition (not supported yet)

<a name=".pysource.sensitivit.corr_freq"></a>
#### corr\_freq

```python
corr_freq(u, v, model, freq=None, dft_sub=None, **kwargs)
```

Standard cross-correlation imaging condition with on-th-fly-dft


**Parameters:**


u: TimeFunction or Tuple
Forward wavefield (tuple of fields for TTI or dft)
v: TimeFunction or Tuple
Adjoint wavefield (tuple of fields for TTI)
model: Model
Model structure
freq: Array
Array of frequencies for on-the-fly DFT
factor: int
Subsampling factor for DFT

<a name=".pysource.sensitivit.corr_fields"></a>
#### corr\_fields

```python
corr_fields(u, v, model, **kwargs)
```

Cross correlation of forward and adjoint wavefield


**Parameters:**


u: TimeFunction or Tuple
Forward wavefield (tuple of fields for TTI or dft)
v: TimeFunction or Tuple
Adjoint wavefield (tuple of fields for TTI)
model: Model
Model structure

<a name=".pysource.sensitivit.isic_g"></a>
#### isic\_g

```python
isic_g(u, v, model, **kwargs)
```

Inverse scattering imaging condition


**Parameters:**


u: TimeFunction or Tuple
Forward wavefield (tuple of fields for TTI or dft)
v: TimeFunction or Tuple
Adjoint wavefield (tuple of fields for TTI)
model: Model
Model structure

<a name=".pysource.sensitivit.isic_freq_g"></a>
#### isic\_freq\_g

```python
isic_freq_g(u, v, model, **kwargs)
```

Inverse scattering imaging condition


**Parameters:**


u: TimeFunction or Tuple
Forward wavefield (tuple of fields for TTI or dft)
v: TimeFunction or Tuple
Adjoint wavefield (tuple of fields for TTI)
model: Model
Model structure

<a name=".pysource.sensitivit.lin_src"></a>
#### lin\_src

```python
lin_src(model, u, isic=False)
```

Source for linearized modeling


**Parameters:**


u: TimeFunction or Tuple
Forward wavefield (tuple of fields for TTI or dft)
model: Model
Model containing the perturbation dm

<a name=".pysource.sensitivit.basic_src"></a>
#### basic\_src

```python
basic_src(model, u, **kwargs)
```

Basic source for linearized modeling


**Parameters:**


u: TimeFunction or Tuple
Forward wavefield (tuple of fields for TTI or dft)
model: Model
Model containing the perturbation dm

<a name=".pysource.sensitivit.isic_s"></a>
#### isic\_s

```python
isic_s(model, u, **kwargs)
```

ISIC source for linearized modeling


**Parameters:**


u: TimeFunction or Tuple
Forward wavefield (tuple of fields for TTI or dft)
model: Model
Model containing the perturbation dm

<a name=".pysource.sensitivit.ic_dict"></a>
#### ic\_dict

<a name=".pysource.sensitivit.ls_dict"></a>
#### ls\_dict

<a name=".pysource.sources"></a>
## pysource.sources

<a name=".pysource.sources.__all__"></a>
#### \_\_all\_\_

<a name=".pysource.sources.TimeAxis"></a>
### TimeAxis

```python
class TimeAxis(object)
```

Data object to store the TimeAxis. Exactly three of the four key arguments
must be prescribed. Because of remainder values it is not possible to create
a TimeAxis that exactly adhears to the inputs therefore start, stop, step
and num values should be taken from the TimeAxis object rather than relying
upon the input values.
The four possible cases are:
start is None: start = step*(1 - num) + stop
step is None: step = (stop - start)/(num - 1)
num is None: num = ceil((stop - start + step)/step);
because of remainder stop = step*(num - 1) + start
stop is None: stop = step*(num - 1) + start

**Parameters:**


start : float, optional
Start of time axis.
step : float, optional
Time interval.
num : int, optional
Number of values (Note: this is the number of intervals + 1).
Stop value is reset to correct for remainder.
stop : float, optional
End time.

<a name=".pysource.sources.TimeAxis.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(start=None, step=None, num=None, stop=None)
```

<a name=".pysource.sources.TimeAxis.__str__"></a>
#### \_\_str\_\_

```python
 | __str__()
```

<a name=".pysource.sources.TimeAxis._rebuild"></a>
#### \_rebuild

```python
 | _rebuild()
```

<a name=".pysource.sources.TimeAxis.time_values"></a>
#### time\_values

```python
 | @cached_property
 | time_values()
```

<a name=".pysource.sources.PointSource"></a>
### PointSource

```python
class PointSource(SparseTimeFunction)
```

Symbolic data object for a set of sparse point sources
:param name: Name of the symbol representing this source
:param grid: :class:`Grid` object defining the computational domain.
:param coordinates: Point coordinates for this source
:param data: (Optional) Data values to initialise point data
:param ntime: (Optional) Number of timesteps for which to allocate data
:param npoint: (Optional) Number of sparse points represented by this source
:param dimension: :(Optional) class:`Dimension` object for
representing the number of points in this source
Note, either the dimensions `ntime` and `npoint` or the fully
initialised `data` array need to be provided.

<a name=".pysource.sources.PointSource.__new__"></a>
#### \_\_new\_\_

```python
 | __new__(cls, *args, **kwargs)
```

<a name=".pysource.sources.Receiver"></a>
#### Receiver

<a name=".pysource.sources.Shot"></a>
#### Shot

<a name=".pysource.sources.WaveletSource"></a>
### WaveletSource

```python
class WaveletSource(PointSource)
```

Abstract base class for symbolic objects that encapsulate a set of
sources with a pre-defined source signal wavelet.
:param name: Name for the resulting symbol
:param grid: :class:`Grid` object defining the computational domain.
:param f0: Peak frequency for Ricker wavelet in kHz
:param time: Discretized values of time in ms

<a name=".pysource.sources.WaveletSource.__new__"></a>
#### \_\_new\_\_

```python
 | __new__(cls, *args, **kwargs)
```

<a name=".pysource.sources.WaveletSource.wavelet"></a>
#### wavelet

```python
 | wavelet(f0, t)
```

Defines a wavelet with a peak frequency f0 at time t.
:param f0: Peak frequency in kHz
:param t: Discretized values of time in ms

<a name=".pysource.sources.RickerSource"></a>
### RickerSource

```python
class RickerSource(WaveletSource)
```

Symbolic object that encapsulate a set of sources with a
pre-defined Ricker wavelet:
http://subsurfwiki.org/wiki/Ricker_wavelet
:param name: Name for the resulting symbol
:param grid: :class:`Grid` object defining the computational domain.
:param f0: Peak frequency for Ricker wavelet in kHz
:param time: Discretized values of time in ms

<a name=".pysource.sources.RickerSource.wavelet"></a>
#### wavelet

```python
 | wavelet(f0, t)
```

Defines a Ricker wavelet with a peak frequency f0 at time t.
:param f0: Peak frequency in kHz
:param t: Discretized values of time in ms

<a name=".pysource.sources.GaborSource"></a>
### GaborSource

```python
class GaborSource(WaveletSource)
```

Symbolic object that encapsulate a set of sources with a
pre-defined Gabor wavelet:
https://en.wikipedia.org/wiki/Gabor_wavelet
:param name: Name for the resulting symbol
:param grid: :class:`Grid` object defining the computational domain.
:param f0: Peak frequency for Ricker wavelet in kHz
:param time: Discretized values of time in ms

<a name=".pysource.sources.GaborSource.wavelet"></a>
#### wavelet

```python
 | wavelet(f0, t)
```

Defines a Gabor wavelet with a peak frequency f0 at time t.
:param f0: Peak frequency in kHz
:param t: Discretized values of time in ms

<a name=".pysource.kernels"></a>
## pysource.kernels

<a name=".pysource.kernels.wave_kernel"></a>
#### wave\_kernel

```python
wave_kernel(model, u, fw=True, q=None, fs=False)
```

Pde kernel corresponding the the model for the input wavefield


**Parameters:**


model: Model
Physical model
u : TimeFunction or tuple
wavefield (tuple if TTI)
fw : Bool
Whether forward or backward in time propagation
q : TimeFunction or Expr
Full time-space source
fs : Bool
Freesurface flag

<a name=".pysource.kernels.acoustic_kernel"></a>
#### acoustic\_kernel

```python
acoustic_kernel(model, u, fw=True, q=None)
```

Acoustic wave equation time stepper


**Parameters:**


model: Model
Physical model
u : TimeFunction or tuple
wavefield (tuple if TTI)
fw : Bool
Whether forward or backward in time propagation
q : TimeFunction or Expr
Full time-space source

<a name=".pysource.kernels.tti_kernel"></a>
#### tti\_kernel

```python
tti_kernel(model, u1, u2, fw=True, q=None)
```

TTI wave equation (one from my paper) time stepper


**Parameters:**


model: Model
Physical model
u1 : TimeFunction
First component (pseudo-P) of the wavefield
u2 : TimeFunction
First component (pseudo-P) of the wavefield
fw: Bool
Whether forward or backward in time propagation
q : TimeFunction or Expr
Full time-space source as a tuple (one value for each component)

<a name=".pysource.adjoint_test_J"></a>
## pysource.adjoint\_test\_J

<a name=".pysource.adjoint_test_J.parser"></a>
#### parser

<a name=".pysource.adjoint_test_J.args"></a>
#### args

<a name=".pysource.adjoint_test_J.is_tti"></a>
#### is\_tti

<a name=".pysource.adjoint_test_J.shape"></a>
#### shape

<a name=".pysource.adjoint_test_J.spacing"></a>
#### spacing

<a name=".pysource.adjoint_test_J.origin"></a>
#### origin

<a name=".pysource.adjoint_test_J.v"></a>
#### v

<a name=".pysource.adjoint_test_J.rho"></a>
#### rho

<a name=".pysource.adjoint_test_J.v[:]"></a>
#### v[:]

<a name=".pysource.adjoint_test_J.rho[:]"></a>
#### rho[:]

<a name=".pysource.adjoint_test_J.vp_i"></a>
#### vp\_i

<a name=".pysource.adjoint_test_J.v0[v < 1.51]"></a>
#### v0[v < 1.51]

<a name=".pysource.adjoint_test_J.v0"></a>
#### v0

<a name=".pysource.adjoint_test_J.rho0"></a>
#### rho0

<a name=".pysource.adjoint_test_J.dm"></a>
#### dm

<a name=".pysource.adjoint_test_J.dm[:, -1]"></a>
#### dm[:, -1]

<a name=".pysource.adjoint_test_J.t0"></a>
#### t0

<a name=".pysource.adjoint_test_J.tn"></a>
#### tn

<a name=".pysource.adjoint_test_J.dt"></a>
#### dt

<a name=".pysource.adjoint_test_J.nt"></a>
#### nt

<a name=".pysource.adjoint_test_J.time_axis"></a>
#### time\_axis

<a name=".pysource.adjoint_test_J.f1"></a>
#### f1

<a name=".pysource.adjoint_test_J.src"></a>
#### src

<a name=".pysource.adjoint_test_J.src.coordinates.data[0, :]"></a>
#### src.coordinates.data[0, :]

<a name=".pysource.adjoint_test_J.src.coordinates.data[0, -1]"></a>
#### src.coordinates.data[0, -1]

<a name=".pysource.adjoint_test_J.rec_t"></a>
#### rec\_t

<a name=".pysource.adjoint_test_J.rec_t.coordinates.data[:, 0]"></a>
#### rec\_t.coordinates.data[:, 0]

<a name=".pysource.adjoint_test_J.rec_t.coordinates.data[:, 1]"></a>
#### rec\_t.coordinates.data[:, 1]

<a name=".pysource.adjoint_test_J.dD_hat, u0l"></a>
#### dD\_hat, u0l

<a name=".pysource.adjoint_test_J._, u0"></a>
#### \_, u0

<a name=".pysource.adjoint_test_J.dm_hat"></a>
#### dm\_hat

<a name=".pysource.adjoint_test_J.a"></a>
#### a

<a name=".pysource.adjoint_test_J.b"></a>
#### b

<a name=".pysource.geom_utils"></a>
## pysource.geom\_utils

<a name=".pysource.geom_utils.src_rec"></a>
#### src\_rec

```python
src_rec(model, u, src_coords=None, rec_coords=None, wavelet=None, fw=True, nt=None)
```

Generates the source injection and receiver interpolation.
This function is fully abstracted and does not care whether this is a forward or adjoint wave-equation.
The source is the source term of the equation
The receiver is the measurment term

Therefore, for the adjoint, this function has to be called as:
src_rec(model, v, src_coords=rec_coords, ...)
because the data is the sources


**Parameters:**


model : Model
Physical model
u : TimeFunction or tuple
Wavefield to inject into and read from
src_coords : Array
Physical coordinates of the sources
rec_coords : Array
Physical coordinates of the receivers
wavelet: Array
Data for the source
fw=True:
Whether the direction is forward or backward in time
nt: int
Number of time steps

<a name=".pysource.geom_utils.AcquisitionGeometry"></a>
### AcquisitionGeometry

```python
class AcquisitionGeometry(Pickable)
```

Encapsulate the geometry of an acquisition:
- receiver positions and number
- source positions and number
In practice this would only point to a segy file with the
necessary information

<a name=".pysource.geom_utils.AcquisitionGeometry.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(model, rec_positions, src_positions, t0, tn, **kwargs)
```

In practice would be __init__(segyfile) and all below parameters
would come from a segy_read (at property call rather than at init)

<a name=".pysource.geom_utils.AcquisitionGeometry.resample"></a>
#### resample

```python
 | resample(dt)
```

<a name=".pysource.geom_utils.AcquisitionGeometry.time_axis"></a>
#### time\_axis

```python
 | @property
 | time_axis()
```

<a name=".pysource.geom_utils.AcquisitionGeometry.model"></a>
#### model

```python
 | @model.setter
 | model(model)
```

<a name=".pysource.geom_utils.AcquisitionGeometry.src_type"></a>
#### src\_type

```python
 | @property
 | src_type()
```

<a name=".pysource.geom_utils.AcquisitionGeometry.grid"></a>
#### grid

```python
 | @property
 | grid()
```

<a name=".pysource.geom_utils.AcquisitionGeometry.f0"></a>
#### f0

```python
 | @property
 | f0()
```

<a name=".pysource.geom_utils.AcquisitionGeometry.tn"></a>
#### tn

```python
 | @property
 | tn()
```

<a name=".pysource.geom_utils.AcquisitionGeometry.t0"></a>
#### t0

```python
 | @property
 | t0()
```

<a name=".pysource.geom_utils.AcquisitionGeometry.dt"></a>
#### dt

```python
 | @property
 | dt()
```

<a name=".pysource.geom_utils.AcquisitionGeometry.nt"></a>
#### nt

```python
 | @property
 | nt()
```

<a name=".pysource.geom_utils.AcquisitionGeometry.nrec"></a>
#### nrec

```python
 | @property
 | nrec()
```

<a name=".pysource.geom_utils.AcquisitionGeometry.nsrc"></a>
#### nsrc

```python
 | @property
 | nsrc()
```

<a name=".pysource.geom_utils.AcquisitionGeometry.dtype"></a>
#### dtype

```python
 | @property
 | dtype()
```

<a name=".pysource.geom_utils.AcquisitionGeometry.rec"></a>
#### rec

```python
 | @property
 | rec()
```

<a name=".pysource.geom_utils.AcquisitionGeometry.src"></a>
#### src

```python
 | @property
 | src()
```

<a name=".pysource.geom_utils.AcquisitionGeometry._pickle_args"></a>
#### \_pickle\_args

<a name=".pysource.geom_utils.AcquisitionGeometry._pickle_kwargs"></a>
#### \_pickle\_kwargs

<a name=".pysource.geom_utils.sources"></a>
#### sources

<a name=".pysource.wave_utils"></a>
## pysource.wave\_utils

<a name=".pysource.wave_utils.wavefield"></a>
#### wavefield

```python
wavefield(model, space_order, save=False, nt=None, fw=True, name='')
```

Create the wavefield for the wave equation


**Parameters:**



model : Model
Physical model
space_order: int
Spatial discretization order
save : Bool
Whether or not to save the time history
nt : int (optional)
Number of time steps if the wavefield is saved
fw : Bool
Forward or backward (for naming)
name: string
Custom name attached to default (u+name)

<a name=".pysource.wave_utils.wf_as_src"></a>
#### wf\_as\_src

```python
wf_as_src(v, w=1)
```

Weighted source as a time-space wavefield


**Parameters:**


u: TimeFunction or Tuple
Forward wavefield (tuple of fields for TTI or dft)
w: Float or Expr (optional)
Weight for the source expression (default=1)

<a name=".pysource.wave_utils.extented_src"></a>
#### extented\_src

```python
extented_src(model, weight, wavelet, q=0)
```

Extended source for modelling where the source is the outer product of
a spatially varying weight and a time-dependent wavelet i.e.:
u.dt2 - u.laplace = w(x)*q(t)
This function returns the extended source w(x)*q(t)


**Parameters:**


model: Model
Physical model structure
weight: Array
Array of weight for the spatial Function
wavelet: Array
Time-serie for the time-varying source
q: Symbol or Expr (optional)
Previously existing source to be added to (source will be q +  w(x)*q(t))

<a name=".pysource.wave_utils.extended_src_weights"></a>
#### extended\_src\_weights

```python
extended_src_weights(model, wavelet, v)
```

Adjoint of extended source. This function returns the expression to obtain
the spatially varrying weights from the wavefield and time-dependent wavelet


**Parameters:**


model: Model
Physical model structure
wavelet: Array
Time-serie for the time-varying source
v: TimeFunction
Wavefield to get the weights from

<a name=".pysource.wave_utils.freesurface"></a>
#### freesurface

```python
freesurface(field, npml, forward=True)
```

Generate the stencil that mirrors the field as a free surface modeling for
the acoustic wave equation


**Parameters:**


field: TimeFunction or Tuple
Field for which to add a free surface
npml: int
Number of ABC points
forward: Bool
Whether it is forward or backward propagation (in time)

<a name=".pysource.wave_utils.otf_dft"></a>
#### otf\_dft

```python
otf_dft(u, freq, dt, factor=None)
```

On the fly DFT wavefield (frequency slices) and expression


**Parameters:**


u: TimeFunction or Tuple
Forward wavefield
freq: Array
Array of frequencies for on-the-fly DFT
factor: int
Subsampling factor for DFT

<a name=".pysource.wave_utils.sub_time"></a>
#### sub\_time

```python
sub_time(time, factor, dt=1, freq=None)
```

Subsampled  time axis


**Parameters:**


time: Dimension
time Dimension
factor: int
Subsampling factor

