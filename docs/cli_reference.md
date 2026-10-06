# CLI configuration reference

Generated from `beat.cli.config` by `scripts/gen_cli_reference.py` -- do not edit.
Quantities are strings with units, e.g. `"0.05 ms"`.

## `[geometry]`

### FolderGeometry (`folder`)

| Field | Type | Default | Description |
|---|---|---|---|
| `unit` | str | `'mm'` | Length unit of the mesh coordinates |
| `folder` | path | `'geometry'` | type=folder: folder to read. Generated types (slab, lv_ellipsoid, biv_ellipsoid, ukb): cache root, each mesh is cached in its own <hash>/ subfolder |
| `type` | 'folder' | `'folder'` |  |
| `fibers` | FromGeometryFibers \| AxisFibers \| IsotropicFibers | `FromGeometryFibers(type='from_geometry')` |  |

### IntervalGeometry (`interval`)

| Field | Type | Default | Description |
|---|---|---|---|
| `unit` | str | `'mm'` | Length unit of the mesh coordinates |
| `folder` | path | `'geometry'` | type=folder: folder to read. Generated types (slab, lv_ellipsoid, biv_ellipsoid, ukb): cache root, each mesh is cached in its own <hash>/ subfolder |
| `type` | 'interval' | `'interval'` |  |
| `length` | float | `10.0` | Cable length (geometry.unit) |
| `dx` | float | `0.1` | Element size (geometry.unit) |
| `fibers` | FromGeometryFibers \| AxisFibers \| IsotropicFibers | `IsotropicFibers(type='isotropic')` |  |

### RectangleGeometry (`rectangle`)

| Field | Type | Default | Description |
|---|---|---|---|
| `unit` | str | `'mm'` | Length unit of the mesh coordinates |
| `folder` | path | `'geometry'` | type=folder: folder to read. Generated types (slab, lv_ellipsoid, biv_ellipsoid, ukb): cache root, each mesh is cached in its own <hash>/ subfolder |
| `type` | 'rectangle' | `'rectangle'` |  |
| `lx` | float | `1.0` |  |
| `ly` | float | `1.0` |  |
| `dx` | float | `0.05` |  |
| `fibers` | FromGeometryFibers \| AxisFibers \| IsotropicFibers | `IsotropicFibers(type='isotropic')` |  |

### BoxSlabGeometry (`box_slab`)

Structured tetrahedral box (beat.geometry.get_3D_slab_mesh); no gmsh needed.

| Field | Type | Default | Description |
|---|---|---|---|
| `unit` | str | `'mm'` | Length unit of the mesh coordinates |
| `folder` | path | `'geometry'` | type=folder: folder to read. Generated types (slab, lv_ellipsoid, biv_ellipsoid, ukb): cache root, each mesh is cached in its own <hash>/ subfolder |
| `type` | 'box_slab' | `'box_slab'` |  |
| `lx` | float | `20.0` |  |
| `ly` | float | `7.0` |  |
| `lz` | float | `3.0` |  |
| `dx` | float | `0.5` |  |
| `fibers` | FromGeometryFibers \| AxisFibers \| IsotropicFibers | `AxisFibers(type='axis', direction='x')` |  |

### SlabGeometry (`slab`)

| Field | Type | Default | Description |
|---|---|---|---|
| `unit` | str | `'mm'` | Length unit of the mesh coordinates |
| `folder` | path | `'geometry'` | type=folder: folder to read. Generated types (slab, lv_ellipsoid, biv_ellipsoid, ukb): cache root, each mesh is cached in its own <hash>/ subfolder |
| `fiber_angle_endo` | float | `60.0` |  |
| `fiber_angle_epi` | float | `-60.0` |  |
| `fiber_space` | str | `'P_1'` |  |
| `fibers` | FromGeometryFibers \| AxisFibers \| IsotropicFibers | `FromGeometryFibers(type='from_geometry')` |  |
| `type` | 'slab' | `'slab'` |  |
| `lx` | float | `20.0` |  |
| `ly` | float | `7.0` |  |
| `lz` | float | `3.0` |  |
| `dx` | float | `1.0` |  |

### LVEllipsoidGeometry (`lv_ellipsoid`)

| Field | Type | Default | Description |
|---|---|---|---|
| `unit` | str | `'mm'` | Length unit of the mesh coordinates |
| `folder` | path | `'geometry'` | type=folder: folder to read. Generated types (slab, lv_ellipsoid, biv_ellipsoid, ukb): cache root, each mesh is cached in its own <hash>/ subfolder |
| `fiber_angle_endo` | float | `60.0` |  |
| `fiber_angle_epi` | float | `-60.0` |  |
| `fiber_space` | str | `'P_1'` |  |
| `fibers` | FromGeometryFibers \| AxisFibers \| IsotropicFibers | `FromGeometryFibers(type='from_geometry')` |  |
| `type` | 'lv_ellipsoid' | `'lv_ellipsoid'` |  |
| `r_short_endo` | float | `7.0` |  |
| `r_short_epi` | float | `10.0` |  |
| `r_long_endo` | float | `17.0` |  |
| `r_long_epi` | float | `20.0` |  |
| `psize_ref` | float | `3.0` |  |
| `mu_apex_endo` | float | `-3.141592653589793` |  |
| `mu_base_endo` | float | `-1.2722641256100204` |  |
| `mu_apex_epi` | float | `-3.141592653589793` |  |
| `mu_base_epi` | float | `-1.318116071652818` |  |

### BiVEllipsoidGeometry (`biv_ellipsoid`)

| Field | Type | Default | Description |
|---|---|---|---|
| `unit` | str | `'mm'` | Length unit of the mesh coordinates |
| `folder` | path | `'geometry'` | type=folder: folder to read. Generated types (slab, lv_ellipsoid, biv_ellipsoid, ukb): cache root, each mesh is cached in its own <hash>/ subfolder |
| `fiber_angle_endo` | float | `60.0` |  |
| `fiber_angle_epi` | float | `-60.0` |  |
| `fiber_space` | str | `'P_1'` |  |
| `fibers` | FromGeometryFibers \| AxisFibers \| IsotropicFibers | `FromGeometryFibers(type='from_geometry')` |  |
| `type` | 'biv_ellipsoid' | `'biv_ellipsoid'` |  |
| `char_length` | float | `0.5` |  |

### UKBGeometry (`ukb`)

| Field | Type | Default | Description |
|---|---|---|---|
| `unit` | str | `'mm'` | Length unit of the mesh coordinates |
| `folder` | path | `'geometry'` | type=folder: folder to read. Generated types (slab, lv_ellipsoid, biv_ellipsoid, ukb): cache root, each mesh is cached in its own <hash>/ subfolder |
| `fiber_angle_endo` | float | `60.0` |  |
| `fiber_angle_epi` | float | `-60.0` |  |
| `fiber_space` | str | `'P_1'` |  |
| `fibers` | FromGeometryFibers \| AxisFibers \| IsotropicFibers | `FromGeometryFibers(type='from_geometry')` |  |
| `type` | 'ukb' | `'ukb'` |  |
| `mode` | int | `-1` |  |
| `std` | float | `1.5` |  |
| `case` | 'ED' \| 'ES' | `'ED'` |  |
| `char_length_max` | float | `5.0` |  |
| `char_length_min` | float | `5.0` |  |
| `clipped` | bool | `False` |  |

## `[geometry.fibers]`

### FromGeometryFibers (`from_geometry`)

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | 'from_geometry' | `'from_geometry'` |  |

### AxisFibers (`axis`)

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | 'axis' | `'axis'` |  |
| `direction` | 'x' \| 'y' \| 'z' | `'x'` |  |

### IsotropicFibers (`isotropic`)

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | 'isotropic' | `'isotropic'` |  |

## `[ep]`

### EPConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `chi` | Quantity | `'1400 cm**-1'` | Surface to volume ratio |
| `C_m` | Quantity | `'1 uF/cm**2'` | Membrane capacitance |
| `conductivity` | ConductivityConfig | `ConductivityConfig(preset=None, sigma_il=None, sigma_it=None, sigma_el=None, sigma_et=None)` |  |

### ConductivityConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `preset` | 'Niederer' \| 'Bishop' (optional) | – | Literature set from beat.conductivities.default_conductivities; explicit sigma_* override it field by field |
| `sigma_il` | Quantity (optional) | – |  |
| `sigma_it` | Quantity (optional) | – |  |
| `sigma_el` | Quantity (optional) | – |  |
| `sigma_et` | Quantity (optional) | – |  |

## `[cell]`

### CellConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `ode_file` | path | **required** | Path to the gotranx .ode cell model |
| `scheme` | str | `'generalized_rush_larsen'` |  |
| `v_name` | str | `'v'` | Name of the transmembrane potential state |
| `parameters` | dict[str, float] | `{}` |  |
| `steady_state` | SteadyStateConfig (optional) | – | If set, pre-pace a single cell to steady state before the tissue run |
| `layers` | NoLayers \| TransmuralLayers \| CellMarkerLayers | `NoLayers(method='none')` |  |
| `regions` | dict[str, RegionConfig] | `{}` |  |

### SteadyStateConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `num_beats` | int | `20` |  |
| `BCL` | Quantity | `'1000 ms'` |  |
| `dt` | Quantity | `'0.05 ms'` |  |
| `track` | list[str] | `[]` | States to record |

### RegionConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `parameters` | dict[str, float] | `{}` |  |

## `[cell.layers]`

### NoLayers (`none`)

| Field | Type | Default | Description |
|---|---|---|---|
| `method` | 'none' | `'none'` |  |

### TransmuralLayers (`transmural`)

| Field | Type | Default | Description |
|---|---|---|---|
| `method` | 'transmural' | `'transmural'` |  |
| `endo_size` | float | `0.3` |  |
| `epi_size` | float | `0.3` |  |
| `endo_markers` | list[str] (optional) | – | Facet markers of the endocardium. Default: ["ENDO"] if present, else ["LV", "RV"] (BiV) |
| `epi_marker` | str | `'EPI'` |  |

### CellMarkerLayers (`cell_markers`)

| Field | Type | Default | Description |
|---|---|---|---|
| `method` | 'cell_markers' | `'cell_markers'` |  |
| `map` | dict[str, str] | **required** | region name -> geometry cell marker name |

## `[[stimulus]]`

### MarkerStimulus (`marker`)

| Field | Type | Default | Description |
|---|---|---|---|
| `amplitude` | str | **required** | Current density: uA/cm**2 for a marker stimulus on a facet marker, uA/cm**3 for a marker stimulus on a cell marker and for box/random_endocardial stimuli (any mesh dimension); see beat.stimulation.define_stimulus |
| `duration` | Quantity | `'2 ms'` |  |
| `start` | Quantity | `'0 ms'` |  |
| `period` | Quantity (optional) | – | If set, repeat the pulse every period |
| `num_pulses` | int (optional) | – | Number of pulses (requires period). Default: until the end time |
| `type` | 'marker' | `'marker'` |  |
| `marker` | str | **required** | Facet or cell marker name from the geometry |

### BoxStimulus (`box`)

| Field | Type | Default | Description |
|---|---|---|---|
| `amplitude` | str | **required** | Current density: uA/cm**2 for a marker stimulus on a facet marker, uA/cm**3 for a marker stimulus on a cell marker and for box/random_endocardial stimuli (any mesh dimension); see beat.stimulation.define_stimulus |
| `duration` | Quantity | `'2 ms'` |  |
| `start` | Quantity | `'0 ms'` |  |
| `period` | Quantity (optional) | – | If set, repeat the pulse every period |
| `num_pulses` | int (optional) | – | Number of pulses (requires period). Default: until the end time |
| `type` | 'box' | `'box'` |  |
| `min` | list[float] | **required** |  |
| `max` | list[float] | **required** |  |

### RandomEndocardialStimulus (`random_endocardial`)

| Field | Type | Default | Description |
|---|---|---|---|
| `amplitude` | str | **required** | Current density: uA/cm**2 for a marker stimulus on a facet marker, uA/cm**3 for a marker stimulus on a cell marker and for box/random_endocardial stimuli (any mesh dimension); see beat.stimulation.define_stimulus |
| `duration` | Quantity | `'2 ms'` |  |
| `start` | Quantity | `'0 ms'` |  |
| `period` | Quantity (optional) | – | If set, repeat the pulse every period |
| `num_pulses` | int (optional) | – | Number of pulses (requires period). Default: until the end time |
| `type` | 'random_endocardial' | `'random_endocardial'` |  |
| `markers` | list[str] | `['LV', 'RV']` |  |
| `num_points` | int | `200` |  |
| `delay_range` | tuple[Quantity, Quantity] | `('0 ms', '4 ms')` |  |
| `seed` | int | `0` |  |
| `tol` | float | `1.0` | Radius around each point (geometry.unit) |

## `[solver]`

### SolverConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `dt` | Quantity | `'0.05 ms'` |  |
| `theta` | float | `1.0` | Splitting: 1.0 Godunov, 0.5 Strang |
| `end_time` | Quantity (optional) | – | Simulated end time. Give either this or num_beats and BCL |
| `num_beats` | int (optional) | – | Run length in beats: end time = num_beats x BCL |
| `BCL` | Quantity (optional) | – | Basic cycle length, only used for the run length (num_beats x BCL). It does not pace anything: set a [[stimulus]] period for that |
| `pde` | ThetaPDE \| IrksomePDE | `ThetaPDE(type='theta', theta=0.5, linear_solver='direct')` |  |
| `ode` | DolfinODE \| IrksomeODE \| ExternalOperatorODE | `DolfinODE(type='dolfin')` |  |
| `petsc_options` | dict[str, str \| int \| float \| bool] | `{}` |  |

## `[solver.pde]`

### ThetaPDE (`theta`)

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | 'theta' | `'theta'` |  |
| `theta` | float | `0.5` | PDE theta-scheme parameter |
| `linear_solver` | 'direct' \| 'iterative' | `'direct'` |  |

### IrksomePDE (`irksome`)

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | 'irksome' | `'irksome'` |  |
| `tableau` | str | `'RadauIIA'` | Irksome Butcher tableau class name |
| `stages` | int | `1` |  |
| `linear_solver` | 'direct' \| 'iterative' | `'direct'` |  |

## `[solver.ode]`

### DolfinODE (`dolfin`)

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | 'dolfin' | `'dolfin'` |  |

### IrksomeODE (`irksome`)

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | 'irksome' | `'irksome'` |  |
| `tableau` | str | `'RadauIIA'` |  |
| `stages` | int | `1` |  |

### ExternalOperatorODE (`external_operator`)

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | 'external_operator' | `'external_operator'` |  |

## `[output]`

### OutputConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `folder` | path | `'output'` |  |
| `save_every` | Quantity | `'1 ms'` |  |
| `fields` | list[str] | `[]` | Extra ODE states to save |
| `checkpoint_every` | Quantity | `'0 ms'` | Restart checkpoint interval; 0 = end only |
| `performance` | bool | `False` |  |
| `log_every` | int | `100` |  |

## `[postprocess]`

### PostprocessConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `points` | dict[str, list[float]] | `{}` |  |
| `activation_threshold` | float | `0.0` |  |
| `ecg` | ECGConfig (optional) | – |  |
| `vtx` | bool | `True` |  |
| `make_gif` | bool | `False` |  |

## `[postprocess.ecg]`

### ECGConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `electrodes` | dict[str, list[float]] | **required** | Electrode name -> position (2 or 3 numbers), in `unit`. Lead systems look electrodes up by name (twelve-lead: LA, RA, LL, V1-V6) |
| `unit` | str (optional) | – | Length unit of the electrode positions; default geometry.unit |
| `leads` | 'none' \| 'twelve-lead' | `'none'` | Derived lead system written to ecg_leads.csv: none or twelve-lead |
| `reference` | 'potential' \| 'position' | `'potential'` | How the leads' reference points are formed (needs leads != none): potential = Wilson terminal from the potentials, position = potentials evaluated at the derived points |
| `sigma_b` | float | `1.0` | Bath conductivity |
