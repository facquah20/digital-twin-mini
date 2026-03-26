#  River Ecosystem Digital Twin
## Real-Time Contaminant Transport Simulation

---

## Overview

A full scientific computing application that creates a **live 3D digital twin** of a 2 km river
system — simulating, visualising, and tracking the transport of multiple contaminant types in
real time using a physics-based PDE solver.

---

## Physics Model

The transport of contaminants is governed by the **2D depth-averaged Advection-Diffusion-Reaction equation**:

```
∂C/∂t + u·∂C/∂x + v·∂C/∂y = Dx·∂²C/∂x² + Dy·∂²C/∂y² − λC + S(x,y,t)
```

| Symbol | Meaning | Units |
|--------|---------|-------|
| `C`    | Contaminant concentration | mg/L |
| `u, v` | Depth-averaged flow velocity (x, y) | m/s |
| `Dx, Dy` | Longitudinal / transverse dispersion | m²/s |
| `λ`    | First-order decay rate | 1/s |
| `S`    | Source/sink term | mg/L/s |

### Numerical Scheme
- **Spatial**: Explicit Upwind (advection) + Central Difference (diffusion)
- **Temporal**: Forward Euler with CFL stability check
- **Grid**: 200 × 40 finite difference cells (Δx=10m, Δy=2m)
- **Stability**: CFL ≤ 0.9 enforced automatically

### Velocity Field
Generated using **Manning's equation** (depth-averaged):

```
U = (H^(2/3) · √S₀) / n
```

With meander-driven secondary circulation superimposed.

---

## Features

### 🌊 River Geometry
- Realistic **bathymetry** (river bed topography) with pools and riffles
- Parabolic cross-section (deeper in centre, shallower at banks)
- Sinuous thalweg (deepest path follows meanders)
- Green terrain banks on both sides

### ☣️ Contaminant Sources (5 Active)
| Source | Type | Location |
|--------|------|----------|
| Industrial Effluent (Factory A) | Heavy Metals | Km 0.48 |
| Chemical Spill | Organic | Km 0.65 |
| Agricultural Runoff | Nutrients | Km 0.85 |
| Sewage Overflow | Pathogens | Km 1.10 |
| Mining Leachate | Heavy Metals | Km 1.40 |

Each source has:
- Gaussian plume injection kernel
- Pulsed discharge cycles (industrial sources)
- Type-specific decay rates

### 📡 Monitoring Stations (6 Points)
- Upstream Reference (baseline)
- Industrial Outfall Zone
- Agricultural Buffer Zone
- Municipal Water Intake
- Wetland Buffer
- Downstream Gauge

Each station shows **real-time alert status**:
- 🟢 GREEN — Safe (below warning threshold)
- 🟠 ORANGE — Warning (above warning, below critical)
- 🔴 RED — Critical (above critical threshold)

### 🎨 Visualisation Layers
Switch between layers with keyboard keys:
- `[1]` Heavy Metals concentration
- `[2]` Nutrients concentration
- `[3]` Pathogens concentration
- `[4]` Organic compounds
- `[5]` Total Weighted Contamination Index

The colour map: **Green → Yellow → Red** (clean → contaminated)

---

## Installation

```bash
# 1. Create virtual environment (recommended)
python -m venv river_env
source river_env/bin/activate      # Linux/Mac
river_env\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python river_digital_twin.py
```

### Dependencies
- `numpy` — numerical arrays
- `scipy` — Gaussian smoothing, interpolation
- `pyvista` — 3D VTK-based visualisation

---

## Controls

| Key / Action | Function |
|---|---|
| Mouse drag | Rotate 3D view |
| Scroll wheel | Zoom in/out |
| Shift + drag | Pan view |
| `R` | Reset camera to default |
| `SPACE` | Pause / Resume simulation |
| `1` – `5` | Switch visualisation layer |
| `P` then click | Probe concentration at any river point |
| `S` | Toggle flow streamlines |
| `Q` / `Esc` | Quit |

---

## Architecture

```
river_digital_twin.py
│
├── RiverGeometry
│   ├── _generate_bathymetry()      ← Realistic bed topography
│   ├── _generate_velocity_field()  ← Manning's equation + meanders
│   └── build_surface_mesh()        ← PyVista StructuredGrid meshes
│
├── ContaminantTransport
│   ├── step()                      ← Advance PDE one time step
│   ├── add_sources()               ← Gaussian plume injection
│   └── probe(x, y)                 ← Read concentration at location
│
└── RiverDigitalTwin
    ├── run()                       ← Main visualisation loop
    ├── _attach_concentration_to_mesh()
    ├── _update_particles()         ← Animated flow tracers
    ├── _build_source_markers()     ← 3D source indicators
    ├── _build_station_markers()    ← Colour-coded alert spheres
    └── _format_hud_text()          ← Live data overlay
```

---

## Extending the Model

### Add a new contaminant source
```python
sources.append(ContaminantSource(
    name="New Spill Site",
    x=1300,              # metres from upstream
    y=5,                 # metres from centreline
    concentration=35.0,  # mg/L
    flow_rate=0.5,       # m³/s
    contaminant_type='heavy_metal',   # or 'organic', 'nutrient', 'pathogen'
    color=(1.0, 0.0, 0.0)
))
```

### Change decay rate
```python
transport.decay_rates['heavy_metal'] = 5e-6   # faster decay
```

### Increase grid resolution
```python
NX = 400   # finer cells → more accurate but slower
NY = 80
```

---

## Scientific References

1. Fischer et al. (1979). *Mixing in Inland and Coastal Waters*. Academic Press.
2. Elder, J.W. (1959). The dispersion of marked fluid in turbulent shear flow. *J. Fluid Mech.*
3. Chapra, S.C. (1997). *Surface Water-Quality Modeling*. McGraw-Hill.
4. LeVeque, R.J. (2002). *Finite Volume Methods for Hyperbolic Problems*. Cambridge.
