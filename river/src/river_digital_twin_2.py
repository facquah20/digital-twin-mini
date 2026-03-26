"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         RIVER ECOSYSTEM DIGITAL TWIN — CONTAMINANT TRANSPORT MODEL          ║
║         Scientific Computing + Real-Time 3D Visualization Engine            ║
║         Built with PyVista, NumPy, SciPy | Advection-Diffusion PDE          ║
╚══════════════════════════════════════════════════════════════════════════════╝

PHYSICS MODEL:
  ∂C/∂t + u·∇C = D·∇²C - λC + S(x,y,t)

  Where:
    C     = contaminant concentration [mg/L]
    u     = flow velocity field [m/s]
    D     = diffusion/dispersion coefficient [m²/s]
    λ     = first-order decay rate [1/s]
    S     = source term [mg/L/s]

FEATURES:
  • Real-time 3D river mesh with bathymetry
  • Advection-diffusion contaminant transport
  • Multiple contaminant sources (industrial, agricultural, point/non-point)
  • Live concentration probing at any river section
  • Animated flow particles
  • Full HUD with monitoring station data
  • VR-style first-person navigation

USAGE:
  python river_digital_twin.py

CONTROLS:
  Mouse drag     : Rotate view
  Scroll wheel   : Zoom in/out
  Shift+drag     : Pan
  R              : Reset camera
  Space          : Pause/Resume simulation
  1/2/3          : Switch contaminant layers
  P              : Pick point (probe concentration)
  S              : Toggle flow streamlines
  Q/Esc          : Quit
"""

import numpy as np
import pyvista as pv
from pyvista import themes
import time
import threading
from dataclasses import dataclass
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

RIVER_LENGTH   = 2000.0    # meters along x-axis
RIVER_WIDTH    = 80.0      # meters along y-axis
RIVER_DEPTH    = 5.0       # max depth meters
NX             = 200       # grid cells in x (along river)
NY             = 40        # grid cells in y (across river)
NZ             = 8         # vertical layers
DT             = 0.5       # time step [seconds]
TOTAL_TIME     = 3600.0    # 1 hour simulation
UPDATE_FPS     = 15        # visualization update rate

# ──────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ContaminantSource:
    name: str
    x: float           # position along river [m]
    y: float           # lateral position [m]
    concentration: float  # source concentration [mg/L]
    flow_rate: float   # discharge [m³/s]
    contaminant_type: str  # 'heavy_metal', 'organic', 'nutrient', 'pathogen'
    color: Tuple[float, float, float]
    active: bool = True

@dataclass
class MonitoringStation:
    name: str
    x: float
    y: float
    river_km: float
    readings: List[float]
    threshold_warning: float   # mg/L
    threshold_critical: float  # mg/L

# ──────────────────────────────────────────────────────────────────────────────
# RIVER GEOMETRY GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

class RiverGeometry:
    """Generates realistic river bathymetry with meanders"""

    def __init__(self, nx=NX, ny=NY, nz=NZ):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = RIVER_LENGTH / nx
        self.dy = RIVER_WIDTH  / ny

        # Base coordinates
        self.x_coords = np.linspace(0, RIVER_LENGTH, nx + 1)
        self.y_coords = np.linspace(-RIVER_WIDTH/2, RIVER_WIDTH/2, ny + 1)

        # Generate bathymetry (depth map)
        self.depth_map = self._generate_bathymetry()
        # Generate flow velocity field
        self.velocity_u, self.velocity_v = self._generate_velocity_field()

    def _generate_bathymetry(self):
        """Realistic river bathymetry: deeper in center, shallower at banks"""
        xx, yy = np.meshgrid(
            np.linspace(0, RIVER_LENGTH, self.nx),
            np.linspace(-RIVER_WIDTH/2, RIVER_WIDTH/2, self.ny),
            indexing='ij'
        )

        # Cross-section profile (parabolic)
        y_norm = yy / (RIVER_WIDTH / 2)
        depth_profile = RIVER_DEPTH * (1.0 - y_norm**2)

        # Longitudinal variation (riffles and pools)
        pool_riffle = 0.8 + 0.4 * np.sin(2 * np.pi * xx / 400.0)

        # Random bed roughness
        rng = np.random.default_rng(42)
        roughness = 0.3 * rng.standard_normal((self.nx, self.ny))
        roughness = self._smooth_field(roughness, sigma=3)

        # Meander influence (sinuosity shifts the thalweg)
        meander_offset = 0.3 * np.sin(2 * np.pi * xx / 600.0)
        meander_depth = RIVER_DEPTH * (1.0 - (y_norm - meander_offset)**2)
        meander_depth = np.clip(meander_depth, 0.5, RIVER_DEPTH)

        depth = 0.6 * depth_profile + 0.4 * meander_depth
        depth *= pool_riffle
        depth += roughness * 0.15
        depth = np.clip(depth, 0.3, RIVER_DEPTH * 1.2)

        return depth

    def _generate_velocity_field(self):
        """2D depth-averaged velocity field using simplified Manning's equation"""
        # Manning's n coefficient
        n_manning = 0.035

        # Hydraulic radius ≈ depth (wide river approximation)
        H = self.depth_map  # [m]
        S0 = 0.0005  # bed slope

        # Depth-averaged velocity (Manning's)
        U_mag = (H**(2/3) * np.sqrt(S0)) / n_manning

        # y-gradient of depth drives lateral velocity
        u = U_mag.copy()
        v = np.zeros_like(u)

        # Add meander-driven secondary circulation
        xx = np.linspace(0, RIVER_LENGTH, self.nx)[:, np.newaxis]
        yy = np.linspace(-RIVER_WIDTH/2, RIVER_WIDTH/2, self.ny)[np.newaxis, :]

        v += 0.15 * U_mag * np.cos(2 * np.pi * xx / 600.0) * (yy / RIVER_WIDTH)

        # Smooth velocity field
        u = self._smooth_field(u, sigma=2)
        v = self._smooth_field(v, sigma=2)

        # Zero velocity at banks
        bank_mask = np.abs(yy) > (RIVER_WIDTH/2 * 0.92)
        u *= (1 - 0.95 * bank_mask.astype(float))
        v *= (1 - 0.95 * bank_mask.astype(float))

        return u, v

    @staticmethod
    def _smooth_field(field, sigma=2):
        """Gaussian smoothing"""
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(field, sigma=sigma)

    def build_surface_mesh(self):
        """Build PyVista structured surface mesh of river bed + water surface"""
        # Create 3D structured grid
        x = np.linspace(0, RIVER_LENGTH, self.nx + 1)
        y = np.linspace(-RIVER_WIDTH/2, RIVER_WIDTH/2, self.ny + 1)

        xx, yy = np.meshgrid(x, y, indexing='ij')

        # Water surface (z = 0 plane with slight waves)
        rng = np.random.default_rng(0)
        wave_z = 0.05 * np.sin(2 * np.pi * xx / 50.0) * np.cos(2 * np.pi * yy / 20.0)

        # Bed elevation (negative = below waterline)
        # Interpolate depth to nodes
        from scipy.interpolate import RegularGridInterpolator
        xc = np.linspace(0, RIVER_LENGTH, self.nx)
        yc = np.linspace(-RIVER_WIDTH/2, RIVER_WIDTH/2, self.ny)
        interp = RegularGridInterpolator((xc, yc), self.depth_map, method='linear',
                                          bounds_error=False, fill_value=1.0)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        depth_nodes = interp(pts).reshape(self.nx + 1, self.ny + 1)

        bed_z = -depth_nodes

        # Water surface mesh
        water_points = np.column_stack([
            xx.ravel(), yy.ravel(), wave_z.ravel()
        ])
        surface = pv.StructuredGrid()
        surface.dimensions = [self.ny + 1, self.nx + 1, 1]
        surface.points = water_points[:, [1, 0, 2]]  # reorder for PyVista

        # River bed mesh
        bed_points = np.column_stack([
            xx.ravel(), yy.ravel(), bed_z.ravel()
        ])
        bed = pv.StructuredGrid()
        bed.dimensions = [self.ny + 1, self.nx + 1, 1]
        bed.points = bed_points[:, [1, 0, 2]]

        return surface, bed, depth_nodes


# ──────────────────────────────────────────────────────────────────────────────
# CONTAMINANT TRANSPORT PHYSICS ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class ContaminantTransport:
    """
    2D depth-averaged advection-diffusion-reaction model.

    Governing equation (explicit upwind scheme):
      ∂C/∂t = -u∂C/∂x - v∂C/∂y + Dx∂²C/∂x² + Dy∂²C/∂y² - λC + S
    """

    def __init__(self, nx=NX, ny=NY, velocity_u=None, velocity_v=None):
        self.nx = nx
        self.ny = ny
        self.dx = RIVER_LENGTH / nx
        self.dy = RIVER_WIDTH  / ny
        self.dt = DT

        # Velocity fields [m/s]
        self.u = velocity_u if velocity_u is not None else np.ones((nx, ny)) * 0.5
        self.v = velocity_v if velocity_v is not None else np.zeros((nx, ny))

        # Dispersion coefficients [m²/s]
        # Longitudinal dispersion >> transverse (Elder's formula)
        U_mean = np.mean(np.abs(self.u))
        self.Dx = 5.0 + 0.5 * U_mean * self.dx   # longitudinal
        self.Dy = 0.15 * U_mean * self.dy          # transverse

        # First-order decay rates per contaminant type [1/s]
        self.decay_rates = {
            'heavy_metal': 1e-6,    # conservative
            'organic':     2e-5,    # biodegradable
            'nutrient':    5e-6,    # moderate decay
            'pathogen':    3e-4,    # rapid die-off
        }

        # Initialize concentration fields — one per contaminant type
        self.concentrations = {
            'heavy_metal': np.zeros((nx, ny)),
            'organic':     np.zeros((nx, ny)),
            'nutrient':    np.zeros((nx, ny)),
            'pathogen':    np.zeros((nx, ny)),
        }

        # Background / ambient levels [mg/L]
        self.background = {
            'heavy_metal': 0.002,
            'organic':     0.5,
            'nutrient':    0.8,
            'pathogen':    0.1,
        }
        for key, val in self.background.items():
            self.concentrations[key][:] = val

        # Simulation time
        self.time = 0.0
        self.step_count = 0

        # Check CFL stability
        cfl_x = np.max(np.abs(self.u)) * self.dt / self.dx
        cfl_y = np.max(np.abs(self.v)) * self.dt / self.dy
        if cfl_x > 0.9 or cfl_y > 0.9:
            self.dt = 0.4 * min(self.dx / np.max(np.abs(self.u) + 1e-6),
                                 self.dy / np.max(np.abs(self.v) + 1e-6))

    def add_sources(self, sources: List[ContaminantSource], t: float):
        """Apply source terms to concentration fields"""
        for src in sources:
            if not src.active:
                continue
            # Convert position to grid indices
            ix = int(src.x / RIVER_LENGTH * self.nx)
            iy = int((src.y + RIVER_WIDTH/2) / RIVER_WIDTH * self.ny)
            ix = np.clip(ix, 0, self.nx - 1)
            iy = np.clip(iy, 0, self.ny - 1)

            # Gaussian plume injection kernel (3x3 spread)
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    ni, nj = ix + di, iy + dj
                    if 0 <= ni < self.nx and 0 <= nj < self.ny:
                        weight = np.exp(-(di**2 + dj**2) / 2.0)
                        c_type = src.contaminant_type
                        # Flux = Q * C / (dx*dy*H_mean)
                        flux = src.flow_rate * src.concentration * self.dt / (
                            self.dx * self.dy * 2.0) * weight
                        self.concentrations[c_type][ni, nj] += flux

            # Pulsed release for realism (industrial discharge cycles)
            if 'industrial' in src.name.lower():
                pulse = 0.5 * (1 + np.sin(2 * np.pi * t / 3600.0))
                self.concentrations[src.contaminant_type][ix, iy] *= (1 + 0.3 * pulse)

    def step(self, sources: List[ContaminantSource]):
        """Advance one time step using explicit upwind + central diff scheme"""
        self.add_sources(sources, self.time)

        for c_type, C in self.concentrations.items():
            lam = self.decay_rates[c_type]
            C_new = C.copy()

            # Interior points only
            i = slice(1, self.nx - 1)
            j = slice(1, self.ny - 1)

            u = self.u[i, j]
            v = self.v[i, j]

            # Upwind advection (x-direction)
            adv_x = np.where(
                u > 0,
                u * (C[i, j] - C[0:self.nx-2, j]) / self.dx,
                u * (C[2:self.nx,   j] - C[i, j]) / self.dx
            )
            # Upwind advection (y-direction)
            adv_y = np.where(
                v > 0,
                v * (C[i, j] - C[i, 0:self.ny-2]) / self.dy,
                v * (C[i, 2:self.ny]   - C[i, j]) / self.dy
            )
            # Central difference diffusion
            diff_x = self.Dx * (C[2:self.nx, j] - 2*C[i, j] + C[0:self.nx-2, j]) / self.dx**2
            diff_y = self.Dy * (C[i, 2:self.ny] - 2*C[i, j] + C[i, 0:self.ny-2]) / self.dy**2

            # Decay + update
            C_new[i, j] = C[i, j] + self.dt * (
                -adv_x - adv_y + diff_x + diff_y - lam * C[i, j]
            )

            # Inflow BC (upstream = background)
            C_new[0, :] = self.background[c_type]
            # Outflow BC (zero gradient)
            C_new[-1, :] = C_new[-2, :]
            # Reflective bank BCs
            C_new[:, 0]  = C_new[:, 1]
            C_new[:, -1] = C_new[:, -2]

            # Clip to physical bounds
            C_new = np.clip(C_new, 0.0, 1e4)
            self.concentrations[c_type] = C_new

        self.time += self.dt
        self.step_count += 1

    def get_total_concentration(self):
        """Weighted sum of all contaminant types"""
        weights = {
            'heavy_metal': 10.0,   # high toxicity weight
            'organic':      1.0,
            'nutrient':     2.0,
            'pathogen':     5.0,
        }
        total = np.zeros((self.nx, self.ny))
        for c_type, C in self.concentrations.items():
            total += weights[c_type] * C
        return total

    def probe(self, x_m: float, y_m: float) -> dict:
        """Read all contaminant concentrations at a spatial location"""
        ix = int(np.clip(x_m / RIVER_LENGTH * self.nx, 0, self.nx - 1))
        iy = int(np.clip((y_m + RIVER_WIDTH/2) / RIVER_WIDTH * self.ny, 0, self.ny - 1))
        result = {}
        for c_type, C in self.concentrations.items():
            result[c_type] = float(C[ix, iy])
        result['total_weighted'] = float(self.get_total_concentration()[ix, iy])
        result['x_m'] = x_m
        result['y_m'] = y_m
        result['river_km'] = x_m / 1000.0
        result['time_s']  = self.time
        return result


# ──────────────────────────────────────────────────────────────────────────────
# MONITORING STATION MANAGER
# ──────────────────────────────────────────────────────────────────────────────

def setup_monitoring_stations() -> List[MonitoringStation]:
    stations = [
        MonitoringStation("Upstream Reference",    200,   0, 0.2, [], 0.5,  2.0),
        MonitoringStation("Industrial Outfall",    500,  10, 0.5, [], 1.0,  5.0),
        MonitoringStation("Agricultural Zone",     900, -15, 0.9, [], 2.0,  8.0),
        MonitoringStation("Municipal Intake",     1200,   5, 1.2, [], 1.5,  6.0),
        MonitoringStation("Wetland Buffer",       1500, -20, 1.5, [], 3.0, 10.0),
        MonitoringStation("Downstream Gauge",     1800,   0, 1.8, [], 1.0,  4.0),
    ]
    return stations


def setup_contaminant_sources() -> List[ContaminantSource]:
    sources = [
        ContaminantSource(
            name="Industrial Effluent (Factory A)",
            x=480, y=12,
            concentration=45.0,
            flow_rate=0.8,
            contaminant_type='heavy_metal',
            color=(0.9, 0.1, 0.1)
        ),
        ContaminantSource(
            name="Agricultural Runoff",
            x=850, y=-18,
            concentration=12.0,
            flow_rate=2.5,
            contaminant_type='nutrient',
            color=(0.2, 0.8, 0.2)
        ),
        ContaminantSource(
            name="Sewage Overflow",
            x=1100, y=8,
            concentration=28.0,
            flow_rate=0.4,
            contaminant_type='pathogen',
            color=(0.8, 0.5, 0.1)
        ),
        ContaminantSource(
            name="Chemical Spill (Organic)",
            x=650, y=0,
            concentration=60.0,
            flow_rate=0.2,
            contaminant_type='organic',
            color=(0.6, 0.0, 0.8)
        ),
        ContaminantSource(
            name="Mining Leachate",
            x=1400, y=-25,
            concentration=20.0,
            flow_rate=1.2,
            contaminant_type='heavy_metal',
            color=(0.7, 0.3, 0.0)
        ),
    ]
    return sources


# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZATION ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class RiverDigitalTwin:
    """Main Digital Twin visualization and simulation controller"""

    def __init__(self):
        print("\n" + "="*70)
        print("  RIVER ECOSYSTEM DIGITAL TWIN — INITIALIZING")
        print("="*70)

        print("  ► Building river geometry...")
        self.geometry   = RiverGeometry()
        self.sources    = setup_contaminant_sources()
        self.stations   = setup_monitoring_stations()

        print("  ► Initializing contaminant transport model...")
        self.transport  = ContaminantTransport(
            velocity_u=self.geometry.velocity_u,
            velocity_v=self.geometry.velocity_v
        )

        print("  ► Warming up physics (spin-up period)...")
        for _ in range(60):
            self.transport.step(self.sources)

        self.paused     = False
        self.active_layer = 'Total Contamination Index'   # what to display
        self.show_streamlines = True
        self.selected_point = None
        self.sim_thread = None
        self._lock = threading.Lock()

        print("  ► Building 3D meshes...")
        self.water_surface, self.bed_mesh, self.depth_nodes = \
            self.geometry.build_surface_mesh()
        self._attach_concentration_to_mesh()

        # Particle positions for flow animation
        rng = np.random.default_rng(1)
        n_particles = 300
        self.particle_x = rng.uniform(0, RIVER_LENGTH, n_particles)
        self.particle_y = rng.uniform(-RIVER_WIDTH/2 * 0.9, RIVER_WIDTH/2 * 0.9, n_particles)
        self.particle_age = rng.uniform(0, 1, n_particles)

        print("  ► Launching visualization...\n")

    def _attach_concentration_to_mesh(self):
        """Map concentration field to water surface mesh as scalar array"""
        C_total = self.transport.get_total_concentration()
        C_hm    = self.transport.concentrations['heavy_metal']
        C_nu    = self.transport.concentrations['nutrient']
        C_pa    = self.transport.concentrations['pathogen']
        C_or    = self.transport.concentrations['organic']
        vel_mag = np.sqrt(self.geometry.velocity_u**2 + self.geometry.velocity_v**2)

        # Interpolate from cell-centers (NX×NY) to nodes (NX+1)×(NY+1)
        def interp_to_nodes(field_cells):
            """
            Bilinear cell-to-node interpolation.
            Input:  (NX, NY)       — cell-centred values
            Output: (NX+1, NY+1)  — node values  (size = 8241 for 200×40 grid)

            Strategy: pad the cell array by 1 on every side with edge replication,
            giving (NX+2)×(NY+2), then average the 4 cells that surround each node.
            Node (i,j) sits at the corner shared by cells (i-1,j-1),(i,j-1),(i-1,j),(i,j)
            in the padded array → indices [i:i+1, j:j+1] after padding.
            """
            # Pad: shape becomes (NX+2, NY+2)
            p = np.pad(field_cells, ((1, 1), (1, 1)), mode='edge')
            # Each node (i,j) for i in [0..NX], j in [0..NY]  ← (NX+1)×(NY+1) nodes
            # is the average of p[i,j], p[i+1,j], p[i,j+1], p[i+1,j+1]
            node = 0.25 * (p[0:NX+1, 0:NY+1] + p[1:NX+2, 0:NY+1] +
                           p[0:NX+1, 1:NY+2] + p[1:NX+2, 1:NY+2])
            # node.shape == (NX+1, NY+1) == (201, 41) ✓
            return node

        c_total_nodes = interp_to_nodes(C_total).ravel()
        c_hm_nodes    = interp_to_nodes(C_hm).ravel()
        c_nu_nodes    = interp_to_nodes(C_nu).ravel()
        c_pa_nodes    = interp_to_nodes(C_pa).ravel()
        c_or_nodes    = interp_to_nodes(C_or).ravel()
        vel_nodes     = interp_to_nodes(vel_mag).ravel()
        depth_nodes   = self.depth_nodes.ravel()

        # PyVista structured grid has points ordered Y-first
        # We need to reorder from [NX+1 × NY+1] to [NY+1 × NX+1]
        def reorder(arr):
            return arr.reshape(NX+1, NY+1).T.ravel()

        self.water_surface.point_data['Total Contamination Index'] = reorder(c_total_nodes)
        self.water_surface.point_data['Heavy Metals [mg/L]']       = reorder(c_hm_nodes)
        self.water_surface.point_data['Nutrients [mg/L]']          = reorder(c_nu_nodes)
        self.water_surface.point_data['Pathogens [mg/L]']          = reorder(c_pa_nodes)
        self.water_surface.point_data['Organics [mg/L]']           = reorder(c_or_nodes)
        self.water_surface.point_data['Flow Velocity [m/s]']       = reorder(vel_nodes)
        self.water_surface.point_data['Water Depth [m]']           = reorder(depth_nodes)

    def _update_particles(self, dt_real=0.1):
        """Advect flow visualization particles"""
        for i in range(len(self.particle_x)):
            ix = int(np.clip(self.particle_x[i] / RIVER_LENGTH * NX, 0, NX-1))
            iy = int(np.clip((self.particle_y[i] + RIVER_WIDTH/2) / RIVER_WIDTH * NY, 0, NY-1))

            u_local = self.geometry.velocity_u[ix, iy]
            v_local = self.geometry.velocity_v[ix, iy]

            self.particle_x[i] += u_local * dt_real * 10   # visual speed scale
            self.particle_y[i] += v_local * dt_real * 10

            # Reset particle if it exits the domain
            if (self.particle_x[i] > RIVER_LENGTH * 0.99 or
                self.particle_x[i] < 0 or
                abs(self.particle_y[i]) > RIVER_WIDTH/2 * 0.95):
                rng = np.random.default_rng(i + int(time.time() * 1000) % 10000)
                self.particle_x[i] = rng.uniform(0, RIVER_LENGTH * 0.1)
                self.particle_y[i] = rng.uniform(-RIVER_WIDTH/2 * 0.7, RIVER_WIDTH/2 * 0.7)

        # Build particle polydata
        z_vals = np.full(len(self.particle_x), 0.15)  # slightly above water surface
        pts = np.column_stack([self.particle_y, self.particle_x, z_vals])
        particle_cloud = pv.PolyData(pts)
        return particle_cloud

    def _build_source_markers(self):
        """Create 3D cylinder markers for contamination sources"""
        actors = []
        for src in self.sources:
            # Vertical cylinder at source location
            cyl = pv.Cylinder(
                center=(src.y, src.x, 1.5),
                direction=(0, 0, 1),
                radius=2.0,
                height=4.0,
                resolution=16
            )
            actors.append((cyl, src.color, src.name))

            # Cone on top (discharge indicator)
            cone = pv.Cone(
                center=(src.y, src.x, 4.0),
                direction=(0, 0, -1),
                radius=3.0,
                height=2.0,
                resolution=16
            )
            actors.append((cone, src.color, src.name))
        return actors

    def _build_station_markers(self):
        """Create monitoring station markers"""
        actors = []
        for st in self.stations:
            # Get current reading
            reading = self.transport.probe(st.x, st.y)['total_weighted']

            # Color by alert level
            if reading > st.threshold_critical:
                col = (1.0, 0.0, 0.0)   # RED - critical
            elif reading > st.threshold_warning:
                col = (1.0, 0.65, 0.0)  # ORANGE - warning
            else:
                col = (0.0, 0.9, 0.3)   # GREEN - safe

            sphere = pv.Sphere(radius=2.5, center=(st.y, st.x, 0.5))
            actors.append((sphere, col, st.name))

            # Vertical pole
            line = pv.Line(
                pointa=(st.y, st.x, 0.5),
                pointb=(st.y, st.x, 8.0)
            )
            actors.append((line, col, st.name + "_pole"))
        return actors

    def _concentration_colormap(self):
        """Custom colormap: blue=clean → yellow → red=contaminated"""
        return 'RdYlGn_r'   # Red-Yellow-Green reversed

    def _format_hud_text(self):
        """Build the HUD overlay text"""
        t = self.transport.time
        hours   = int(t // 3600)
        minutes = int((t % 3600) // 60)
        seconds = int(t % 60)

        lines = [
            "╔═══════════════════════════════════════════════╗",
            "║   RIVER DIGITAL TWIN — CONTAMINANT TRACKER   ║",
            "╚═══════════════════════════════════════════════╝",
            f"  Simulation Time: {hours:02d}h {minutes:02d}m {seconds:02d}s",
            f"  Time Step:       {self.transport.dt:.2f} s",
            f"  Grid:            {NX} × {NY} cells",
            f"  River:           {RIVER_LENGTH/1000:.1f} km × {RIVER_WIDTH:.0f} m",
            "",
            "─── MONITORING STATIONS ────────────────────────",
        ]

        for st in self.stations:
            reading = self.transport.probe(st.x, st.y)
            total   = reading['total_weighted']
            hm      = reading['heavy_metal']
            nu      = reading['nutrient']

            if total > st.threshold_critical:
                status = "⚠ CRITICAL"
            elif total > st.threshold_warning:
                status = "! WARNING "
            else:
                status = "✓ SAFE    "

            lines.append(f"  {status} | {st.name[:22]:<22}")
            lines.append(f"           Total:{total:6.2f}  HM:{hm:.3f}  Nu:{nu:.3f}")

        lines += [
            "",
            "─── ACTIVE SOURCES ─────────────────────────────",
        ]
        for src in self.sources:
            if src.active:
                lines.append(f"  • {src.name[:38]:<38}")

        lines += [
            "",
            "─── CONTROLS ───────────────────────────────────",
            "  [SPACE] Pause/Resume    [R] Reset View",
            "  [1] Heavy Metals  [2] Nutrients",
            "  [3] Pathogens     [4] Organics  [5] Total",
            "  [P] Probe Point   [S] Streamlines",
            "  [Q] Quit",
        ]

        if self.paused:
            lines.append("")
            lines.append("  ⏸  SIMULATION PAUSED")

        if self.selected_point:
            sp = self.selected_point
            lines += [
                "",
                "─── PROBE READING ──────────────────────────────",
                f"  Location: Km {sp['river_km']:.2f}, Y={sp['y_m']:.1f}m",
                f"  Heavy Metals:  {sp['heavy_metal']:.4f} mg/L",
                f"  Nutrients:     {sp['nutrient']:.4f} mg/L",
                f"  Pathogens:     {sp['pathogen']:.4f} mg/L",
                f"  Organics:      {sp['organic']:.4f} mg/L",
                f"  ─────────────────────────────────",
                f"  Weighted Index:{sp['total_weighted']:.3f}",
            ]

        return "\n".join(lines)

    def run(self):
        """Main visualization loop"""

        # ── PyVista Plotter Setup ──
        pv.set_plot_theme('dark')

        pl = pv.Plotter(
            title="River Ecosystem Digital Twin — Real-Time Contaminant Tracking",
            window_size=[1600, 900]
        )

        # Background gradient (sky)
        pl.set_background([0.05, 0.08, 0.15], top=[0.15, 0.22, 0.35])

        # ── Water Surface ──
        layer_name = 'Total Contamination Index'
        water_actor = pl.add_mesh(
            self.water_surface,
            scalars=layer_name,
            cmap=self._concentration_colormap(),
            show_scalar_bar=True,
            scalar_bar_args={
                'title': 'Contamination\nIndex',
                'vertical': True,
                'position_x': 0.85,
                'position_y': 0.3,
                'height': 0.4,
                'width': 0.04,
                'color': 'white',
            },
            opacity=0.80,
            smooth_shading=True,
            name='water_surface',
            clim=[0, 30]
        )

        # ── River Bed ──
        pl.add_mesh(
            self.bed_mesh,
            color=[0.45, 0.35, 0.22],
            opacity=0.95,
            smooth_shading=True,
            name='river_bed'
        )

        # ── Terrain Banks ──
        self._add_terrain_banks(pl)

        # ── Source Markers ──
        for mesh, col, name in self._build_source_markers():
            pl.add_mesh(mesh, color=col, opacity=0.9, name=f'src_{name}')

        # ── Monitoring Station Markers ──
        for mesh, col, name in self._build_station_markers():
            if name.endswith('_pole'):
                pl.add_mesh(mesh, color=col, line_width=3, name=f'st_{name}')
            else:
                pl.add_mesh(mesh, color=col, opacity=1.0, name=f'st_{name}')

        # ── Flow Particles ──
        p_cloud = self._update_particles()
        pl.add_mesh(
            p_cloud,
            color='cyan',
            point_size=4,
            render_points_as_spheres=True,
            opacity=0.6,
            name='particles'
        )

        # ── HUD Text Overlay ──
        hud = pl.add_text(
            self._format_hud_text(),
            position='upper_left',
            font_size=7,
            color='white',
            font='courier',
            name='hud_text'
        )

        # ── Compass / Scale Bar ──
        pl.add_text(
            f"▶ Flow Direction  |  Grid: {NX}×{NY}  |  Δx={RIVER_LENGTH/NX:.1f}m  Δy={RIVER_WIDTH/NY:.1f}m",
            position='lower_edge',
            font_size=8,
            color='lightblue',
            name='scale_bar'
        )

        # ── Camera Setup (isometric VR-style view) ──
        pl.camera_position = [
            (RIVER_WIDTH/2 * 3, -RIVER_LENGTH * 0.3, RIVER_LENGTH * 0.25),
            (0, RIVER_LENGTH / 2, 0),
            (0, 0, 1)
        ]
        pl.enable_terrain_style(mouse_wheel_zooms=0.3)

        # ── Keyboard Callbacks ──
        layer_map = {
            '1': ('Heavy Metals [mg/L]',  [0, 5]),
            '2': ('Nutrients [mg/L]',     [0, 3]),
            '3': ('Pathogens [mg/L]',     [0, 8]),
            '4': ('Organics [mg/L]',      [0, 10]),
            '5': ('Total Contamination Index', [0, 30]),
        }

        def make_layer_cb(lname, clim):
            def cb():
                self.active_layer = lname
                pl.update_scalars(lname, mesh=self.water_surface, render=False)
                water_actor.mapper.scalar_range = clim
                pl.render()
            return cb

        for key, (lname, clim) in layer_map.items():
            pl.add_key_event(key, make_layer_cb(lname, clim))

        def toggle_pause():
            self.paused = not self.paused
        pl.add_key_event('space', toggle_pause)

        def toggle_streamlines():
            self.show_streamlines = not self.show_streamlines
        pl.add_key_event('s', toggle_streamlines)

        # ── Timer Callback (main update loop) ──
        sim_steps_per_frame = 4
        frame_counter = [0]
        last_station_update = [0]

        def update_callback():
            if not self.paused:
                # Advance physics
                for _ in range(sim_steps_per_frame):
                    self.transport.step(self.sources)

                # Update concentration on mesh
                self._attach_concentration_to_mesh()
                pl.update_scalars(
                    self.active_layer,
                    mesh=self.water_surface,
                    render=False
                )

            # Always update particles
            p_new = self._update_particles(dt_real=0.08)
            pl.update(1)
            try:
                pl.remove_actor('particles')
                pl.add_mesh(
                    p_new,
                    color='cyan',
                    point_size=4,
                    render_points_as_spheres=True,
                    opacity=0.6,
                    name='particles'
                )
            except Exception:
                pass

            # Update monitoring station colors every 5 frames
            if frame_counter[0] % 5 == 0:
                for mesh, col, name in self._build_station_markers():
                    if not name.endswith('_pole'):
                        try:
                            pl.remove_actor(f'st_{name}')
                            pl.add_mesh(mesh, color=col, opacity=1.0,
                                        name=f'st_{name}')
                        except Exception:
                            pass

            # Update HUD every frame
            try:
                pl.remove_actor('hud_text')
                pl.add_text(
                    self._format_hud_text(),
                    position='upper_left',
                    font_size=7,
                    color='white',
                    font='courier',
                    name='hud_text'
                )
            except Exception:
                pass

            frame_counter[0] += 1

        pl.add_timer_event(callback=update_callback, max_steps=int(1000 / UPDATE_FPS),duration=1000)

        # ── Pick Callback (probe concentration at clicked point) ──
        def pick_callback(point):
            if point is None:
                return
            # point is in PyVista's (y, x, z) order due to our mesh construction
            x_world = float(point[1])
            y_world = float(point[0])
            self.selected_point = self.transport.probe(x_world, y_world)

            # Visual marker at picked point
            sphere = pv.Sphere(radius=3.0, center=point)
            try:
                pl.remove_actor('probe_marker')
            except Exception:
                pass
            pl.add_mesh(sphere, color='yellow', opacity=0.9, name='probe_marker')

        pl.enable_surface_point_picking(
            callback=pick_callback,
            show_message=True,
            color='yellow',
            point_size=15,
            tolerance=0.025,
            pickable_window=True
        )

        # ── Add Legend ──
        legend_entries = [
            ["Heavy Metal Source",   "red"],
            ["Nutrient / Ag Runoff", "green"],
            ["Pathogen Source",      "orange"],
            ["Organic Contaminant",  "purple"],
            ["Monitoring Station",   "cyan"],
        ]
        pl.add_legend(
            labels=legend_entries,
            bcolor=(0.0, 0.0, 0.0),
            border=True,
            size=(0.18, 0.15),
            loc='lower right'
        )

        # ── Show ──
        print("  Visualization window launching...")
        print("  CONTROLS:")
        print("    Mouse drag = Rotate | Scroll = Zoom | Shift+drag = Pan")
        print("    [SPACE]=Pause  [1-5]=Layer  [P]=Probe  [R]=Reset  [Q]=Quit")
        print()

        pl.show(auto_close=False, interactive=True)

    def _add_terrain_banks(self, pl):
        """Add green riverbanks and terrain on both sides"""
        # Left bank
        x = np.linspace(0, RIVER_LENGTH, 50)
        y_left  = np.linspace(RIVER_WIDTH/2, RIVER_WIDTH/2 + 60, 10)
        y_right = np.linspace(-RIVER_WIDTH/2 - 60, -RIVER_WIDTH/2, 10)

        rng = np.random.default_rng(7)

        for y_arr, side in [(y_left, 'L'), (y_right, 'R')]:
            xx, yy = np.meshgrid(x, y_arr, indexing='ij')
            # Terrain elevation
            zz = 0.5 + 0.8 * np.abs(yy - (RIVER_WIDTH/2 * (1 if side=='L' else -1))) / 60.0
            zz += 0.3 * np.sin(xx / 100) * np.cos(yy / 30)
            zz += 0.1 * rng.standard_normal(zz.shape)

            bank = pv.StructuredGrid()
            bank.dimensions = [len(y_arr), len(x), 1]
            pts = np.column_stack([yy.ravel(), xx.ravel(), zz.ravel()])
            bank.points = pts
            pl.add_mesh(bank, color=[0.25, 0.55, 0.18], opacity=0.95,
                        smooth_shading=True, name=f'bank_{side}')


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def check_dependencies():
    missing = []
    for pkg in ['pyvista', 'numpy', 'scipy']:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"   Install with:  pip install {' '.join(missing)}")
        return False
    return True


if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║        RIVER ECOSYSTEM DIGITAL TWIN                                  ║
║        Real-Time Contaminant Transport Simulation                    ║
║                                                                      ║
║  Physics:  2D Advection-Diffusion-Reaction PDE                       ║
║  Solver:   Explicit Upwind Finite Difference                         ║
║  Domain:   2 km river × 80 m width × 200×40 grid                    ║
║  Sources:  Industrial | Agricultural | Sewage | Chemical             ║
║  Stations: 6 Monitoring Points with threshold alerts                 ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    if not check_dependencies():
        exit(1)

    twin = RiverDigitalTwin()
    twin.run()
