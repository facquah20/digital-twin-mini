"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         RIVER ECOSYSTEM DIGITAL TWIN — CONTAMINANT TRANSPORT MODEL          ║
║         Multi-Threaded Architecture | PyQt6 + PyVista                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  THREAD ARCHITECTURE:                                                        ║
║                                                                              ║
║  ┌─────────────────┐   Lock-protected    ┌──────────────────────────────┐   ║
║  │  PhysicsThread  │ ──── SharedState ──▶│  Qt Main Thread (UI + VTK)   │   ║
║  │  (daemon)       │                     │  • PyVista 3D viewport        │   ║
║  │  • PDE solver   │   Qt Signals        │  • PyQt6 control panels       │   ║
║  │  • ~60 Hz       │ ──────────────────▶ │  • Live charts (pyqtgraph)    │   ║
║  └─────────────────┘                     │  • Station alert table        │   ║
║                                          └──────────────────────────────┘   ║
║                                                                              ║
║  PHYSICS:  ∂C/∂t + u·∇C = D·∇²C − λC + S(x,y,t)                           ║
║  SOLVER:   Explicit upwind FD | CFL-stable | 200×40 grid                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

INSTALL:
    pip install numpy scipy pyvista pyvistaqt PyQt6 pyqtgraph

RUN:
    python river_digital_twin.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
import time
import threading
import csv
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

# ── Numerical ─────────────────────────────────────────────────────────────────
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

# ── Qt ────────────────────────────────────────────────────────────────────────
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QPushButton, QComboBox, QSlider, QGroupBox,
    QTableWidget, QTableWidgetItem, QTabWidget, QStatusBar, QFileDialog,
    QCheckBox, QProgressBar, QFrame, QGridLayout, QSpinBox, QDoubleSpinBox,
    QMessageBox, QToolBar, QSizePolicy
)
from PyQt6.QtCore import (
    Qt, QTimer, pyqtSignal, QObject, QThread, QMutex, QMutexLocker
)
from PyQt6.QtGui import QColor, QFont, QAction, QIcon, QPalette

# ── PyVista ───────────────────────────────────────────────────────────────────
import pyvista as pv
from pyvistaqt import QtInteractor

# ── PyQtGraph (live charts) ───────────────────────────────────────────────────
try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
    pg.setConfigOptions(antialias=True, background='#0d1117', foreground='#c9d1d9')
except ImportError:
    HAS_PYQTGRAPH = False

import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# DOMAIN CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

RIVER_LENGTH = 2000.0   # m
RIVER_WIDTH  = 80.0     # m
RIVER_DEPTH  = 5.0      # m (max)
NX           = 200      # longitudinal cells
NY           = 40       # transverse cells
DT_BASE      = 0.5      # base time step [s]
PHYSICS_HZ   = 60       # physics thread target rate [Hz]
RENDER_HZ    = 20       # render timer rate [Hz]
HISTORY_LEN  = 300      # samples kept per station chart

CONTAMINANT_TYPES = ['heavy_metal', 'organic', 'nutrient', 'pathogen']

DECAY_RATES = {
    'heavy_metal': 1e-6,
    'organic':     2e-5,
    'nutrient':    5e-6,
    'pathogen':    3e-4,
}

BACKGROUND_CONC = {
    'heavy_metal': 0.002,
    'organic':     0.5,
    'nutrient':    0.8,
    'pathogen':    0.1,
}

TOXICITY_WEIGHTS = {
    'heavy_metal': 10.0,
    'organic':      1.0,
    'nutrient':     2.0,
    'pathogen':     5.0,
}

# ──────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ContaminantSource:
    name: str
    x: float
    y: float
    concentration: float   # mg/L
    flow_rate: float       # m³/s
    contaminant_type: str
    color: Tuple[float, float, float]
    active: bool = True

@dataclass
class MonitoringStation:
    name: str
    x: float
    y: float
    river_km: float
    threshold_warning:  float
    threshold_critical: float
    readings: List[float] = field(default_factory=list)   # history buffer

@dataclass
class SimSnapshot:
    """
    Immutable snapshot passed from physics thread → render thread.
    Deep-copied so the physics thread can keep writing without locks held.
    """
    concentrations: Dict[str, np.ndarray]   # {type: NX×NY array}
    total:          np.ndarray              # NX×NY weighted sum
    sim_time:       float
    step_count:     int
    station_data:   List[dict]             # one dict per station
    probe:          Optional[dict]         # last probe reading, or None


# ──────────────────────────────────────────────────────────────────────────────
# SHARED STATE  (physics writes | render reads — guarded by a mutex)
# ──────────────────────────────────────────────────────────────────────────────

class SharedState:
    def __init__(self):
        self._mutex   = QMutex()
        self._snap: Optional[SimSnapshot] = None
        self.paused   = False
        self.speed    = 4           # physics steps per physics-thread tick
        self.active_source_toggles: Dict[str, bool] = {}

    # ── Physics thread writes ─────────────────────────────────────────────────
    def put_snapshot(self, snap: SimSnapshot):
        locker = QMutexLocker(self._mutex)
        self._snap = snap

    # ── Render thread reads ───────────────────────────────────────────────────
    def get_snapshot(self) -> Optional[SimSnapshot]:
        locker = QMutexLocker(self._mutex)
        return self._snap


# ──────────────────────────────────────────────────────────────────────────────
# RIVER GEOMETRY
# ──────────────────────────────────────────────────────────────────────────────

class RiverGeometry:
    def __init__(self):
        self.dx = RIVER_LENGTH / NX
        self.dy = RIVER_WIDTH  / NY
        self.depth_map    = self._bathymetry()
        self.velocity_u, self.velocity_v = self._velocity_field()
        self.water_surface, self.bed_mesh, self.depth_nodes = self._build_meshes()

    # ── Bathymetry ────────────────────────────────────────────────────────────
    def _bathymetry(self) -> np.ndarray:
        xx, yy = np.meshgrid(
            np.linspace(0, RIVER_LENGTH, NX),
            np.linspace(-RIVER_WIDTH/2, RIVER_WIDTH/2, NY),
            indexing='ij'
        )
        y_norm       = yy / (RIVER_WIDTH / 2)
        parabolic    = RIVER_DEPTH * (1.0 - y_norm**2)
        pool_riffle  = 0.8 + 0.4 * np.sin(2*np.pi*xx / 400.0)
        rng          = np.random.default_rng(42)
        roughness    = gaussian_filter(0.3 * rng.standard_normal((NX, NY)), sigma=3)
        meander_off  = 0.3 * np.sin(2*np.pi*xx / 600.0)
        meander_dep  = np.clip(RIVER_DEPTH * (1.0 - (y_norm - meander_off)**2), 0.5, RIVER_DEPTH)
        depth        = (0.6*parabolic + 0.4*meander_dep) * pool_riffle + roughness*0.15
        return np.clip(depth, 0.3, RIVER_DEPTH * 1.2)

    # ── Manning velocity field ────────────────────────────────────────────────
    def _velocity_field(self) -> Tuple[np.ndarray, np.ndarray]:
        H  = self.depth_map
        S0 = 0.0005
        n  = 0.035
        U  = gaussian_filter((H**(2/3) * np.sqrt(S0)) / n, sigma=2)
        xx = np.linspace(0, RIVER_LENGTH, NX)[:, None]
        yy = np.linspace(-RIVER_WIDTH/2, RIVER_WIDTH/2, NY)[None, :]
        V  = gaussian_filter(0.15 * U * np.cos(2*np.pi*xx/600.0) * (yy/RIVER_WIDTH), sigma=2)
        bank = np.abs(yy) > (RIVER_WIDTH/2 * 0.92)
        U   *= (1 - 0.95*bank); V *= (1 - 0.95*bank)
        return U, V

    # ── PyVista meshes ────────────────────────────────────────────────────────
    def _build_meshes(self):
        x  = np.linspace(0, RIVER_LENGTH, NX+1)
        y  = np.linspace(-RIVER_WIDTH/2, RIVER_WIDTH/2, NY+1)
        xx, yy = np.meshgrid(x, y, indexing='ij')

        # Interpolate depth to nodes
        xc = np.linspace(0, RIVER_LENGTH, NX)
        yc = np.linspace(-RIVER_WIDTH/2, RIVER_WIDTH/2, NY)
        depth_nodes = RegularGridInterpolator(
            (xc, yc), self.depth_map, method='linear',
            bounds_error=False, fill_value=1.0
        )(np.column_stack([xx.ravel(), yy.ravel()])).reshape(NX+1, NY+1)

        wave_z = 0.05 * np.sin(2*np.pi*xx/50.0) * np.cos(2*np.pi*yy/20.0)

        def _make_grid(z_arr):
            g = pv.StructuredGrid()
            g.dimensions = [NY+1, NX+1, 1]
            pts = np.column_stack([xx.ravel(), yy.ravel(), z_arr.ravel()])
            g.points = pts[:, [1, 0, 2]]
            return g

        surface = _make_grid(wave_z)
        bed      = _make_grid(-depth_nodes)
        return surface, bed, depth_nodes


# ──────────────────────────────────────────────────────────────────────────────
# CONTAMINANT TRANSPORT ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class ContaminantTransport:
    def __init__(self, geo: RiverGeometry):
        self.geo  = geo
        self.dx   = geo.dx
        self.dy   = geo.dy
        self.u    = geo.velocity_u
        self.v    = geo.velocity_v
        self.time = 0.0
        self.step_count = 0

        U_mean = np.mean(np.abs(self.u))
        self.Dx = 5.0 + 0.5 * U_mean * self.dx
        self.Dy = 0.15  * U_mean * self.dy

        # CFL-safe dt
        self.dt = min(
            DT_BASE,
            0.4 * self.dx / max(np.max(np.abs(self.u)), 1e-6),
            0.4 * self.dy / max(np.max(np.abs(self.v)), 1e-6),
        )

        # Concentration fields
        self.C: Dict[str, np.ndarray] = {}
        for ct in CONTAMINANT_TYPES:
            self.C[ct] = np.full((NX, NY), BACKGROUND_CONC[ct])

    # ── Source injection ──────────────────────────────────────────────────────
    def _inject_sources(self, sources: List[ContaminantSource]):
        for src in sources:
            if not src.active:
                continue
            ix = int(np.clip(src.x / RIVER_LENGTH * NX, 0, NX-1))
            iy = int(np.clip((src.y + RIVER_WIDTH/2) / RIVER_WIDTH * NY, 0, NY-1))
            pulse = 1.0
            if 'industrial' in src.name.lower():
                pulse += 0.3 * np.sin(2*np.pi*self.time / 3600.0)
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    ni, nj = ix+di, iy+dj
                    if 0 <= ni < NX and 0 <= nj < NY:
                        w    = np.exp(-(di**2 + dj**2) / 2.0)
                        flux = (src.flow_rate * src.concentration * self.dt
                                / (self.dx * self.dy * 2.0)) * w * pulse
                        self.C[src.contaminant_type][ni, nj] += flux

    # ── One PDE time step ─────────────────────────────────────────────────────
    def step(self, sources: List[ContaminantSource]):
        self._inject_sources(sources)
        dt = self.dt
        dx, dy = self.dx, self.dy

        i = slice(1, NX-1)
        j = slice(1, NY-1)
        u = self.u[i, j]
        v = self.v[i, j]

        for ct in CONTAMINANT_TYPES:
            C   = self.C[ct]
            lam = DECAY_RATES[ct]

            adv_x = np.where(u > 0,
                u * (C[i,j] - C[0:NX-2, j]) / dx,
                u * (C[2:NX, j] - C[i,j]) / dx)
            adv_y = np.where(v > 0,
                v * (C[i,j] - C[i, 0:NY-2]) / dy,
                v * (C[i, 2:NY] - C[i,j]) / dy)
            dif_x = self.Dx * (C[2:NX,j] - 2*C[i,j] + C[0:NX-2,j]) / dx**2
            dif_y = self.Dy * (C[i,2:NY] - 2*C[i,j] + C[i,0:NY-2]) / dy**2

            C_new      = C.copy()
            C_new[i,j] = C[i,j] + dt*(-adv_x - adv_y + dif_x + dif_y - lam*C[i,j])
            C_new[0,:]  = BACKGROUND_CONC[ct]   # inflow
            C_new[-1,:] = C_new[-2,:]            # outflow
            C_new[:,0]  = C_new[:,1]             # left bank
            C_new[:,-1] = C_new[:,-2]            # right bank
            self.C[ct]  = np.clip(C_new, 0.0, 1e4)

        self.time      += dt
        self.step_count += 1

    # ── Derived fields ────────────────────────────────────────────────────────
    def total(self) -> np.ndarray:
        return sum(TOXICITY_WEIGHTS[ct] * self.C[ct] for ct in CONTAMINANT_TYPES)

    def probe(self, x_m: float, y_m: float) -> dict:
        ix = int(np.clip(x_m / RIVER_LENGTH * NX, 0, NX-1))
        iy = int(np.clip((y_m + RIVER_WIDTH/2) / RIVER_WIDTH * NY, 0, NY-1))
        out = {ct: float(self.C[ct][ix, iy]) for ct in CONTAMINANT_TYPES}
        out['total_weighted'] = float(self.total()[ix, iy])
        out['x_m'] = x_m; out['y_m'] = y_m
        out['river_km'] = x_m / 1000.0
        out['time_s']   = self.time
        return out

    def snapshot_concentrations(self) -> Dict[str, np.ndarray]:
        return {ct: self.C[ct].copy() for ct in CONTAMINANT_TYPES}


# ──────────────────────────────────────────────────────────────────────────────
# PHYSICS THREAD
# ──────────────────────────────────────────────────────────────────────────────

class PhysicsThread(QThread):
    """
    Dedicated daemon thread running the PDE solver at ~PHYSICS_HZ.
    Never touches Qt widgets — only writes to SharedState via mutex.
    Emits a signal after each batch so the main thread can pull the snapshot.
    """
    snapshot_ready = pyqtSignal()

    def __init__(self, transport: ContaminantTransport,
                 sources: List[ContaminantSource],
                 stations: List[MonitoringStation],
                 shared: SharedState):
        super().__init__()
        self.transport = transport
        self.sources   = sources
        self.stations  = stations
        self.shared    = shared
        self._running  = True
        self.daemon   = True

        # Last probe request (set from UI thread, read here)
        self._probe_request: Optional[Tuple[float,float]] = None
        self._probe_lock = threading.Lock()

    def request_probe(self, x: float, y: float):
        with self._probe_lock:
            self._probe_request = (x, y)

    def stop(self):
        self._running = False

    def run(self):
        tick_s = 1.0 / PHYSICS_HZ
        last   = time.perf_counter()

        while self._running:
            t0 = time.perf_counter()

            if not self.shared.paused:
                steps = self.shared.speed

                # Update source active flags from UI toggles
                for src in self.sources:
                    if src.name in self.shared.active_source_toggles:
                        src.active = self.shared.active_source_toggles[src.name]

                for _ in range(steps):
                    self.transport.step(self.sources)

                # Station readings (append to rolling history)
                station_data = []
                T = self.transport.total()
                for st in self.stations:
                    val = self.transport.probe(st.x, st.y)
                    st.readings.append(val['total_weighted'])
                    if len(st.readings) > HISTORY_LEN:
                        st.readings = st.readings[-HISTORY_LEN:]
                    station_data.append({
                        'name':     st.name,
                        'river_km': st.river_km,
                        'readings': list(st.readings),
                        'current':  val,
                        'warn':     st.threshold_warning,
                        'crit':     st.threshold_critical,
                    })

                # Probe if requested
                probe_result = None
                with self._probe_lock:
                    if self._probe_request:
                        probe_result = self.transport.probe(*self._probe_request)
                        self._probe_request = None

                snap = SimSnapshot(
                    concentrations = self.transport.snapshot_concentrations(),
                    total          = T.copy(),
                    sim_time       = self.transport.time,
                    step_count     = self.transport.step_count,
                    station_data   = station_data,
                    probe          = probe_result,
                )
                self.shared.put_snapshot(snap)
                self.snapshot_ready.emit()

            # Throttle to target Hz
            elapsed = time.perf_counter() - t0
            sleep_t = tick_s - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)


# ──────────────────────────────────────────────────────────────────────────────
# HELPER: cell→node interpolation
# ──────────────────────────────────────────────────────────────────────────────

def cells_to_nodes(field: np.ndarray) -> np.ndarray:
    """NX×NY → (NX+1)×(NY+1) bilinear cell-to-node interpolation."""
    p    = np.pad(field, ((1,1),(1,1)), mode='edge')          # (NX+2)×(NY+2)
    node = 0.25*(p[0:NX+1,0:NY+1] + p[1:NX+2,0:NY+1] +
                 p[0:NX+1,1:NY+2] + p[1:NX+2,1:NY+2])        # (NX+1)×(NY+1)
    return node

def to_pyvista_order(arr: np.ndarray) -> np.ndarray:
    """PyVista StructuredGrid stores points Y-major; transpose accordingly."""
    return arr.reshape(NX+1, NY+1).T.ravel()


# ──────────────────────────────────────────────────────────────────────────────
# 3D VIEWPORT WIDGET
# ──────────────────────────────────────────────────────────────────────────────

LAYER_CONFIG = {
    'Total Contamination Index': ('total',        [0, 30]),
    'Heavy Metals [mg/L]':       ('heavy_metal',  [0, 5]),
    'Nutrients [mg/L]':          ('nutrient',     [0, 3]),
    'Pathogens [mg/L]':          ('pathogen',     [0, 8]),
    'Organics [mg/L]':           ('organic',      [0, 10]),
    'Flow Velocity [m/s]':       ('velocity',     [0, 1.5]),
    'Water Depth [m]':           ('depth',        [0, RIVER_DEPTH]),
}

class ViewportWidget(QWidget):
    probe_requested = pyqtSignal(float, float)   # x_m, y_m

    def __init__(self, geo: RiverGeometry):
        super().__init__()
        self.geo          = geo
        self.active_layer = 'Total Contamination Index'
        self._last_probe  = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)

        # ── PyVista QtInteractor (VTK inside Qt, non-blocking) ───────────────
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter)

        pv.set_plot_theme('dark')
        self.plotter.set_background([0.05, 0.08, 0.15], top=[0.15, 0.22, 0.35])

        self._init_scalars()
        self._build_static_scene()
        self._init_particles()
       

    # ── One-time scene construction ───────────────────────────────────────────
    def _build_static_scene(self):
        pl = self.plotter

        # Water surface (scalars updated every frame)
        self.water_actor = pl.add_mesh(
            self.geo.water_surface,
            scalars='Total Contamination Index',
            cmap='RdYlGn_r',
            show_scalar_bar=True,
            scalar_bar_args=dict(
                title='Contamination\nIndex',
                vertical=True,
                position_x=0.88, position_y=0.25,
                height=0.45, width=0.04,
                color='white', fmt='%.1f',
            ),
            opacity=0.82,
            smooth_shading=True,
            name='water_surface',
            clim=[0, 30],
        )

        # River bed
        pl.add_mesh(self.geo.bed_mesh,
                    color=[0.42, 0.32, 0.20],
                    opacity=0.96, smooth_shading=True, name='river_bed')

        # Banks
        self._add_banks(pl)

        # Scale bar text
        pl.add_text(
            f"▶ Flow  |  {NX}×{NY} grid  |  Δx={RIVER_LENGTH/NX:.0f}m  Δy={RIVER_WIDTH/NY:.1f}m",
            position='lower_edge', font_size=8, color='lightblue', name='scalebar')

        # Camera
        pl.camera_position = [
            ( RIVER_WIDTH*1.8, -RIVER_LENGTH*0.28, RIVER_LENGTH*0.22),
            ( 0,  RIVER_LENGTH/2, 0),
            ( 0,  0, 1),
        ]
        pl.enable_terrain_style()

        # Pick callback — runs in Qt main thread (safe)
        pl.enable_surface_point_picking(
            callback=self._on_pick,
            show_message=False,
            color='yellow',
            point_size=12,
            tolerance=0.025,
        )

    def _on_pick(self, point):
        if point is None:
            return
        x_m = float(point[1])   # PyVista stores our mesh Y-major
        y_m = float(point[0])
        self.probe_requested.emit(x_m, y_m)
        # Draw marker
        sphere = pv.Sphere(radius=3.0, center=point)
        try:    self.plotter.remove_actor('probe_marker')
        except: pass
        self.plotter.add_mesh(sphere, color='yellow', opacity=0.9, name='probe_marker')

    def _add_banks(self, pl):
        rng = np.random.default_rng(7)
        x   = np.linspace(0, RIVER_LENGTH, 50)
        for y_arr, side in [
            (np.linspace( RIVER_WIDTH/2, RIVER_WIDTH/2+60, 10), 'L'),
            (np.linspace(-RIVER_WIDTH/2-60, -RIVER_WIDTH/2, 10), 'R'),
        ]:
            xx, yy = np.meshgrid(x, y_arr, indexing='ij')
            zz = (0.5 + 0.8*np.abs(yy - (RIVER_WIDTH/2*(1 if side=='L' else -1)))/60.0
                  + 0.3*np.sin(xx/100)*np.cos(yy/30)
                  + 0.1*rng.standard_normal(xx.shape))
            bank = pv.StructuredGrid()
            bank.dimensions = [len(y_arr), len(x), 1]
            bank.points = np.column_stack([yy.ravel(), xx.ravel(), zz.ravel()])
            pl.add_mesh(bank, color=[0.22, 0.52, 0.16], smooth_shading=True,
                        name=f'bank_{side}')

    def _init_particles(self):
        rng = np.random.default_rng(1)
        n   = 400
        self.px  = rng.uniform(0, RIVER_LENGTH, n)
        self.py  = rng.uniform(-RIVER_WIDTH/2*0.88, RIVER_WIDTH/2*0.88, n)
        self._add_particle_mesh()

    def _add_particle_mesh(self):
        z   = np.full(len(self.px), 0.18)
        pts = np.column_stack([self.py, self.px, z])
        cloud = pv.PolyData(pts)
        try:    self.plotter.remove_actor('particles')
        except: pass
        self.plotter.add_mesh(cloud, color='deepskyblue', point_size=4,
                              render_points_as_spheres=True, opacity=0.65,
                              name='particles')

    def _init_scalars(self):
        """Populate all scalar arrays on the mesh once (even if zeroed)."""
        dummy = np.zeros(self.geo.water_surface.n_points)
        for key in LAYER_CONFIG:
            self.geo.water_surface.point_data[key] = dummy.copy()
        depth_arr = to_pyvista_order(cells_to_nodes(self.geo.depth_nodes))
        vel_arr   = to_pyvista_order(cells_to_nodes(
            np.sqrt(self.geo.velocity_u**2 + self.geo.velocity_v**2)))
        self.geo.water_surface.point_data['Water Depth [m]']    = depth_arr
        self.geo.water_surface.point_data['Flow Velocity [m/s]'] = vel_arr

    # ── Called by render timer ────────────────────────────────────────────────
    def update_from_snapshot(self, snap: SimSnapshot):
        if snap is None:
            return

        # 1. Update concentration scalars
        total_nodes = to_pyvista_order(cells_to_nodes(snap.total))
        self.geo.water_surface.point_data['Total Contamination Index'] = total_nodes

        for ct in CONTAMINANT_TYPES:
            label = {
                'heavy_metal': 'Heavy Metals [mg/L]',
                'nutrient':    'Nutrients [mg/L]',
                'pathogen':    'Pathogens [mg/L]',
                'organic':     'Organics [mg/L]',
            }[ct]
            self.geo.water_surface.point_data[label] = \
                to_pyvista_order(cells_to_nodes(snap.concentrations[ct]))

        self.plotter.update_scalars(
            self.active_layer,
            mesh=self.geo.water_surface,
            render=False,
        )

        # 2. Advect particles
        self._advect_particles(dt=0.06)

        # 3. Render
        self.plotter.render()

    def _advect_particles(self, dt=0.06):
        geo = self.geo
        ix  = np.clip((self.px / RIVER_LENGTH * NX).astype(int), 0, NX-1)
        iy  = np.clip(((self.py + RIVER_WIDTH/2) / RIVER_WIDTH * NY).astype(int), 0, NY-1)
        u_p = geo.velocity_u[ix, iy]
        v_p = geo.velocity_v[ix, iy]
        self.px += u_p * dt * 12
        self.py += v_p * dt * 12

        # Recycle out-of-bounds particles back to upstream
        rng  = np.random.default_rng(int(time.time()*1000) % 2**31)
        mask = ((self.px > RIVER_LENGTH*0.98) | (self.px < 0) |
                (np.abs(self.py) > RIVER_WIDTH/2*0.93))
        n_reset = mask.sum()
        if n_reset:
            self.px[mask] = rng.uniform(0, RIVER_LENGTH*0.08, n_reset)
            self.py[mask] = rng.uniform(-RIVER_WIDTH/2*0.72, RIVER_WIDTH/2*0.72, n_reset)
        self._add_particle_mesh()

    def set_layer(self, layer_name: str):
        self.active_layer = layer_name
        _, clim = LAYER_CONFIG[layer_name]
        self.plotter.update_scalars(layer_name, mesh=self.geo.water_surface, render=False)
        self.water_actor.mapper.scalar_range = clim
        self.plotter.render()

    def add_source_markers(self, sources: List[ContaminantSource]):
        for src in sources:
            cyl = pv.Cylinder(center=(src.y, src.x, 1.5),
                              direction=(0,0,1), radius=2.0, height=4.0, resolution=16)
            cone = pv.Cone(center=(src.y, src.x, 4.2),
                           direction=(0,0,-1), radius=3.0, height=2.2, resolution=16)
            self.plotter.add_mesh(cyl,  color=src.color, opacity=0.88, name=f'src_cyl_{src.name}')
            self.plotter.add_mesh(cone, color=src.color, opacity=0.88, name=f'src_cone_{src.name}')

    def update_station_markers(self, station_data: List[dict],
                               stations: List[MonitoringStation]):
        for st, sd in zip(stations, station_data):
            total = sd['current']['total_weighted']
            col   = (1.,0.,0.) if total > sd['crit'] else \
                    (1.,0.65,0.) if total > sd['warn'] else (0.,0.9,0.3)
            sph  = pv.Sphere(radius=2.5, center=(st.y, st.x, 0.5))
            line = pv.Line((st.y,st.x,0.5),(st.y,st.x,8.0))
            try:    self.plotter.remove_actor(f'st_sph_{st.name}')
            except: pass
            self.plotter.add_mesh(sph,  color=col, opacity=1.0, name=f'st_sph_{st.name}')
            self.plotter.add_mesh(line, color=col, line_width=3, name=f'st_pole_{st.name}')

    def reset_camera(self):
        self.plotter.camera_position = [
            ( RIVER_WIDTH*1.8, -RIVER_LENGTH*0.28, RIVER_LENGTH*0.22),
            ( 0, RIVER_LENGTH/2, 0), (0, 0, 1)]
        self.plotter.render()


# ──────────────────────────────────────────────────────────────────────────────
# LIVE CHART PANEL (pyqtgraph)
# ──────────────────────────────────────────────────────────────────────────────

STATION_COLORS = ['#00ff88','#ff6b35','#4ecdc4','#ffe66d','#a8e6cf','#ff8b94']

class ChartPanel(QWidget):
    def __init__(self, stations: List[MonitoringStation]):
        super().__init__()
        self.stations = stations
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4,4,4,4)

        if not HAS_PYQTGRAPH:
            layout.addWidget(QLabel("Install pyqtgraph for live charts:\npip install pyqtgraph"))
            return

        self.plot = pg.PlotWidget(title="Monitoring Station — Contamination Index (weighted)")
        self.plot.setLabel('left',   'Contamination Index')
        self.plot.setLabel('bottom', 'Sample')
        self.plot.addLegend(offset=(10, 10))
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot)

        self.curves = {}
        for i, st in enumerate(stations):
            col  = STATION_COLORS[i % len(STATION_COLORS)]
            pen  = pg.mkPen(color=col, width=2)
            self.curves[st.name] = self.plot.plot(
                [], [], pen=pen, name=st.name[:24])

    def update(self, station_data: List[dict]):
        if not HAS_PYQTGRAPH:
            return
        for sd in station_data:
            name = sd['name']
            if name in self.curves:
                hist = sd['readings']
                self.curves[name].setData(np.arange(len(hist)), hist)


# ──────────────────────────────────────────────────────────────────────────────
# STATION TABLE PANEL
# ──────────────────────────────────────────────────────────────────────────────

class StationTablePanel(QWidget):
    def __init__(self, stations: List[MonitoringStation]):
        super().__init__()
        self.stations = stations
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4,4,4,4)

        self.table = QTableWidget(len(stations), 8)
        self.table.setHorizontalHeaderLabels([
            'Station', 'Km', 'Status',
            'Total', 'Heavy Metal', 'Nutrient', 'Pathogen', 'Organic'
        ])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        font = QFont('Courier New', 9)
        self.table.setFont(font)
        layout.addWidget(self.table)

    def update(self, station_data: List[dict]):
        for row, sd in enumerate(station_data):
            c   = sd['current']
            tot = c['total_weighted']

            if tot > sd['crit']:
                status, bg = '⚠ CRITICAL', QColor('#5a0000')
            elif tot > sd['warn']:
                status, bg = '! WARNING',  QColor('#3a2800')
            else:
                status, bg = '✓ SAFE',     QColor('#002a10')

            vals = [
                sd['name'][:22],
                f"{sd['river_km']:.1f}",
                status,
                f"{tot:.3f}",
                f"{c['heavy_metal']:.4f}",
                f"{c['nutrient']:.4f}",
                f"{c['pathogen']:.4f}",
                f"{c['organic']:.4f}",
            ]
            for col, val in enumerate(vals):
                item = QTableWidgetItem(val)
                item.setBackground(bg)
                if col == 2:
                    fg = QColor('#ff4444') if 'CRIT' in status else \
                         QColor('#ffaa00') if 'WARN' in status else \
                         QColor('#44ff88')
                    item.setForeground(fg)
                self.table.setItem(row, col, item)

        self.table.resizeColumnsToContents()


# ──────────────────────────────────────────────────────────────────────────────
# PROBE PANEL
# ──────────────────────────────────────────────────────────────────────────────

class ProbePanel(QGroupBox):
    def __init__(self):
        super().__init__("📍 Point Probe — Click river surface in 3D view")
        layout = QGridLayout(self)
        font   = QFont('Courier New', 9)

        labels = ['Location', 'River Km', 'Heavy Metal', 'Nutrient',
                  'Pathogen', 'Organic', 'Weighted Index', 'Sim Time']
        self._fields = {}
        for row, lbl in enumerate(labels):
            layout.addWidget(QLabel(lbl + ':'), row, 0)
            val = QLabel('—')
            val.setFont(font)
            val.setStyleSheet('color: #7dd3fc;')
            layout.addWidget(val, row, 1)
            self._fields[lbl] = val

    def update(self, probe: Optional[dict]):
        if probe is None:
            return
        t = probe['time_s']
        self._fields['Location'].setText(
            f"x={probe['x_m']:.1f} m,  y={probe['y_m']:.1f} m")
        self._fields['River Km'].setText(f"{probe['river_km']:.3f} km")
        self._fields['Heavy Metal'].setText(f"{probe['heavy_metal']:.5f} mg/L")
        self._fields['Nutrient'].setText(f"{probe['nutrient']:.5f} mg/L")
        self._fields['Pathogen'].setText(f"{probe['pathogen']:.5f} mg/L")
        self._fields['Organic'].setText(f"{probe['organic']:.5f} mg/L")
        self._fields['Weighted Index'].setText(f"{probe['total_weighted']:.4f}")
        self._fields['Sim Time'].setText(
            f"{int(t//3600):02d}h {int((t%3600)//60):02d}m {int(t%60):02d}s")


# ──────────────────────────────────────────────────────────────────────────────
# CONTROL PANEL (left sidebar)
# ──────────────────────────────────────────────────────────────────────────────

class ControlPanel(QWidget):
    layer_changed    = pyqtSignal(str)
    pause_toggled    = pyqtSignal(bool)
    speed_changed    = pyqtSignal(int)
    source_toggled   = pyqtSignal(str, bool)
    export_csv       = pyqtSignal()
    screenshot       = pyqtSignal()
    reset_camera     = pyqtSignal()

    def __init__(self, sources: List[ContaminantSource], shared: SharedState):
        super().__init__()
        self.shared  = shared
        self.sources = sources
        self.setFixedWidth(270)
        self.setStyleSheet("""
            QWidget { background:#0d1117; color:#c9d1d9; font-size:11px; }
            QGroupBox { border:1px solid #30363d; border-radius:4px;
                        margin-top:8px; padding-top:6px; color:#58a6ff; font-weight:bold; }
            QPushButton { background:#21262d; border:1px solid #30363d;
                          border-radius:4px; padding:5px 10px; color:#c9d1d9; }
            QPushButton:hover  { background:#30363d; }
            QPushButton:pressed{ background:#0d1117; }
            QPushButton#pause  { background:#1f6feb; color:white; }
            QPushButton#pause:checked { background:#da3633; }
            QComboBox { background:#21262d; border:1px solid #30363d;
                        border-radius:4px; padding:3px; color:#c9d1d9; }
            QSlider::groove:horizontal { height:4px; background:#30363d; border-radius:2px; }
            QSlider::handle:horizontal { width:12px; height:12px; margin:-4px 0;
                                         background:#1f6feb; border-radius:6px; }
            QCheckBox { spacing:6px; }
            QCheckBox::indicator { width:14px; height:14px; border:1px solid #30363d;
                                   border-radius:3px; background:#21262d; }
            QCheckBox::indicator:checked { background:#1f6feb; }
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8,8,8,8)
        outer.setSpacing(6)

        # ── Title ──
        title = QLabel("🌊 RIVER DIGITAL TWIN")
        title.setStyleSheet("font-size:13px; font-weight:bold; color:#58a6ff; padding:4px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(title)

        # ── Sim clock ──
        self.clock_label = QLabel("⏱  00h 00m 00s")
        self.clock_label.setStyleSheet("color:#7ee787; font-family:Courier New; font-size:12px;")
        self.clock_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self.clock_label)

        self.step_label = QLabel("Steps: 0")
        self.step_label.setStyleSheet("color:#6e7681; font-size:10px;")
        self.step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self.step_label)

        # ── Playback ──
        pb_box = QGroupBox("Playback")
        pb_lay = QVBoxLayout(pb_box)

        btn_row = QHBoxLayout()
        self.pause_btn = QPushButton("⏸  Pause")
        self.pause_btn.setObjectName("pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self._on_pause)
        btn_row.addWidget(self.pause_btn)

        reset_btn = QPushButton("📷 Reset View")
        reset_btn.clicked.connect(self.reset_camera)
        btn_row.addWidget(reset_btn)
        pb_lay.addLayout(btn_row)

        pb_lay.addWidget(QLabel("Simulation Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 16)
        self.speed_slider.setValue(4)
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.valueChanged.connect(self._on_speed)
        pb_lay.addWidget(self.speed_slider)
        self.speed_val_lbl = QLabel("4 steps/tick")
        self.speed_val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pb_lay.addWidget(self.speed_val_lbl)
        outer.addWidget(pb_box)

        # ── Layer selector ──
        layer_box = QGroupBox("Visualisation Layer")
        layer_lay = QVBoxLayout(layer_box)
        self.layer_combo = QComboBox()
        for name in LAYER_CONFIG:
            self.layer_combo.addItem(name)
        self.layer_combo.currentTextChanged.connect(self.layer_changed)
        layer_lay.addWidget(self.layer_combo)
        outer.addWidget(layer_box)

        # ── Source toggles ──
        src_box  = QGroupBox("Contaminant Sources")
        src_lay  = QVBoxLayout(src_box)
        src_lay.setSpacing(2)
        self._src_checks: Dict[str, QCheckBox] = {}
        SRC_ICONS = {'heavy_metal':'🔴','organic':'🟣','nutrient':'🟢','pathogen':'🟠'}
        for src in sources:
            icon  = SRC_ICONS.get(src.contaminant_type, '⚫')
            chk   = QCheckBox(f"{icon} {src.name[:28]}")
            chk.setChecked(True)
            chk.toggled.connect(lambda checked, n=src.name: self.source_toggled.emit(n, checked))
            src_lay.addWidget(chk)
            self._src_checks[src.name] = chk
            shared.active_source_toggles[src.name] = True
        outer.addWidget(src_box)

        # ── Export / Screenshot ──
        tools_box = QGroupBox("Tools")
        tools_lay = QVBoxLayout(tools_box)
        exp_btn   = QPushButton("💾  Export Data to CSV")
        exp_btn.clicked.connect(self.export_csv)
        tools_lay.addWidget(exp_btn)
        scr_btn   = QPushButton("📸  Save Screenshot")
        scr_btn.clicked.connect(self.screenshot)
        tools_lay.addWidget(scr_btn)
        outer.addWidget(tools_box)

        outer.addStretch()

        # ── Legend ──
        leg_box  = QGroupBox("Legend")
        leg_lay  = QGridLayout(leg_box)
        legend   = [("🔴 Heavy Metal","#ff4444"),("🟢 Nutrient","#44ff88"),
                    ("🟠 Pathogen","#ffaa33"),("🟣 Organic","#bb88ff"),
                    ("🔵 Monitor Stn","#00ccff")]
        for i, (lbl, col) in enumerate(legend):
            lw = QLabel(lbl)
            lw.setStyleSheet(f"color:{col};")
            leg_lay.addWidget(lw, i//2, i%2)
        outer.addWidget(leg_box)

    def _on_pause(self, checked: bool):
        self.shared.paused = checked
        self.pause_btn.setText("▶  Resume" if checked else "⏸  Pause")
        self.pause_toggled.emit(checked)

    def _on_speed(self, val: int):
        self.shared.speed = val
        self.speed_val_lbl.setText(f"{val} steps/tick")
        self.speed_changed.emit(val)

    def update_clock(self, sim_time: float, step_count: int):
        h = int(sim_time // 3600)
        m = int((sim_time % 3600) // 60)
        s = int(sim_time % 60)
        self.clock_label.setText(f"⏱  {h:02d}h {m:02d}m {s:02d}s")
        self.step_label.setText(f"Steps: {step_count:,}")

    def on_source_toggled(self, name: str, checked: bool):
        self.shared.active_source_toggles[name] = checked


# ──────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ──────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("River Ecosystem Digital Twin — Contaminant Transport")
        self.resize(1600, 950)
        self._apply_dark_palette()

        # ── Domain objects ──
        print("  ► Building river geometry...")
        self.geo      = RiverGeometry()
        self.sources  = self._make_sources()
        self.stations = self._make_stations()
        self.shared   = SharedState()

        print("  ► Initializing physics engine...")
        self.transport = ContaminantTransport(self.geo)

        print("  ► Running physics spin-up (60 steps)...")
        for _ in range(60):
            self.transport.step(self.sources)

        # ── Shared state initial source toggles ──
        for src in self.sources:
            self.shared.active_source_toggles[src.name] = True

        # ── Build UI ──
        self._build_ui()

        # ── Physics thread ──
        self.physics_thread = PhysicsThread(
            self.transport, self.sources, self.stations, self.shared)
        self.physics_thread.snapshot_ready.connect(self._on_physics_tick)
        self.physics_thread.start()

        # ── Render timer (Qt main thread, drives 3D + panels) ──
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._render_tick)
        self._render_timer.start(int(1000 / RENDER_HZ))

        # ── Station marker refresh (less frequent) ──
        self._station_timer = QTimer(self)
        self._station_timer.timeout.connect(self._refresh_station_markers)
        self._station_timer.start(500)   # every 0.5 s

        # Frame counter for CSV export buffer
        self._export_buffer: List[dict] = []
        self._last_snap: Optional[SimSnapshot] = None

        print("  ► GUI ready.\n")

    # ── Dark palette ──────────────────────────────────────────────────────────
    def _apply_dark_palette(self):
        pal = QPalette()
        pal.setColor(QPalette.ColorRole.Window,          QColor('#0d1117'))
        pal.setColor(QPalette.ColorRole.WindowText,      QColor('#c9d1d9'))
        pal.setColor(QPalette.ColorRole.Base,            QColor('#161b22'))
        pal.setColor(QPalette.ColorRole.AlternateBase,   QColor('#21262d'))
        pal.setColor(QPalette.ColorRole.Text,            QColor('#c9d1d9'))
        pal.setColor(QPalette.ColorRole.Button,          QColor('#21262d'))
        pal.setColor(QPalette.ColorRole.ButtonText,      QColor('#c9d1d9'))
        pal.setColor(QPalette.ColorRole.Highlight,       QColor('#1f6feb'))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor('#ffffff'))
        self.setPalette(pal)

    # ── Domain factories ──────────────────────────────────────────────────────
    def _make_sources(self) -> List[ContaminantSource]:
        return [
            ContaminantSource("Industrial Effluent (Factory A)",
                              480, 12, 45.0, 0.8, 'heavy_metal', (0.9,0.1,0.1)),
            ContaminantSource("Chemical Spill (Organic Solvent)",
                              650,  0, 60.0, 0.2, 'organic',     (0.6,0.0,0.8)),
            ContaminantSource("Agricultural Runoff",
                              850,-18, 12.0, 2.5, 'nutrient',    (0.2,0.8,0.2)),
            ContaminantSource("Sewage Overflow",
                             1100,  8, 28.0, 0.4, 'pathogen',    (0.9,0.5,0.1)),
            ContaminantSource("Mining Leachate",
                             1400,-25, 20.0, 1.2, 'heavy_metal', (0.7,0.3,0.0)),
        ]

    def _make_stations(self) -> List[MonitoringStation]:
        return [
            MonitoringStation("Upstream Reference",    200,   0, 0.2, 0.5,  2.0),
            MonitoringStation("Industrial Outfall",    500,  10, 0.5, 1.0,  5.0),
            MonitoringStation("Agricultural Zone",     900, -15, 0.9, 2.0,  8.0),
            MonitoringStation("Municipal Intake",     1200,   5, 1.2, 1.5,  6.0),
            MonitoringStation("Wetland Buffer",       1500, -20, 1.5, 3.0, 10.0),
            MonitoringStation("Downstream Gauge",     1800,   0, 1.8, 1.0,  4.0),
        ]

    # ── UI layout ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_lay = QHBoxLayout(central)
        root_lay.setContentsMargins(4,4,4,4)
        root_lay.setSpacing(4)

        # Left: control panel
        self.ctrl = ControlPanel(self.sources, self.shared)
        self.ctrl.layer_changed.connect(self._on_layer_changed)
        self.ctrl.source_toggled.connect(self.ctrl.on_source_toggled)
        self.ctrl.export_csv.connect(self._export_csv)
        self.ctrl.screenshot.connect(self._screenshot)
        self.ctrl.reset_camera.connect(lambda: self.viewport.reset_camera())
        root_lay.addWidget(self.ctrl)

        # Right: splitter (3D top | tabs bottom)
        right_split = QSplitter(Qt.Orientation.Vertical)

        # 3D viewport
        self.viewport = ViewportWidget(self.geo)
        self.viewport.probe_requested.connect(self._on_probe_request)
        self.viewport.add_source_markers(self.sources)
        right_split.addWidget(self.viewport)

        # Bottom tabs
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane  { border:1px solid #30363d; }
            QTabBar::tab      { background:#161b22; color:#8b949e; padding:6px 14px;
                                border:1px solid #30363d; }
            QTabBar::tab:selected { background:#21262d; color:#c9d1d9; }
        """)

        self.station_table = StationTablePanel(self.stations)
        tabs.addTab(self.station_table, "📊 Station Data")

        self.chart_panel = ChartPanel(self.stations)
        tabs.addTab(self.chart_panel, "📈 Live Charts")

        self.probe_panel = ProbePanel()
        tabs.addTab(self.probe_panel, "📍 Probe")

        right_split.addWidget(tabs)
        right_split.setSizes([620, 280])

        root_lay.addWidget(right_split, stretch=1)

        # Status bar
        self.status = QStatusBar()
        self.status.setStyleSheet("background:#161b22; color:#8b949e; font-size:10px;")
        self.setStatusBar(self.status)
        self.status.showMessage(
            "Physics thread running  |  Click river surface to probe concentration  |  "
            "Drag to rotate  |  Scroll to zoom  |  Shift+drag to pan")

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_physics_tick(self):
        """Called from physics thread via signal — safe to read shared state."""
        pass  # render timer drives all updates; signal just wakes this if needed

    def _render_tick(self):
        """Qt main thread: pull latest snapshot and update all panels."""
        snap = self.shared.get_snapshot()
        if snap is None:
            return
        self._last_snap = snap

        # 3D viewport
        self.viewport.update_from_snapshot(snap)

        # Sidebar clock
        self.ctrl.update_clock(snap.sim_time, snap.step_count)

        # Station table + chart
        self.station_table.update(snap.station_data)
        self.chart_panel.update(snap.station_data)

        # Probe panel (only if new probe result arrived)
        if snap.probe is not None:
            self.probe_panel.update(snap.probe)

        # Buffer for CSV export (1 row per render tick)
        if len(self._export_buffer) < 50_000:
            row = {'sim_time': snap.sim_time, 'step': snap.step_count}
            for sd in snap.station_data:
                c = sd['current']
                for ct in CONTAMINANT_TYPES:
                    row[f"{sd['name']}_{ct}"] = c[ct]
                row[f"{sd['name']}_total"] = c['total_weighted']
            self._export_buffer.append(row)

    def _refresh_station_markers(self):
        snap = self._last_snap
        if snap:
            self.viewport.update_station_markers(snap.station_data, self.stations)

    def _on_layer_changed(self, name: str):
        self.viewport.set_layer(name)

    def _on_probe_request(self, x: float, y: float):
        """Forward probe request to physics thread (thread-safe)."""
        self.physics_thread.request_probe(x, y)

    def _export_csv(self):
        if not self._export_buffer:
            QMessageBox.information(self, "Export", "No data to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "river_twin_data.csv", "CSV Files (*.csv)")
        if not path:
            return
        keys = list(self._export_buffer[0].keys())
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self._export_buffer)
        self.status.showMessage(f"✓ Exported {len(self._export_buffer)} rows → {path}", 5000)

    def _screenshot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "river_twin_screenshot.png", "PNG (*.png)")
        if path:
            self.viewport.plotter.screenshot(path)
            self.status.showMessage(f"✓ Screenshot saved → {path}", 4000)

    def closeEvent(self, event):
        self.physics_thread.stop()
        self.physics_thread.wait(2000)
        self._render_timer.stop()
        self._station_timer.stop()
        self.viewport.plotter.close()
        event.accept()


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def check_deps():
    required = {'numpy':'numpy','scipy':'scipy','pyvista':'pyvista',
                'pyvistaqt':'pyvistaqt','PyQt6':'PyQt6'}
    optional = {'pyqtgraph':'pyqtgraph (for live charts)'}
    missing  = [pkg for pkg in required if __import__('importlib').util.find_spec(pkg) is None]
    if missing:
        print(f"\n  Missing required packages: {', '.join(missing)}")
        print(f"    pip install {' '.join(missing)}\n")
        return False
    for pkg, label in optional.items():
        if __import__('importlib').util.find_spec(pkg) is None:
            print(f"⚠   Optional package missing: {label}")
            print(f"    pip install {pkg}\n")
    return True


if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║        RIVER ECOSYSTEM DIGITAL TWIN  v2.0                            ║
║        Multi-Threaded | PyQt6 + PyVista + pyqtgraph                  ║
║                                                                      ║
║  Physics thread  → PDE solver @ 60 Hz (daemon, never blocks UI)      ║
║  Qt main thread  → 3D render + panels @ 20 Hz                        ║
║  Mutex-protected shared state bridges the two threads                ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    if not check_deps():
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    win = MainWindow()
    win.show()
    sys.exit(app.exec())