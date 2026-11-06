# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Qianyv-Li8262
"""Timelike camera geodesics in isotropic Schwarzschild coordinates, M=c=1.

ds^2=-alpha^2 dt^2+B^2 dx.dx, alpha=(1-u)/(1+u), B=(1+u)^2, u=1/(2r).
Metric: https://www.einsteintoolkit.org/thornguide/EinsteinInitialData/IDAnalyticBH/documentation.html
Beta is measured by a local static observer in camera forward/right/up axes,
matching blackholekernel. Camera axes stay fixed in world coordinates.
"""

import math
import numpy as np


def camera_basis(yaw, pitch, roll):
    """Return the renderer's forward/right/up convention, including at poles."""
    if not np.isfinite([yaw, pitch, roll]).all():
        raise ValueError('Camera angles must be finite')
    forward = np.array([math.cos(yaw)*math.cos(pitch),
                        math.sin(yaw)*math.cos(pitch), math.sin(pitch)])
    right = np.cross(forward, [0., 0., 1.])
    norm = np.linalg.norm(right)
    right = right/norm if norm > 1e-12 else np.array([math.sin(yaw), -math.cos(yaw), 0.])
    up = np.cross(right, forward)
    return (forward, right*math.cos(roll) + up*math.sin(roll),
            up*math.cos(roll) - right*math.sin(roll))


class FreeFallCamera:
    """Evolve position, canonical momentum p, and coordinate time with RK4.

    gamma=sqrt(1+p.p/B^2), dx/dtau=p/B^2,
    dp/dtau=-gamma^2 grad(log alpha)+(gamma^2-1) grad(log B),
    dt/dtau=gamma/alpha. Substeps shrink near the horizon. The stop radius
    matches the ray kernel's existing cutoff, outside the chart singularity.
    """

    stop_radius = 0.55

    def __init__(self, position, velocity, basis, start_time=0.0):
        position = np.array(position, dtype=np.float64, copy=True)
        velocity = np.asarray(velocity, dtype=np.float64)
        self.basis = np.asarray(basis, dtype=np.float64).T.copy()
        if position.shape != (3,) or velocity.shape != (3,):
            raise ValueError('Camera position and velocity must have three components')
        if (not np.isfinite(position).all() or not np.isfinite(velocity).all()
                or not math.isfinite(start_time)):
            raise ValueError('Camera position, velocity, and time must be finite')
        if (self.basis.shape != (3, 3) or not np.isfinite(self.basis).all()
                or not np.allclose(self.basis.T @ self.basis, np.eye(3), atol=1e-7)):
            raise ValueError('Camera basis must be orthonormal')
        radius = np.linalg.norm(position)
        speed2 = float(velocity @ velocity)
        if radius <= self.stop_radius:
            raise ValueError('Initial camera radius must be greater than 0.55')
        if speed2 >= 1:
            raise ValueError('Initial camera speed must be below light speed (beta < 1)')
        B = (1 + 0.5/radius)**2
        momentum = B / math.sqrt(1-speed2) * (self.basis @ velocity)
        self._state = np.r_[position, momentum, float(start_time)]
        self.proper_time = 0.0
        self.stopped = False

    @property
    def position(self):
        return self._state[:3].copy()

    @property
    def time(self):
        return float(self._state[6])

    @property
    def velocity(self):
        """Return local beta along the camera's forward/right/up axes."""
        radius = np.linalg.norm(self._state[:3])
        B = (1 + 0.5/radius)**2
        p = self._state[3:6]
        gamma = math.sqrt(1 + float(p @ p)/(B*B))
        return self.basis.T @ (p/(B*gamma))

    @staticmethod
    def _rates(state):
        x, p = state[:3], state[3:6]
        radius = np.linalg.norm(x)
        u = 0.5/radius
        B = (1+u)**2
        alpha = (1-u)/(1+u)
        p2 = float(p @ p)/(B*B)
        gamma2 = 1+p2
        grad_log_alpha = x/(radius**3*(1-u*u))
        grad_log_B = -x/(radius**3*(1+u))
        return np.r_[p/(B*B), -gamma2*grad_log_alpha + p2*grad_log_B,
                     math.sqrt(gamma2)/alpha]

    @classmethod
    def _rk4(cls, state, dtau):
        k1 = cls._rates(state)
        k2 = cls._rates(state + 0.5*dtau*k1)
        k3 = cls._rates(state + 0.5*dtau*k2)
        k4 = cls._rates(state + dtau*k3)
        return state + (dtau/6)*(k1 + 2*k2 + 2*k3 + k4)

    def advance(self, d_tau):
        """Advance by camera proper time; return False on the stop boundary."""
        if not math.isfinite(d_tau) or d_tau < 0:
            raise ValueError('d_tau must be finite and nonnegative')
        if self.stopped:
            return False
        remaining = float(d_tau)
        while remaining > 0:
            radius = np.linalg.norm(self._state[:3])
            B = (1+0.5/radius)**2
            p = self._state[3:6]
            gamma = math.sqrt(1+float(p @ p)/(B*B))
            dtau = min(remaining, 0.05, 0.1*(radius-0.5)*B/gamma)
            candidate = self._rk4(self._state, dtau)
            if np.linalg.norm(candidate[:3]) <= self.stop_radius:
                # Locate the cutoff before feeding an invalid camera to CUDA.
                low, high = 0.0, dtau
                for _ in range(40):
                    middle = (low+high)/2
                    if np.linalg.norm(self._rk4(self._state, middle)[:3]) > self.stop_radius:
                        low = middle
                    else:
                        high = middle
                self._state = self._rk4(self._state, low)
                self.proper_time += low
                self.stopped = True
                return False
            self._state = candidate
            self.proper_time += dtau
            remaining = max(0.0, remaining-dtau)
        return True
