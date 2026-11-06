# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Qianyv-Li8262
"""Analytic free-fall/orbit checks without loading CUDA or rendering assets."""
import math
import unittest

import numpy as np

from helpers.camera_motion import FreeFallCamera, camera_basis


class CameraMotionTests(unittest.TestCase):
    def make_camera(self, position=(10., 0., 0.), velocity=(0., 0., 0.)):
        return FreeFallCamera(position, velocity, np.eye(3), start_time=5.)

    def test_first_state_preserves_config_and_camera_local_velocity(self):
        position = np.array([15.53, -26.44, .15])
        velocity = np.array([.3, -.2, .1])
        basis = camera_basis(-10.49, .04, -.12)
        camera = FreeFallCamera(position, velocity, basis, 17.)
        np.testing.assert_array_equal(camera.position, position)
        np.testing.assert_allclose(camera.velocity, velocity, atol=1e-15)
        self.assertEqual(camera.time, 17.)
        self.assertEqual(camera.proper_time, 0.)
        dt = 1e-5
        radius = np.linalg.norm(position)
        B = (1+.5/radius)**2
        alpha = (1-.5/radius)/(1+.5/radius)
        gamma = 1/math.sqrt(1-float(velocity @ velocity))
        camera.advance(dt)
        np.testing.assert_allclose((camera.position-position)/dt,
                                   gamma/B*(np.array(basis).T @ velocity), rtol=1e-5)
        self.assertAlmostEqual((camera.time-17.)/dt, gamma/alpha, places=6)
        np.testing.assert_array_equal(position, [15.53, -26.44, .15])

    def test_release_from_rest_matches_schwarzschild_energy_law(self):
        camera = self.make_camera()
        energy2 = ((1-.5/10)/(1+.5/10))**2
        self.assertTrue(camera.advance(10.))
        radius = np.linalg.norm(camera.position)
        areal_radius = radius*(1+.5/radius)**2
        self.assertLess(radius, 10.)
        self.assertLess(camera.velocity[0], 0.)
        np.testing.assert_allclose(camera.position[1:], 0, atol=1e-14)
        expected_speed2 = 1-(1-2/areal_radius)/energy2
        self.assertAlmostEqual(float(camera.velocity @ camera.velocity), expected_speed2, places=11)
        self.assertGreater(camera.time-5., camera.proper_time)

    def test_tangential_initial_velocity_supports_a_circular_geodesic(self):
        radius = 10.
        areal_radius = radius*(1+.5/radius)**2
        speed = math.sqrt(1/(areal_radius-2))
        camera = self.make_camera(velocity=(0., speed, 0.))
        initial_angular_momentum = np.cross(camera.position, camera._state[3:6])
        self.assertTrue(camera.advance(100.))
        self.assertAlmostEqual(np.linalg.norm(camera.position), radius, places=8)
        self.assertGreater(np.linalg.norm(camera.position-np.array([radius, 0, 0])), 1.)
        np.testing.assert_allclose(np.cross(camera.position, camera._state[3:6]),
                                   initial_angular_momentum, rtol=1e-10, atol=1e-12)

    def test_frame_intervals_do_not_change_the_trajectory(self):
        a = self.make_camera(velocity=(-.4, .1, 0))
        b = self.make_camera(velocity=(-.4, .1, 0))
        a.advance(3.)
        for _ in range(30):
            b.advance(.1)
        np.testing.assert_allclose(a.position, b.position, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(a.velocity, b.velocity, rtol=1e-11, atol=1e-11)
        self.assertAlmostEqual(a.time, b.time, places=10)

    def test_stop_before_horizon_keeps_state_finite_and_speed_subluminal(self):
        camera = self.make_camera(position=(.56, 0, 0), velocity=(-.2, 0, 0))
        self.assertFalse(camera.advance(1.))
        self.assertTrue(camera.stopped)
        self.assertAlmostEqual(np.linalg.norm(camera.position), .55, places=10)
        self.assertTrue(np.isfinite(camera._state).all())
        self.assertLess(np.linalg.norm(camera.velocity), 1.)
        last = camera._state.copy()
        self.assertFalse(camera.advance(1.))
        np.testing.assert_array_equal(camera._state, last)

    def test_invalid_initial_conditions_and_time_steps(self):
        for position, velocity in [((.5, 0, 0), (0, 0, 0)),
                                   ((10, 0, 0), (1, 0, 0)),
                                   ((10, 0, 0), (.8, .8, 0)),
                                   ((math.nan, 0, 0), (0, 0, 0))]:
            with self.subTest(position=position, velocity=velocity), self.assertRaises(ValueError):
                self.make_camera(position, velocity)
        for interval in [-1., math.nan, math.inf]:
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                self.make_camera().advance(interval)

    def test_camera_basis_at_vertical_pitch_is_finite_and_orthonormal(self):
        for pitch in [-math.pi/2, 0., math.pi/2]:
            basis = np.array(camera_basis(.7, pitch, .4))
            np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-14)


if __name__ == '__main__':
    unittest.main()
