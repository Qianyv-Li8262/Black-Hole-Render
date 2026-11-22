"""CPU checks for scatter checkpoints and angular quadrature."""
import errno
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from bake_scatter import latest_epoch, quadrature, reserve_epoch_file
from scatter_table import FORMAT, configured_table, read_epoch, table_shape


class ScatterTests(unittest.TestCase):
    def test_quadrature_preserves_sphere_moments(self):
        q = quadrature(256, 5).astype(np.float64)
        x, y, z, weight = q.T
        np.testing.assert_allclose(x*x + y*y + z*z, 1, atol=1e-7)
        self.assertAlmostEqual(weight.sum(), 4*math.pi, places=6)
        for degree in range(1, 11):
            expected = 0 if degree % 2 else 4*math.pi/(degree+1)
            self.assertAlmostEqual(weight @ z**degree, expected, places=6)
        with self.assertRaises(ValueError):
            quadrature(32, 5)

    def test_table_layout_and_retuning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "epoch.json"
            meta = dict(format=FORMAT, order=2, grid=[2, 3, 4], g=0.8, albedo=0.8, scene={})
            np.save(root / "table.npy", np.zeros(table_shape(meta["grid"], meta["order"]), np.float16))
            cfg = type("Config", (), {"SCATTER_TABLE": str(path), "SCATTER_G": 0.5})
            with patch("scatter_table.scene_settings", return_value={}), patch("scatter_table.ScatterTable") as upload:
                path.write_text(json.dumps(dict(meta, epoch=1)))
                configured_table(cfg)
                self.assertEqual(upload.call_args.args[0]["g"], 0.5)
                upload.reset_mock()
                path.write_text(json.dumps(dict(meta, epoch=2)))
                with self.assertRaisesRegex(ValueError, "rebaking"):
                    configured_table(cfg)
            np.save(root / "table.npy", np.zeros((2, 3), np.float16))
            with self.assertRaisesRegex(ValueError, "shape"):
                read_epoch(path)

    def test_latest_epoch_ignores_staging_and_checks_published_epoch_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".scatter-interrupted" / "epoch_0010").mkdir(parents=True)
            self.assertIsNone(latest_epoch(root))

            meta = dict(format=FORMAT, order=0, grid=[2, 2, 2], g=0.8, albedo=0.8)
            for epoch in (9, 10):
                folder = root / f"epoch_{epoch:04d}"
                folder.mkdir()
                np.save(folder / "table.npy", np.zeros(table_shape(meta["grid"], meta["order"]), np.float16))
                (folder / "epoch.json").write_text(json.dumps(dict(meta, epoch=epoch)))

            path = root / "epoch_0010" / "epoch.json"
            self.assertEqual(latest_epoch(root), path)
            path.write_text(json.dumps(dict(meta, epoch=9)))
            with self.assertRaisesRegex(ValueError, "does not match"):
                latest_epoch(root)

    def test_reserving_an_epoch_reports_insufficient_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.npy"
            path.touch()
            with patch("bake_scatter.shutil.disk_usage") as disk_usage:
                disk_usage.return_value.free = 1024
                with self.assertRaises(OSError) as error:
                    reserve_epoch_file(path, 2048)
            self.assertEqual(error.exception.errno, errno.ENOSPC)
