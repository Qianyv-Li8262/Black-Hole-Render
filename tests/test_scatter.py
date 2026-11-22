"""CPU checks for scatter checkpoints and angular quadrature."""
import errno
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from bake_scatter import latest_epoch, prune_rolling_epochs, publish_epoch, quadrature, reserve_epoch_file
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

    def test_switch_keeps_the_path_for_offline_and_realtime(self):
        cfg = SimpleNamespace(OPEN_SCATTER=False, SCATTER_TABLE="configured/epoch.json")
        with patch("scatter_table.read_epoch", return_value=({"scene": {}}, object())) as read:
            with patch("scatter_table.scene_settings", return_value={}), patch("scatter_table.ScatterTable") as upload:
                self.assertIsNone(configured_table(cfg))
                read.assert_not_called()
                self.assertIs(configured_table(cfg, enabled=True), upload.return_value)
                cfg.OPEN_SCATTER = True
                self.assertIsNone(configured_table(cfg, enabled=False))
                self.assertEqual(read.call_count, 1)
                self.assertIs(configured_table(cfg), upload.return_value)
        cfg.SCATTER_TABLE = None
        with self.assertRaisesRegex(ValueError, "Set SCATTER_TABLE"):
            configured_table(cfg)

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

    def test_rolling_cleanup_preserves_latest_and_archived_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for epoch, policy in enumerate((None, "keep", "rolling", "rolling", "rolling"), 1):
                folder = root / f"epoch_{epoch:04d}"
                folder.mkdir()
                (folder / "table.npy").touch()
                (folder / "epoch.json").write_text(json.dumps(dict(format=FORMAT, epoch=epoch,
                                                                  checkpoint_policy=policy)))
            (root / "epoch_0004" / "notes.txt").touch()

            prune_rolling_epochs(root, 5)
            self.assertEqual(sorted(p.name for p in root.iterdir()),
                             ["epoch_0001", "epoch_0002", "epoch_0004", "epoch_0005"])
            self.assertTrue((root / "epoch_0004" / "table.npy").is_file())

    def test_publish_retries_a_transient_windows_file_handle(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "staging"
            stage.mkdir()
            target = Path(tmp) / "epoch_0001"
            rename = Path.rename
            busy = PermissionError("directory is temporarily locked")
            busy.winerror = 5
            with patch.object(Path, "rename", autospec=True, side_effect=[busy, None]) as move:
                with patch("bake_scatter.time.sleep"):
                    publish_epoch(stage, target)
                self.assertEqual(move.call_count, 2)
            rename(stage, target)
            with patch.object(Path, "rename", autospec=True, side_effect=busy) as move:
                with self.assertRaises(PermissionError):
                    publish_epoch(stage, target)
                self.assertEqual(move.call_count, 1)
