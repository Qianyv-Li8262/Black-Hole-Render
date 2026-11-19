# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 Qianyv-Li8262
"""Validate disk texture semantics before selecting a rendering kernel."""
import json
from pathlib import Path
import numpy as np


_COEFFICIENT_FORMAT = 'native34-coefficients-v1'


def _read_metadata(path):
    sidecar = Path(path).with_suffix('.json')
    if not sidecar.exists():
        return {}
    metadata = json.loads(sidecar.read_text(encoding='utf-8'))
    if not isinstance(metadata, dict):
        raise ValueError(f'Expected a disk metadata object in {sidecar}')
    return metadata


def _half_extent(metadata):
    extent = float(metadata.get('vertical_half_extent', 2.5))
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError('Disk vertical_half_extent must be positive and finite')
    return extent


def disk_half_extent(path):
    """Physical vertical texture bound; old caches retain their 2.5M domain."""
    return _half_extent(_read_metadata(path))


def validate_disk_loop_pair(path_a, path_b):
    """Validate two independent bakes and return their common extent."""
    metadata_a, metadata_b = _read_metadata(path_a), _read_metadata(path_b)
    shared_fields = ('settings', 'physics_lut_sha256', 'core_profile_sha256', 'vertical_half_extent')
    for path, metadata in ((path_a, metadata_a), (path_b, metadata_b)):
        message = f'Disk loop cache {path}: rebake with tools/prebake_native34.py'
        if metadata.get('format') != _COEFFICIENT_FORMAT:
            raise ValueError(f'{message} to generate coefficient metadata')
        for field in (*shared_fields, 'noise_seed'):
            if field not in metadata:
                raise ValueError(f'{message} to record {field}')
        if not isinstance(metadata['settings'], dict) or 'resolution' not in metadata['settings']:
            raise ValueError(f'{message} to record settings.resolution')
        for field in ('physics_lut_sha256', 'core_profile_sha256'):
            if not isinstance(metadata[field], str) or not metadata[field]:
                raise ValueError(f'{message} to record a valid {field}')
        if type(metadata['noise_seed']) is not int or metadata['noise_seed'] < 0:
            raise ValueError(f'{message} to record a valid noise_seed')
        try:
            _half_extent(metadata)
        except (TypeError, ValueError) as error:
            raise ValueError(f'{message} to record a valid vertical_half_extent') from error

    for field in shared_fields:
        if metadata_a[field] != metadata_b[field]:
            raise ValueError(f'Disk loop A/B {field} differs; rebake {path_a} and {path_b} with matching inputs')
    if metadata_a['noise_seed'] == metadata_b['noise_seed']:
        raise ValueError(f'Disk loop A/B noise_seed must differ; rebake {path_b} with a different --seed')
    # Inspect the arrays before GPU initialization without loading the texture data.
    _load_disk_array(path_a, metadata_a, coefficients=True, mmap_mode='r')
    _load_disk_array(path_b, metadata_b, coefficients=True, mmap_mode='r')
    return _half_extent(metadata_a)


def load_disk_cache(path, coefficients=False):
    return _load_disk_array(path, _read_metadata(path), coefficients)


def _load_disk_array(path, metadata, coefficients, mmap_mode=None):
    encoded = metadata.get('format') == _COEFFICIENT_FORMAT
    if encoded != bool(coefficients):
        raise ValueError('Disk cache format and DISK_COEFFICIENTS disagree; generate cache with tools/prebake_native34.py')
    result = np.load(path, mmap_mode=mmap_mode)
    if result.ndim != 4 or result.shape[-1] != 4:
        raise ValueError(f'Invalid disk texture shape in {path}: {result.shape}')
    if encoded and result.dtype != np.float32:
        raise ValueError(f'Independent disk coefficients in {path} must be stored in float32')
    settings = metadata.get('settings')
    if encoded and isinstance(settings, dict) and 'resolution' in settings:
        resolution = settings['resolution']
        if (not isinstance(resolution, list) or len(resolution) != 3
                or any(type(size) is not int or size < 2 for size in resolution)):
            raise ValueError(f'Invalid disk texture resolution in metadata for {path}')
        expected_shape = (*resolution, 4)
        if result.shape != expected_shape:
            raise ValueError(f'Disk texture shape for {path} is {result.shape}, '
                             f'but its metadata expects {expected_shape}. Rebake this texture.')
    return result
