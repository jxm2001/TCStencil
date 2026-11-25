import re
from pathlib import Path

from param import BINARY_DIR, FLAG_DICT, TENSOR_SOURCE


TF32_SOURCE = {
    stencil: f'layout16_tf32/host_{stencil}.cu layout16_tf32/kernel_{stencil}.cu'
    for stencil in FLAG_DICT['stencil_size']
}


def arch_flag(arch):
    return {'A100': 'sm_80', 'V100': 'sm_70'}.get(arch, arch)


def validate_precision_arch(arch, precision):
    if precision not in ('fp16', 'tf32'):
        raise ValueError(f'Unknown precision: {precision}')
    if precision == 'tf32':
        match = re.fullmatch(r'(?:sm|compute)_(\d+)[a-z]?', arch_flag(arch))
        if not match or int(match.group(1)) < 80:
            raise ValueError('TF32 requires A100 or a CUDA architecture >= sm_80; V100 is unsupported')


def validate_mesh_size(mesh_size):
    if mesh_size <= 0 or mesh_size % 16 != 0:
        raise ValueError('mesh size must be a positive multiple of 16')
    if (mesh_size + 32) ** 2 > 2**31 - 1:
        raise ValueError('padded mesh exceeds the kernel integer indexing range')


def validate_tile_size(mesh_size, tile_size):
    validate_mesh_size(mesh_size)
    if tile_size not in FLAG_DICT['tile_size'] or (mesh_size // 16) % tile_size != 0:
        raise ValueError(f'Invalid tile size {tile_size} for mesh size {mesh_size}')


def tensor_type(precision):
    return 'tensor_tf32' if precision == 'tf32' else 'tensor'


def tensor_source(precision):
    return TF32_SOURCE if precision == 'tf32' else TENSOR_SOURCE


def binary_dir(precision, mode):
    return f'{BINARY_DIR}tf32/{mode}/' if precision == 'tf32' else BINARY_DIR


def data_dir(arch, precision):
    layout = 'layout16_tf32' if precision == 'tf32' else 'layout16'
    return Path('data') / arch / layout
