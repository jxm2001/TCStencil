import argparse
import csv
import math
import re
import shlex
import subprocess
from pathlib import Path

from param import COMMON_SOURCE
from stencil_precision import (
    arch_flag, binary_dir, data_dir, tensor_source, tensor_type,
    validate_mesh_size, validate_precision_arch, validate_tile_size,
)


KERNELS = ['box2d1r', 'box2d2r', 'star2d1r', 'star2d2r']
TIME_RE = re.compile(r'\[Time\]\s+Time used:\s+(\S+)\s+ms')


def binary_name(stencil, mesh_size, tile_size, precision='fp16'):
    return f'{tensor_type(precision)}_{stencil}_{mesh_size}_{tile_size}'


def compile_command(arch, stencil, mesh_size, tile_size, precision='fp16'):
    validate_precision_arch(arch, precision)
    validate_tile_size(mesh_size, tile_size)
    output = Path(binary_dir(precision, 'timed')) / binary_name(stencil, mesh_size, tile_size, precision)
    cmd = [
        'nvcc',
        '-O3',
        '-arch',
        arch_flag(arch),
        f'-DMESH_SIZE={mesh_size},TILE_SIZE={tile_size}',
        '-o',
        str(output),
    ]
    cmd.extend(shlex.split(COMMON_SOURCE[stencil]))
    cmd.extend(shlex.split(tensor_source(precision)[stencil]))
    return cmd


def load_best_tiles(filename, mesh_size, precision='fp16', arch='A100'):
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(
            f'{path} not found. Generate it first with: '
            f'python3 script/run_size_gstencil.py {mesh_size} --arch {arch} --precision {precision}'
        )

    tiles = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row['type'] != tensor_type(precision):
                continue
            if int(row['mesh size']) != mesh_size:
                continue
            if row['stencil'] in KERNELS:
                tile_size = int(row['tile size'])
                validate_tile_size(mesh_size, tile_size)
                if row['stencil'] in tiles:
                    raise ValueError(f'Duplicate {row["stencil"]} entry in {path}')
                tiles[row['stencil']] = tile_size

    missing = [stencil for stencil in KERNELS if stencil not in tiles]
    if missing:
        raise RuntimeError(
            f'Missing {tensor_type(precision)} mesh_size={mesh_size} entries in {path}: {", ".join(missing)}'
        )
    return tiles


def compile_binaries(arch, mesh_size, tiles, precision='fp16'):
    Path(binary_dir(precision, 'timed')).mkdir(parents=True, exist_ok=True)
    for stencil in KERNELS:
        name = binary_name(stencil, mesh_size, tiles[stencil], precision)
        print(f'compile {name}')
        subprocess.run(compile_command(arch, stencil, mesh_size, tiles[stencil], precision), check=True)


def run_and_measure(stencil, mesh_size, tile_size, precision='fp16'):
    exe = Path(binary_dir(precision, 'timed')) / binary_name(stencil, mesh_size, tile_size, precision)
    print(f'time {exe}')
    proc = subprocess.run([str(exe)], check=True, text=True, capture_output=True)
    output = proc.stdout + proc.stderr
    match = TIME_RE.search(output)
    if not match:
        raise RuntimeError(f'Could not find timing line in output of {exe}')
    time_ms = float(match.group(1))
    if not math.isfinite(time_ms) or time_ms <= 0:
        raise ValueError(f'Invalid timing from {exe}: {time_ms} ms')
    return time_ms


def gstencil_per_second(mesh_size, time_ms):
    if not math.isfinite(time_ms) or time_ms <= 0:
        raise ValueError(f'Invalid timing: {time_ms} ms')
    return mesh_size * mesh_size / (time_ms * 1e-3) / 1e9


def write_table(rows, filename):
    with open(filename, 'w', newline='') as f:
        fieldnames = ['type', 'stencil', 'mesh size', 'tile size', 'time', 'unit', 'GStencil/s']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_ncu_reports(mesh_size, tiles, precision='fp16'):
    for stencil in KERNELS:
        name = binary_name(stencil, mesh_size, tiles[stencil], precision)
        exe = Path(binary_dir(precision, 'timed')) / name
        report = Path(name)
        print(f'ncu {name}')
        subprocess.run(
            [
                'ncu',
                '--set',
                'full',
                '--launch-skip',
                '5',
                '--launch-count',
                '1',
                '--force-overwrite',
                '-o',
                str(report),
                str(exe),
            ],
            check=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_size', type=int, help='problem size N, must be divisible by 16')
    parser.add_argument('--arch', type=str, default='A100', help='A100, V100, or a CUDA arch like sm_80')
    parser.add_argument('--best-csv', type=str, default=None)
    parser.add_argument('--precision', choices=['fp16', 'tf32'], default='fp16')
    parser.add_argument('--skip-compile', action='store_true')
    parser.add_argument('--skip-time', action='store_true')
    parser.add_argument('--skip-ncu', action='store_true')
    args = parser.parse_args()

    validate_mesh_size(args.mesh_size)
    validate_precision_arch(args.arch, args.precision)

    best_csv = args.best_csv
    if best_csv is None:
        best_csv = data_dir(args.arch, args.precision) / f'gstencil_best_{args.mesh_size}.csv'

    tiles = load_best_tiles(best_csv, args.mesh_size, args.precision, args.arch)

    if not args.skip_compile:
        compile_binaries(args.arch, args.mesh_size, tiles, args.precision)

    if not args.skip_time:
        rows = []
        for stencil in KERNELS:
            tile_size = tiles[stencil]
            time_ms = run_and_measure(stencil, args.mesh_size, tile_size, args.precision)
            rows.append({
                'type': tensor_type(args.precision),
                'stencil': stencil,
                'mesh size': args.mesh_size,
                'tile size': tile_size,
                'time': f'{time_ms:.6f}',
                'unit': 'ms',
                'GStencil/s': f'{gstencil_per_second(args.mesh_size, time_ms):.6f}',
            })
        prefix = 'gstencil_tf32' if args.precision == 'tf32' else 'gstencil'
        output_csv = f'{prefix}_{args.mesh_size}.csv'
        write_table(rows, output_csv)
        print(f'wrote {output_csv}')

    if not args.skip_ncu:
        run_ncu_reports(args.mesh_size, tiles, args.precision)


if __name__ == '__main__':
    main()
