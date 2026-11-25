import argparse
import csv
import itertools
import math
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from param import BASELINE_SOURCE, COMMON_SOURCE, FLAG_DICT
from stencil_precision import (
    arch_flag, binary_dir, data_dir, tensor_source, tensor_type,
    validate_mesh_size, validate_precision_arch, validate_tile_size,
)


TIME_UNIT_TO_SECONDS = {
    's': 1,
    'sec': 1,
    'second': 1,
    'seconds': 1,
    'ms': 1e-3,
    'msec': 1e-3,
    'msecond': 1e-3,
    'millisecond': 1e-3,
    'milliseconds': 1e-3,
    'us': 1e-6,
    'usec': 1e-6,
    'usecond': 1e-6,
    'microsecond': 1e-6,
    'microseconds': 1e-6,
    'ns': 1e-9,
    'nsec': 1e-9,
    'nsecond': 1e-9,
    'nanosecond': 1e-9,
    'nanoseconds': 1e-9,
}


def compile_command_gen(arch, cuda_compute, stencil_size, mesh_size, tile_size, precision='fp16'):
    validate_precision_arch(arch, precision)
    validate_tile_size(mesh_size, tile_size)
    allowed = [tensor_type(precision)] if precision == 'tf32' else FLAG_DICT['cuda_compute']
    if cuda_compute not in allowed:
        raise ValueError(f'{cuda_compute} is not available in {precision} mode')
    output_file_name = f'{cuda_compute}_{stencil_size}_{mesh_size}_{tile_size}'
    source = BASELINE_SOURCE if cuda_compute == 'baseline' else tensor_source(precision)
    cmd = [
        'nvcc',
        '-O3',
        '-arch',
        arch_flag(arch),
        '-DRUN_TIMES=1',
        f'-DMESH_SIZE={mesh_size},TILE_SIZE={tile_size}',
        '-o',
        binary_dir(precision, 'sweep') + output_file_name,
    ]
    cmd.extend(shlex.split(COMMON_SOURCE[stencil_size]))
    cmd.extend(shlex.split(source[stencil_size]))
    return cmd, output_file_name


def iter_configs(mesh_size, precision='fp16'):
    validate_mesh_size(mesh_size)
    compute_types = [tensor_type(precision)] if precision == 'tf32' else FLAG_DICT['cuda_compute']
    for cuda_compute, stencil_size, tile_size in itertools.product(
        compute_types,
        FLAG_DICT['stencil_size'],
        FLAG_DICT['tile_size'],
    ):
        if (mesh_size // 16) % tile_size != 0:
            continue
        yield cuda_compute, stencil_size, mesh_size, tile_size


def run_checked(cmd):
    subprocess.run(cmd, check=True)


def compile_all(arch, mesh_size, jobs, precision='fp16'):
    Path(binary_dir(precision, 'sweep')).mkdir(parents=True, exist_ok=True)
    commands = []
    file_names = []
    for config in iter_configs(mesh_size, precision):
        cmd, file_name = compile_command_gen(arch, *config, precision=precision)
        commands.append(cmd)
        file_names.append(file_name)

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        future_to_cmd = {executor.submit(run_checked, cmd): cmd for cmd in commands}
        for index, future in enumerate(as_completed(future_to_cmd), 1):
            future.result()
            print(f'compiled {index}/{len(commands)}')

    return file_names


def expected_file_names(arch, mesh_size, precision='fp16'):
    return [compile_command_gen(arch, *config, precision=precision)[1]
            for config in iter_configs(mesh_size, precision)]


def run_ncu_all(file_names, raw_result, precision='fp16'):
    raw_result.write_text('')
    for index, file_name in enumerate(file_names, 1):
        print(f'ncu {index}/{len(file_names)} {file_name}')
        subprocess.run(
            [
                'bash',
                '-o',
                'pipefail',
                'script/run_file_with_ncu.sh',
                file_name,
                str(raw_result),
                binary_dir(precision, 'sweep'),
            ],
            check=True,
        )


def load_result(filename):
    rows = []
    with open(filename, newline='') as f:
        for name, time_val, unit in csv.reader(f):
            compute_type, stencil, mesh_size, tile_size = name.strip().rsplit('_', 3)
            rows.append((compute_type, stencil, mesh_size, tile_size, time_val, unit))
    return rows


def time_in_seconds(time_val, unit):
    unit = unit.strip().lower()
    if unit not in TIME_UNIT_TO_SECONDS:
        raise ValueError(f'Unknown time unit: {unit}')
    seconds = time_val * TIME_UNIT_TO_SECONDS[unit]
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f'Invalid duration: {time_val} {unit}')
    return seconds


def calc_gstencil(mesh_size, time_val, unit):
    return mesh_size * mesh_size / time_in_seconds(time_val, unit) / 1e9


def store_best_gstencil_csv(lines, filename, expected_configs=None):
    res_dict = {}
    measured_configs = set()
    for compute_type, stencil, mesh_size, tile_size, time_val, unit in lines:
        mesh_size, tile_size = int(mesh_size), int(tile_size)
        validate_tile_size(mesh_size, tile_size)
        time_val = float(time_val)
        seconds = time_in_seconds(time_val, unit)
        measured_configs.add((compute_type, stencil, mesh_size, tile_size))
        key = f'{compute_type}-{stencil}-{mesh_size}'
        if key not in res_dict or seconds < res_dict[key]['seconds']:
            res_dict[key] = {
                'compute_type': compute_type,
                'stencil': stencil,
                'mesh_size': mesh_size,
                'tile_size': tile_size,
                'time_val': time_val,
                'unit': unit,
                'seconds': seconds,
            }

    if not res_dict:
        raise ValueError('No stencil measurements found')
    if expected_configs is not None and measured_configs != set(expected_configs):
        raise ValueError('Measurements do not match the requested precision, mesh size and tile sweep')

    with open(filename, 'w', newline='') as f:
        fieldnames = ['type', 'stencil', 'mesh size', 'tile size', 'time', 'unit', 'GStencil/s']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for value in res_dict.values():
            gstencil = calc_gstencil(int(value['mesh_size']), value['time_val'], value['unit'])
            writer.writerow({
                'type': value['compute_type'],
                'stencil': value['stencil'],
                'mesh size': value['mesh_size'],
                'tile size': value['tile_size'],
                'time': value['time_val'],
                'unit': value['unit'],
                'GStencil/s': f'{gstencil:.6f}',
            })


def process_results(raw_result, output_csv, mesh_size=None, precision='fp16'):
    lines = load_result(raw_result)
    expected = iter_configs(mesh_size, precision) if mesh_size is not None else None
    store_best_gstencil_csv(lines, output_csv, expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_size', type=int, help='problem size N, must be divisible by 16')
    parser.add_argument('--arch', type=str, default='A100', help='A100, V100, or a CUDA arch like sm_80')
    parser.add_argument('--jobs', type=int, default=24, help='parallel nvcc compile jobs')
    parser.add_argument('--precision', choices=['fp16', 'tf32'], default='fp16')
    parser.add_argument('--skip-compile', action='store_true')
    parser.add_argument('--skip-run', action='store_true')
    parser.add_argument('--skip-process', action='store_true')
    args = parser.parse_args()

    validate_mesh_size(args.mesh_size)
    validate_precision_arch(args.arch, args.precision)
    if args.jobs <= 0:
        parser.error('--jobs must be positive')

    output_dir = data_dir(args.arch, args.precision)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_result = output_dir / f'ncu_result_{args.mesh_size}.txt'
    output_csv = output_dir / f'gstencil_best_{args.mesh_size}.csv'

    if args.skip_compile:
        file_names = expected_file_names(args.arch, args.mesh_size, args.precision)
    else:
        file_names = compile_all(args.arch, args.mesh_size, args.jobs, args.precision)

    if not args.skip_run:
        run_ncu_all(file_names, raw_result, args.precision)

    if not args.skip_process:
        process_results(raw_result, output_csv, args.mesh_size, args.precision)
        print(f'wrote {output_csv}')


if __name__ == '__main__':
    main()
