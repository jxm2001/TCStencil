import argparse
import csv
import itertools
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from param import BASELINE_SOURCE, BINARY_DIR, COMMON_SOURCE, FLAG_DICT, TENSOR_SOURCE


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


def arch_flag(arch):
    if arch == 'A100':
        return 'sm_80'
    if arch == 'V100':
        return 'sm_70'
    return arch


def validate_mesh_size(mesh_size):
    if mesh_size <= 0:
        raise ValueError('mesh size must be positive')
    if mesh_size % 16 != 0:
        raise ValueError('mesh size must be divisible by 16')


def compile_command_gen(arch, cuda_compute, stencil_size, mesh_size, tile_size):
    output_file_name = f'{cuda_compute}_{stencil_size}_{mesh_size}_{tile_size}'
    source = BASELINE_SOURCE if cuda_compute == 'baseline' else TENSOR_SOURCE
    cmd = [
        'nvcc',
        '-O3',
        '-arch',
        arch_flag(arch),
        '-DRUN_TIMES=1',
        f'-DMESH_SIZE={mesh_size},TILE_SIZE={tile_size}',
        '-o',
        BINARY_DIR + output_file_name,
    ]
    cmd.extend(shlex.split(COMMON_SOURCE[stencil_size]))
    cmd.extend(shlex.split(source[stencil_size]))
    return cmd, output_file_name


def iter_configs(mesh_size):
    for cuda_compute, stencil_size, tile_size in itertools.product(
        FLAG_DICT['cuda_compute'],
        FLAG_DICT['stencil_size'],
        FLAG_DICT['tile_size'],
    ):
        if (mesh_size // 16) % tile_size != 0:
            continue
        yield cuda_compute, stencil_size, mesh_size, tile_size


def run_checked(cmd):
    subprocess.run(cmd, check=True)


def compile_all(arch, mesh_size, jobs):
    Path(BINARY_DIR).mkdir(parents=True, exist_ok=True)
    commands = []
    file_names = []
    for config in iter_configs(mesh_size):
        cmd, file_name = compile_command_gen(arch, *config)
        commands.append(cmd)
        file_names.append(file_name)

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        future_to_cmd = {executor.submit(run_checked, cmd): cmd for cmd in commands}
        for index, future in enumerate(as_completed(future_to_cmd), 1):
            future.result()
            print(f'compiled {index}/{len(commands)}')

    return file_names


def expected_file_names(arch, mesh_size):
    return [compile_command_gen(arch, *config)[1] for config in iter_configs(mesh_size)]


def run_ncu_all(file_names, raw_result):
    raw_result.write_text('')
    for index, file_name in enumerate(file_names, 1):
        print(f'ncu {index}/{len(file_names)} {file_name}')
        subprocess.run(
            [
                'bash',
                'script/run_file_with_ncu.sh',
                file_name,
                str(raw_result),
                BINARY_DIR,
            ],
            check=True,
        )


def load_result(filename):
    with open(filename, 'r') as f:
        return [line.replace('_', ',') for line in f.readlines()]


def calc_gstencil(mesh_size, time_val, unit):
    unit = unit.strip().lower()
    if unit not in TIME_UNIT_TO_SECONDS:
        raise ValueError(f'Unknown time unit: {unit}')
    seconds = time_val * TIME_UNIT_TO_SECONDS[unit]
    return mesh_size * mesh_size / seconds / 1e9


def store_best_gstencil_csv(lines, filename):
    res_dict = {}
    for line in lines:
        compute_type, stencil, mesh_size, tile_size, time_val, unit = line.strip().split(',')
        time_val = float(time_val)
        key = f'{compute_type}-{stencil}-{mesh_size}'
        if key not in res_dict or time_val < res_dict[key]['time_val']:
            res_dict[key] = {
                'compute_type': compute_type,
                'stencil': stencil,
                'mesh_size': mesh_size,
                'tile_size': tile_size,
                'time_val': time_val,
                'unit': unit,
            }

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


def process_results(raw_result, output_csv):
    lines = load_result(raw_result)
    store_best_gstencil_csv(lines, output_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_size', type=int, help='problem size N, must be divisible by 16')
    parser.add_argument('--arch', type=str, default='A100', help='A100, V100, or a CUDA arch like sm_80')
    parser.add_argument('--jobs', type=int, default=24, help='parallel nvcc compile jobs')
    parser.add_argument('--skip-compile', action='store_true')
    parser.add_argument('--skip-run', action='store_true')
    parser.add_argument('--skip-process', action='store_true')
    args = parser.parse_args()

    validate_mesh_size(args.mesh_size)

    data_dir = Path(f'./data/{args.arch}/layout16/')
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_result = data_dir / f'ncu_result_{args.mesh_size}.txt'
    output_csv = data_dir / f'gstencil_best_{args.mesh_size}.csv'

    if args.skip_compile:
        file_names = expected_file_names(args.arch, args.mesh_size)
    else:
        file_names = compile_all(args.arch, args.mesh_size, args.jobs)

    if not args.skip_run:
        run_ncu_all(file_names, raw_result)

    if not args.skip_process:
        process_results(raw_result, output_csv)
        print(f'wrote {output_csv}')


if __name__ == '__main__':
    main()
