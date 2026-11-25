import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'script'))

import run_size_gstencil as sweep
import run_size_gstencil_ncu as timed
import stencil_precision as precision


FIELDS = ['type', 'stencil', 'mesh size', 'tile size', 'time', 'unit', 'GStencil/s']


class PrecisionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.previous_cwd = Path.cwd()
        os.chdir(self.directory.name)

    def tearDown(self):
        os.chdir(self.previous_cwd)
        self.directory.cleanup()

    def write_best(self, path, mode='tf32', size=160, tile=5):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for stencil in timed.KERNELS:
                writer.writerow({'type': precision.tensor_type(mode), 'stencil': stencil,
                                 'mesh size': size, 'tile size': tile,
                                 'time': 1, 'unit': 'us', 'GStencil/s': 25.6})

    def test_fp16_compile_is_unchanged(self):
        command, name = sweep.compile_command_gen('A100', 'tensor', 'box2d1r', 160, 5)
        self.assertEqual(name, 'tensor_box2d1r_160_5')
        self.assertEqual(command, [
            'nvcc', '-O3', '-arch', 'sm_80', '-DRUN_TIMES=1', '-DMESH_SIZE=160,TILE_SIZE=5',
            '-o', './bin_ncu/tensor_box2d1r_160_5', './common/main_1r.cu',
            './common/gold_box2d1r.cpp', './layout16/host_box2d1r.cu', './layout16/kernel_box2d1r.cu',
        ])
        baseline, _ = sweep.compile_command_gen('V100', 'baseline', 'star2d2r', 160, 5)
        self.assertIn('sm_70', baseline)
        self.assertIn('./baseline/kernel_star2d2r.cu', baseline)

    def test_tf32_sources_and_binary_modes_are_separate(self):
        for stencil in timed.KERNELS:
            sample, name = sweep.compile_command_gen('A100', 'tensor_tf32', stencil, 160, 5, 'tf32')
            normal = timed.compile_command('A100', stencil, 160, 5, 'tf32')
            self.assertEqual(name, f'tensor_tf32_{stencil}_160_5')
            self.assertIn('-DRUN_TIMES=1', sample)
            self.assertFalse(any('RUN_TIMES' in arg for arg in normal))
            self.assertIn(f'layout16_tf32/kernel_{stencil}.cu', normal)
            self.assertNotEqual(sample[sample.index('-o') + 1], normal[normal.index('-o') + 1])
            self.assertIn('/sweep/', sample[sample.index('-o') + 1])
            self.assertIn('/timed/', normal[normal.index('-o') + 1])

    def test_architecture_validation(self):
        for arch in ('A100', 'sm_80', 'sm_86', 'sm_90a', 'compute_80', 'sm_100'):
            precision.validate_precision_arch(arch, 'tf32')
        for arch in ('V100', 'sm_70', 'sm_75', 'nonsense'):
            with self.subTest(arch=arch), self.assertRaises(ValueError):
                precision.validate_precision_arch(arch, 'tf32')
        precision.validate_precision_arch('V100', 'fp16')
        with self.assertRaises(ValueError):
            precision.validate_precision_arch('A100', 'fp32')

    def test_mesh_and_tile_validation(self):
        for size in (0, -16, 17, 50000):
            with self.subTest(size=size), self.assertRaises(ValueError):
                precision.validate_mesh_size(size)
        for tile in (0, -1, 3, 16):
            with self.subTest(tile=tile), self.assertRaises(ValueError):
                precision.validate_tile_size(160, tile)

    def test_sweep_uses_requested_size_and_precision_only(self):
        configurations = list(sweep.iter_configs(7200, 'tf32'))
        self.assertEqual(len(configurations), 32)
        self.assertEqual({row[0] for row in configurations}, {'tensor_tf32'})
        self.assertEqual({row[2] for row in configurations}, {7200})
        self.assertEqual({row[3] for row in configurations}, {1, 2, 3, 5, 6, 9, 10, 15})
        self.assertEqual(len(list(sweep.iter_configs(7200))), 64)
        with self.assertRaises(ValueError):
            sweep.compile_command_gen('A100', 'baseline', 'box2d1r', 160, 5, 'tf32')

    def test_raw_names_and_mixed_units(self):
        raw = Path('raw.txt')
        raw.write_text('tensor_tf32_box2d1r_160_1,1,us\n'
                       'tensor_tf32_box2d1r_160_5,900,ns\n')
        rows = sweep.load_result(raw)
        self.assertEqual(rows[0][0], 'tensor_tf32')
        sweep.store_best_gstencil_csv(rows, 'best.csv')
        with open('best.csv', newline='') as stream:
            best = list(csv.DictReader(stream))
        self.assertEqual(best[0]['tile size'], '5')
        self.assertAlmostEqual(float(best[0]['GStencil/s']), 160**2 / 900e-9 / 1e9, places=5)
        self.assertEqual(list(best[0]), FIELDS)

    def test_invalid_measurements_are_rejected(self):
        for duration in (0.0, -1.0, float('nan'), float('inf')):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                sweep.calc_gstencil(160, duration, 'us')
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                timed.gstencil_per_second(160, duration)
        with self.assertRaises(ValueError):
            sweep.calc_gstencil(160, 1, 'invalid')
        with self.assertRaises(ValueError):
            sweep.store_best_gstencil_csv([], 'empty.csv')
        self.assertFalse(Path('empty.csv').exists())

    def test_incomplete_or_wrong_precision_sweep_is_rejected(self):
        for compute_type in ('tensor_tf32', 'tensor'):
            with self.subTest(compute_type=compute_type), self.assertRaises(ValueError):
                sweep.store_best_gstencil_csv(
                    [(compute_type, 'box2d1r', '16', '1', '2', 'us')],
                    'incomplete.csv', sweep.iter_configs(16, 'tf32'))

    def test_best_tiles_require_matching_precision_and_size(self):
        path = Path('best.csv')
        self.write_best(path)
        self.assertEqual(timed.load_best_tiles(path, 160, 'tf32'), dict.fromkeys(timed.KERNELS, 5))
        with self.assertRaises(RuntimeError):
            timed.load_best_tiles(path, 160, 'fp16')
        with self.assertRaises(RuntimeError):
            timed.load_best_tiles(path, 320, 'tf32')
        with self.assertRaises(FileNotFoundError):
            timed.load_best_tiles('absent.csv', 160, 'tf32')

    def test_best_tiles_reject_invalid_and_duplicate_tiles(self):
        path = Path('best.csv')
        for tile in (0, 3, 16):
            self.write_best(path, tile=tile)
            with self.subTest(tile=tile), self.assertRaises(ValueError):
                timed.load_best_tiles(path, 160, 'tf32')
        self.write_best(path)
        with path.open('a') as stream:
            stream.write('tensor_tf32,box2d1r,160,5,1,us,25.6\n')
        with self.assertRaises(ValueError):
            timed.load_best_tiles(path, 160, 'tf32')

    def test_timing_parser(self):
        with patch.object(timed.subprocess, 'run') as run:
            run.return_value.stdout = '[Time] Time used: 1.25e-3 ms\n'
            run.return_value.stderr = ''
            self.assertEqual(timed.run_and_measure('box2d1r', 160, 5, 'tf32'), 0.00125)
            self.assertIn('/tf32/timed/', run.call_args.args[0][0])
            for value in ('0', '-1', 'nan', 'inf'):
                run.return_value.stdout = f'[Time] Time used: {value} ms\n'
                with self.subTest(value=value), self.assertRaises(ValueError):
                    timed.run_and_measure('box2d1r', 160, 5, 'tf32')
            run.return_value.stdout = 'missing timing'
            with self.assertRaises(RuntimeError):
                timed.run_and_measure('box2d1r', 160, 5, 'tf32')

    def test_ncu_commands(self):
        with patch.object(timed.subprocess, 'run') as run:
            timed.run_ncu_reports(160, dict.fromkeys(timed.KERNELS, 5), 'tf32')
            self.assertEqual(run.call_count, 4)
            for call, stencil in zip(run.call_args_list, timed.KERNELS):
                name = f'tensor_tf32_{stencil}_160_5'
                self.assertEqual(call.args[0], [
                    'ncu', '--set', 'full', '--launch-skip', '5', '--launch-count', '1',
                    '--force-overwrite', '-o', name, f'bin_ncu/tf32/timed/{name}',
                ])

    def test_ncu_sweep_failures_propagate(self):
        error = subprocess.CalledProcessError(1, ['ncu'])
        with patch.object(sweep.subprocess, 'run', side_effect=error) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                sweep.run_ncu_all(['tensor_tf32_box2d1r_16_1'], Path('raw.txt'), 'tf32')
            self.assertEqual(run.call_args.args[0][:3], ['bash', '-o', 'pipefail'])
            self.assertEqual(run.call_args.args[0][-1], './bin_ncu/tf32/sweep/')

    def test_sweep_workflow(self):
        names = sweep.expected_file_names('A100', 16, 'tf32')

        def fake_ncu(file_names, path, mode):
            self.assertEqual(mode, 'tf32')
            path.write_text(''.join(f'{name},2,us\n' for name in file_names))

        with patch.object(sys, 'argv', ['run_size_gstencil.py', '16', '--precision', 'tf32']), \
                patch.object(sweep, 'compile_all', return_value=names) as compile_all, \
                patch.object(sweep, 'run_ncu_all', side_effect=fake_ncu):
            sweep.main()
        compile_all.assert_called_once_with('A100', 16, 24, 'tf32')
        with Path('data/A100/layout16_tf32/gstencil_best_16.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 4)
        self.assertFalse(Path('data/A100/layout16').exists())

    def test_timed_workflow_keeps_fp16_defaults(self):
        for mode in ('fp16', 'tf32'):
            self.write_best(precision.data_dir('A100', mode) / 'gstencil_best_160.csv', mode)
            args = ['run_size_gstencil_ncu.py', '160']
            if mode == 'tf32':
                args.extend(['--precision', mode])
            with patch.object(sys, 'argv', args), \
                    patch.object(timed, 'compile_binaries') as compile_all, \
                    patch.object(timed, 'run_and_measure', return_value=0.125), \
                    patch.object(timed, 'run_ncu_reports') as profile:
                timed.main()
            tiles = dict.fromkeys(timed.KERNELS, 5)
            compile_all.assert_called_once_with('A100', 160, tiles, mode)
            profile.assert_called_once_with(160, tiles, mode)
            prefix = 'gstencil_tf32' if mode == 'tf32' else 'gstencil'
            with Path(f'{prefix}_160.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row['type'] for row in rows}, {precision.tensor_type(mode)})
            self.assertEqual(len(rows), 4)
            self.assertEqual(float(rows[0]['GStencil/s']), 160**2 / 0.125 / 1e6)

    def test_skip_options_use_existing_tf32_artifacts(self):
        path = precision.data_dir('A100', 'tf32') / 'gstencil_best_160.csv'
        self.write_best(path)
        args = ['run_size_gstencil_ncu.py', '160', '--precision', 'tf32',
                '--skip-compile', '--skip-time', '--skip-ncu']
        with patch.object(sys, 'argv', args), patch.object(timed.subprocess, 'run') as run:
            timed.main()
            run.assert_not_called()
        self.assertFalse(Path('gstencil_tf32_160.csv').exists())

    def test_architecture_rejected_before_work(self):
        for module in (sweep, timed):
            with patch.object(sys, 'argv', ['test', '160', '--arch', 'V100', '--precision', 'tf32']), \
                    patch.object(module.subprocess, 'run') as run, self.assertRaises(ValueError):
                module.main()
            run.assert_not_called()
        self.assertEqual(list(Path('.').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
