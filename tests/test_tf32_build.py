import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'script'))
import run_size_gstencil_ncu as timed


def checked(command, timeout=180):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise AssertionError(f'Command failed: {command}\n{result.stdout}\n{result.stderr}')
    return result


class LayoutTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler unavailable')
    def test_cpu_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = str(Path(directory) / 'layout_test')
            checked(['c++', '-std=c++14', '-O2', '-Wall', '-Wextra', '-Werror',
                     'tests/tf32_layout.cpp', '-o', binary])
            result = checked([binary])
            self.assertEqual(result.stdout.count('CPU mapping cases passed'), 4)


@unittest.skipUnless(shutil.which('nvcc'), 'CUDA compiler unavailable')
class CudaTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.output = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_compile_and_tf32_instructions(self):
        for stencil in timed.KERNELS:
            with self.subTest(stencil=stencil):
                command = timed.compile_command('A100', stencil, 160, 5, 'tf32')
                command[command.index('-o') + 1] = str(self.output / stencil)
                checked(command)
                for arch in ('sm_80', 'sm_90'):
                    ptx = self.output / f'{stencil}_{arch}.ptx'
                    checked(['nvcc', '-O3', '-arch', arch, '-ptx',
                             f'layout16_tf32/kernel_{stencil}.cu', '-o', str(ptx)])
                    text = ptx.read_text()
                    self.assertRegex(text, r'mma\.sync[^\n]*\.tf32\.tf32')
                    self.assertIn('cvt.rna.tf32.f32', text)
                    self.assertNotIn('.f16', text)

    def test_sm70_compile_is_rejected(self):
        result = subprocess.run(
            ['nvcc', '-arch', 'sm_70', '-c', 'layout16_tf32/kernel_box2d1r.cu',
             '-o', str(self.output / 'unsupported.o')],
            cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('TF32 stencil kernels require compute capability 8.0', result.stderr)

    def test_gpu_correctness_and_sanitizers(self):
        probe_done = False
        sanitizer = shutil.which('compute-sanitizer')
        for stencil in timed.KERNELS:
            for tile in (1, 2, 5, 10, 15):
                binary = str(self.output / f'{stencil}_{tile}')
                checked(['nvcc', '-O3', '-arch', 'sm_80', '-DRUN_TIMES=1',
                         f'-DTEST_RADIUS={stencil[-2]}', f'-DTEST_STAR={int(stencil.startswith("star"))}',
                         f'-DTILE_SIZE={tile}', 'tests/tf32_cuda.cu',
                         f'layout16_tf32/host_{stencil}.cu',
                         f'layout16_tf32/kernel_{stencil}.cu', '-o', binary])
                if not probe_done:
                    probe = subprocess.run([binary, '--probe'], capture_output=True, text=True, timeout=30)
                    if probe.returncode == 77:
                        self.skipTest(probe.stdout.strip())
                    self.assertEqual(probe.returncode, 0, probe.stdout + probe.stderr)
                    probe_done = True
                with self.subTest(stencil=stencil, tile=tile):
                    result = checked([binary])
                    self.assertIn('GPU cases passed', result.stdout)
                    if sanitizer and tile == 2:
                        for tool in ('memcheck', 'racecheck', 'synccheck'):
                            checked([sanitizer, '--tool', tool, '--error-exitcode', '1', binary], timeout=300)


if __name__ == '__main__':
    unittest.main()
