"""Helper to create a dummy pure-Python wheel for llama-cpp-python in CI.

Upstream abetlen macosx_11_0_arm64 wheels have a corrupt zip CRC for bundled
dylibs. The test suite uses conftest's FakeLLM and never invokes real llama_cpp.
Installing this stub satisfies the dependency in CI without downloading the
corrupted upstream wheel.
"""
import os
import shutil
import zipfile

dist = 'llama_cpp_python-0.3.32.dist-info'
os.makedirs('pkg/llama_cpp', exist_ok=True)
os.makedirs(f'pkg/{dist}', exist_ok=True)

with open('pkg/llama_cpp/__init__.py', 'w') as f:
    f.write('''__version__ = "0.3.32"
class Llama:
    pass
''')

with open(f'pkg/{dist}/METADATA', 'w') as f:
    f.write('''Metadata-Version: 2.1
Name: llama-cpp-python
Version: 0.3.32
''')

with open(f'pkg/{dist}/WHEEL', 'w') as f:
    f.write('''Wheel-Version: 1.0
Root-Is-Purelib: true
Tag: py3-none-any
''')

with open(f'pkg/{dist}/RECORD', 'w') as f:
    f.write('''llama_cpp/__init__.py,,
''')

whl = 'llama_cpp_python-0.3.32-py3-none-any.whl'
with zipfile.ZipFile(whl, 'w') as z:
    z.write('pkg/llama_cpp/__init__.py', 'llama_cpp/__init__.py')
    z.write(f'pkg/{dist}/METADATA', f'{dist}/METADATA')
    z.write(f'pkg/{dist}/WHEEL', f'{dist}/WHEEL')
    z.write(f'pkg/{dist}/RECORD', f'{dist}/RECORD')

shutil.rmtree('pkg')
print('Created stub wheel:', whl)
