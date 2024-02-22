project('example', 'c')

py3 = import('python').find_installation(pure: false)
py3_dep = py3.dependency()

incdir_numpy = run_command(py3,
  [
    '-c',
    'import os, numpy; print(os.path.relpath(numpy.get_include()))'
  ],
  check: true,
).stdout().strip()

np_dep = declare_dependency(include_directories: incdir_numpy)

py3.extension_module(
    'example',
    'example.c',
    # include_directories: inc_np,
    dependencies: [py3_dep, np_dep],
)