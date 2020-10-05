from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fitness_fixer',
    ext_modules=[
        CUDAExtension('fitness_fixer_cuda', [
            'fitness_nonlamarckian.cpp',
            'fitness_nonlamarckian_cuda.cu',
        ],)
        #  extra_compile_args={
            #  'cxx': [],
            #  'nvcc': ['-g','-G']
        #  })
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
