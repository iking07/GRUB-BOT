from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_attention',
    ext_modules=[
        CUDAExtension('custom_attention', [
            'csrc/attention.cpp',
            'csrc/attention_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
