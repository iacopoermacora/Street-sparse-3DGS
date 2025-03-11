from setuptools import setup, Extension
import pybind11

ctm_exporter = Extension(
    "ctm_exporter",
    sources=["ctm_exporter.cpp"],
    include_dirs=[pybind11.get_include()],
    libraries=["openctm"],
    extra_compile_args=["-std=c++11"]
)

setup(
    name="ctm_exporter",
    version="0.1",
    description="CTM exporter module",
    ext_modules=[ctm_exporter]
)
