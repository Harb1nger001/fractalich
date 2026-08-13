from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

import os
import platform
import subprocess
import sys


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):

    def build_extension(self, ext):

        # Root of the source distribution / repository
        source_dir = os.path.abspath(os.path.dirname(__file__))

        # Temporary build directory used by setuptools
        build_dir = os.path.abspath(
            os.path.join(self.build_temp, "cmake_build")
        )
        os.makedirs(build_dir, exist_ok=True)

        # Directory where setuptools expects the extension
        ext_dir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name))
        )
        os.makedirs(ext_dir, exist_ok=True)

        cmake_args = [
            f"-DCMAKE_INSTALL_PREFIX={ext_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DPYTHON_VERSION={sys.version_info.major}.{sys.version_info.minor}",
        ]

        build_args = ["--config", "Release"]

        if platform.system() == "Windows":
            cmake_args += ["-A", "x64"]

        print("=" * 70)
        print("Building Fractalich C++ extension")
        print(f"Source directory : {source_dir}")
        print(f"Build directory  : {build_dir}")
        print(f"Install directory: {ext_dir}")
        print("=" * 70)

        # Configure
        subprocess.check_call(
            ["cmake", source_dir] + cmake_args,
            cwd=build_dir,
        )

        # Build + install
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "install"] + build_args,
            cwd=build_dir,
        )


setup(
    ext_modules=[
        CMakeExtension("fractalich.fractalich_ext")
    ],
    cmdclass={
        "build_ext": CMakeBuild
    },
    packages=["fractalich"],
    zip_safe=False,
)