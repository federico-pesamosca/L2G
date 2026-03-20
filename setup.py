from typing import Any
from setuptools import setup
from setuptools.extension import Extension
import sys
import os

from pathlib import Path

try:
    from Cython.Build import cythonize
except ImportError:
    sys.exit("Cython not found. Cython is needed to build the extension " +
             "modules.")

try:
    import numpy as np
except ImportError:
    sys.exit("Numpy not found. Numpy is needed to build the extension " +
             "modules.")

USE_OPENMP = True
USE_AVX2 = True
USE_FMA = True

def get_source_files(pyx_file: str) -> list[str]:
    """Apparently the easiest way of putting a commong cpp library is just
    shoving the cpp files into the source list for the Extensions. However this
    annoyingly extends the installation of the package.

    Having a reduced set of files at least lowers it a bit.
    """
    out: list[str] = [pyx_file]
    if sys.platform == "win32":
        return out + ['cpp/rkf45.cpp',
                      'cpp/bicubic.cpp',
                      'cpp/tlas.cpp',
                      'cpp/flt.cpp']

    base_name = os.path.basename(pyx_file)
    match base_name:
        case '_field_line_tracer.pyx':
            out.append('cpp/bicubic.cpp')
            out.append('cpp/tlas.cpp')
            out.append('cpp/flt.cpp')
        case 'rkf45.pyx':
            out.append('cpp/rkf45.cpp')
            out.append('cpp/bicubic.cpp')
        case 'bicubic.pyx':
            out.append('cpp/bicubic.cpp')
        case 'equilibrium_analysis.pyx':
            out.append('cpp/rkf45.cpp')
            out.append('cpp/bicubic.cpp')
        case 'tlas.pyx':
            out.append('cpp/tlas.cpp')
        case 'bfgs_2d.pyx':
            out.append('cpp/bicubic.cpp')
        case 'flt.pyx':
            out.append('cpp/bicubic.cpp')
            out.append('cpp/tlas.cpp')
            out.append('cpp/flt.cpp')
        case _:
            raise Exception(f'Pyx file {base_name} not processed in case')
    return out

def get_extra_macros() -> list[tuple[str, Any]]:
    return [('flt_EXPORTS', None)]

def get_include_directories() -> list[str]:
    """Get the include directories.
    """
    out: list[str] = []

    out.append(np.get_include())
    out.append('cpp')
    return out

def get_libraries() -> list[str]:
    """Get libraries required for linking against flt.so
    """
    out: list[str] = []
    return out

def get_library_dirs() -> list[str]:
    """Get paths to library dirs.
    """
    out: list[str] = []
    return out

def get_extra_compile_args() -> list[str]:
    """Specify extra compilation arguments. Use maximum optimization but avoid
    using march=native as it shown that on some CPUs it can generate segfaulty
    code.
    """
    out: list[str] = []

    if sys.platform == "win32":
        out.append("/std:c++17")
        out.append("/EHsc")
        out.append("/O2")
        if USE_OPENMP:
            out.append("/openmp")
        out.append("/bigobj")
        # Conversion warning
        out.append("/wd4305")
        out.append("/wd4244")
        # dll-interface warning
        out.append("/wd4267")
        out.append("/wd4251")
        if USE_AVX2:
            out.append("/arch:AVX2")
    else:
        out.append("-O3")
        out.append("-march=native")
        out.append("-Wall")
        out.append("-std=gnu++17")
        out.append("-ftree-vectorize")
        if USE_AVX2:
            out.append("-mavx2")
        if USE_FMA:
            out.append("-mfma")
        if USE_OPENMP:
            out.append("-fopenmp")
    return out

def get_extra_link_args() -> list[str]:
    """Extra link flags. For OpenMP or parts of the code in cython that might
    use openmp.
    """
    out: list[str] = []

    if sys.platform == "win32":
        pass

    else:
        if USE_OPENMP:
            out.append("-fopenmp")
        # Reduce size
        out.append("-Wl,--strip-all")
    return out

def get_pyx_files(path: Path) -> list[str]:
    """Collect all the pyx files
    """
    dir_exclusion = ["__pycache__"]
    pyx_files: list[str] = []

    for p in path.glob("*"):
        if p.is_dir():
            if p.stem in dir_exclusion:
                continue
            pyx_files += get_pyx_files(p)
        if p.suffix == ".pyx":
            pyx_files.append(p)
    return pyx_files

def prepare_cython_extensions(pyx_files: list[Path], root_path: Path) -> list[Extension]:
    """Automatically setup the list of extensions. Basically every pyx file in
    the package path directory get's properly defined as an Extension and then
    cythonized.
    """
    extensions: list[Extension] = []
    for p in pyx_files:
        rel_path = p.relative_to(root_path)
        source_file_path = str(rel_path)
        # Correct module name so that it will not have src at the beginning.
        # Collect the parts of the relative path
        parts = rel_path.with_suffix("").parts
        # Remove the "src" and replace it with ModuleName
        module_name = ".".join(list(parts))
        print(f"Cythonizing {source_file_path} as module {module_name}...")
        # Create extensions
        ext = Extension(
            module_name,
            sources=get_source_files(source_file_path),
            include_dirs=get_include_directories(),
            libraries=get_libraries(),
            library_dirs=get_library_dirs(),
            extra_compile_args=get_extra_compile_args(),
            extra_link_args=get_extra_link_args(),
            language="c++",
            define_macros=get_extra_macros()
        )

        # Cythonize returns a list.
        # Add embedsignature so that the functions are visible by Sphinx.
        extensions += cythonize(ext, annotate=True,
                                compiler_directives={'embedsignature': True})

    return extensions

# Collect the pyx files and create extensions from them
setup_path = Path(__file__).resolve().parent
package_path = setup_path / "l2g"
pyx_files = get_pyx_files(package_path)
extensions = prepare_cython_extensions(pyx_files, setup_path)

scripts = ['bin/flat', 'bin/submitFLAT', 'bin/FLAT.sbatch', 'bin/imas2eqdsk',
           'bin/mkEqdskMovie', 'bin/mkImasMovie', 'bin/mkImasMovieFromPsi',
           'bin/med2mat', 'bin/mat2med', 'bin/plotMFlux', 'bin/plotIP',
           'bin/get_disruption_profile_from_imas', 'bin/torbeam_mapper',
           'bin/torbeam_plotter', 'bin/get_disruption_profile_from_imas',
           'bin/mkDisruptionMovie', 'bin/plotEquilibriums',
           'bin/plot_summary_plasma_type']

# Additional cmd scripts if on windows.
if sys.platform == "win32":
    scripts.append('bin/flat.cmd')
    scripts.append('bin/plotMFlux.cmd')

setup(
    ext_modules=extensions,
    packages = ["l2g", "l2g.comp", "l2g.equil", "l2g.hlm", "l2g.plot",
                "l2g.settings", "l2g.mesh", "l2g.external", "l2g.mesh.medio"],
    package_data={"": ["*.pyi"]},
    exclude_package_data={"": ["*.cpp"]},
    scripts = scripts
)
