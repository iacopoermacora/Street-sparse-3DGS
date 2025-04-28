"""
Import helper for cf_py, cf_py_ccc, etc

DO NOT CHANGE. THIS IS AUTOMATICALLY GENERATED.

src/active/support/build_system/cmake/template_cf_py_importer.py.in

Does not import, but adds paths to the python path where you can find the cf_py and cf_py_cc libraries
"""
import os
import pathlib
import sys


class _NotFoundError(Exception):
    pass


def _DetermineProjectBasePath():
    initial_paths = [os.getcwd(), __file__]

    for initial_path in initial_paths:
        try:
            path_list = initial_path.split(os.sep)
            assert len(path_list) > 0, 'Failed to find cf_py project path'
            while path_list[-1] != 'cityfusion':
                path_list = path_list[:-1]
                assert len(path_list) > 0, 'Failed to find cf_py project path'
            if len(path_list) > 2 and path_list[-2] in ['experimental', 'deprecated']:  # For experimental or deprecated, change into active
                path_list[-2] = 'active'
            return os.sep.join(path_list)
        except AssertionError:
            pass
    raise _NotFoundError('Failed to find path to cityfusion project base')


def _AddCityfusionImportPaths():
    _AddThirdPartyLocalInstallPath()
    _AddCityfusionPythonWrapperPath()

    is_cf_path_found = False

    try:
        base_path = _DetermineProjectBasePath()

        libs_dir = os.path.join(base_path, 'libs')
        # Test that the libs subdirectory indeed exists. We might have arrived in /opt/cityfusion
        if os.path.isdir(libs_dir):
            _AddLibraryDirectoriesToSearchPath(libs_dir)
        for non_active in ['experimental']:
            if non_active in os.getcwd() or non_active in __file__:
                non_active_libs_dir = libs_dir.replace('/active/', '/' + non_active + '/')
                if os.path.isdir(non_active_libs_dir):
                    _AddLibraryDirectoriesToSearchPath(non_active_libs_dir)

        is_cf_path_found = True
    except ImportError:
        pass
    except _NotFoundError:
        pass

    if not is_cf_path_found:
        for python_path in sys.path:
            if os.path.isfile(os.path.join(python_path, 'cf', '__init__.py')):
                is_cf_path_found = True

    if not is_cf_path_found:
        raise ImportError('Failed to find both cf on python path and in special tree')

    _AddParentPath()


def _AddLibraryDirectoriesToSearchPath(libraries_directory):
    for lib_dir in os.listdir(libraries_directory):
        if lib_dir.startswith('cf_py'):
            sys.path.append(os.path.join(libraries_directory, lib_dir, 'src'))


def _AddThirdPartyLocalInstallPath():
    version = '{}.{}'.format(sys.version_info.major, sys.version_info.minor)
    # This is the local directory for the clone and is not valid during when deployed.
    user_third_party_install_dir = os.path.normpath('/home/local/CYCLOMEDIA001/iermacora/projects/cityfusion_master/build/third_party/opt/cityfusion')
    deployed_install_dir = '/opt/cityfusion'
    sub_path = 'lib/python{version}/site-packages/'.format(version=version)
    if os.path.isdir(user_third_party_install_dir):
        full_path = os.path.join(user_third_party_install_dir, sub_path)
    else:
        full_path = os.path.join(deployed_install_dir, sub_path)
    # Note that the full path might not exist if the configuration does not include locally installed python libraries
    if full_path not in sys.path:
        sys.path.append(full_path)


def _AddCityfusionPythonWrapperPath():
    try:
        cityfusion_path = pathlib.Path(_DetermineProjectBasePath())
    except _NotFoundError:
        # Assume that we're in the production docker container and cf_cc et al. are on the python path
        return
    project_root = cityfusion_path / '../../../'
    wrapper_path = project_root / 'build/python_wrappers'
    if wrapper_path.is_dir():  # In docker container, this directory does not exist.
        for sub_project_path in wrapper_path.iterdir():
            if (sub_project_path / 'pyproject.toml').is_file():
                sys.path.append(str(sub_project_path))


def _AddParentPath():
    initial_paths = [os.getcwd(), __file__]
    for initial_path in initial_paths:
        sys.path.append(os.path.dirname(os.path.dirname(initial_path)))


if os.environ.get("CF_PYTHON_IMPORTER_EXECUTED", "") != "TRUE":
    os.environ["CF_PYTHON_CF_IMPORTER_EXECUTED"] = "TRUE"
    _AddCityfusionImportPaths()
