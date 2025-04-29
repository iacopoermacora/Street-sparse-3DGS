"""
This file will be automatically generated for each project with python unit tests

WARNING: This file should not be used in normal executables for neither debug nor release.
"""
import os
import pathlib
import sys


def GetBuildDir():
    build_config = 'debug' if '--debug' in sys.argv else 'release'
    if 'CF_PY_TEST_BUILD_CONFIG' in os.environ:
        build_config = os.environ['CF_PY_TEST_BUILD_CONFIG'].lower()
        if build_config not in ['debug', 'release']:
            raise ValueError('Environment variable CF_PY_TEST_BUILD_CONFIG does not have correct value {{debug|release}}. Got "{}"'.format(build_config))
    build_dir = os.path.join(GetProjectRootDir(), 'build', build_config)
    if os.path.isdir(build_dir):
        return build_dir
    raise ImportError('Select C++ build directory does not exist ("{}")'.format(build_dir))


def GetCurrentSourceDir():
    return '/home/local/CYCLOMEDIA001/iermacora/projects/cityfusion_master/src/active/cityfusion/tasks/billboards/billboards_prepare_data' + os.sep


def GetCurrentSourceDirPath() -> pathlib.Path:
    """Returns the directory of the current cmake project"""
    return pathlib.Path(GetCurrentSourceDir())


def GetProjectRootDir():
    return os.path.normpath('/home/local/CYCLOMEDIA001/iermacora/projects/cityfusion_master') + os.sep


def GetProjectRootDirPath() -> pathlib.Path:
    """Returns project root (containing src/{active|experimental})"""
    return pathlib.Path('/home/local/CYCLOMEDIA001/iermacora/projects/cityfusion_master')


def _AddCityfusionPythonLibsToImportPaths():
    is_cf_path_found = False

    try:
        base_path = GetProjectRootDir()

        libs_dir = os.path.join(base_path, 'src/active/cityfusion/libs')
        _AddLibraryDirectoriesToSearchPath(libs_dir)
        for non_active in ['experimental']:
            if non_active in os.getcwd() or non_active in __file__:
                non_active_libs_dir = libs_dir.replace('/active/', '/' + non_active + '/')
                _AddLibraryDirectoriesToSearchPath(non_active_libs_dir)
        is_cf_path_found = True
    except ImportError:
        pass

    # NOTE: Add alternative methods of searching and adding paths here. If found set is_cf_path_found to True

    if not is_cf_path_found:
        raise ImportError('Failed to find both cf on python path and in special tree')


def _AddLibraryDirectoriesToSearchPath(libraries_directory):
    for lib_dir in os.listdir(libraries_directory):
        if lib_dir.startswith('cf_py'):
            sys.path.append(os.path.join(libraries_directory, lib_dir, 'src'))


def _AddCurrentProjectSrcDirToImportPaths():
    """ Add the <current_sub_dir>/src to the python import path. This path is typically needed for unit tests """
    import_dir = os.path.join(GetCurrentSourceDir(), 'src')
    if import_dir not in sys.path:
        sys.path.append(import_dir)


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
    # Also, add system's site-packages in order to include libraries which we build from source (such as tripy)
    sys.path.append('/usr/local/lib/python{version}/site-packages'.format(version=version))


_AddThirdPartyLocalInstallPath()
_AddCityfusionPythonLibsToImportPaths()
_AddCurrentProjectSrcDirToImportPaths()

# Break coding style: Path is only included after above function calls()
import cf_test_helper.io.filesystem.helper_register_filesystem
cf_test_helper.io.filesystem.helper_register_filesystem.RegisterManagedIdentityAsFilesystem()
