# NOTE: this file must import only modules from the standard library.

import os
from pathlib import Path

_PROJECT_DIR: None | Path = None
_PYPROJECT_FILE_NAME = 'pyproject.toml'


def get_project_dir() -> Path:
    global _PROJECT_DIR

    if _PROJECT_DIR is None:
        path = Path.cwd().resolve()
        pyproject_path = path / _PYPROJECT_FILE_NAME
        while str(path) != path.root and not pyproject_path.exists():
            path = path.parent
            pyproject_path = path / _PYPROJECT_FILE_NAME

        if pyproject_path.exists():
            if pyproject_path.parent != Path(__file__).resolve().parent.parent:
                raise RuntimeError(
                    'Failed to find the project directory. '
                    f' Most likely, you are running the code'
                    ' in a virtual environment of a different project,'
                    f' namely, of this one: {pyproject_path.parent}'
                )
            _PROJECT_DIR = path
        else:
            raise RuntimeError(
                'Failed to find the project directory.'
                ' Most likely, you are outside of the project directory.'
            )
    return _PROJECT_DIR


def get_exp_dir() -> Path:
    return get_project_dir() / 'exp'


def get_cache_dir() -> Path:
    path = get_project_dir() / 'cache'
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dir() -> Path:
    return get_project_dir() / 'data'


def get_local_dir() -> Path:
    return get_project_dir() / 'local'


def get_snapshot_dir() -> None | Path:
    snapshot_dir = os.environ.get('SNAPSHOT_PATH')
    return Path(snapshot_dir).resolve() if snapshot_dir else None


def get_tmp_output_dir() -> None | Path:
    tmp_output_dir = os.environ.get('TMP_OUTPUT_PATH')
    return Path(tmp_output_dir).resolve() if tmp_output_dir else None


def is_local() -> bool:
    return get_snapshot_dir() is None
