from torzilla.core import utility as U
from .process import MainProcess


def lanuch(**kwargs):
    mainproc = _load_main_proc(kwargs.get('mainproc', None))
    try:
        proc = None
        proc = mainproc(**kwargs)
        proc.start()
    except Exception as e:
        raise e
    finally:
        if proc: proc.exit()


def _load_main_proc(path):
    mainproc = path
    U.assert_type(mainproc, str)
    if mainproc is None:
        mainproc = MainProcess
    elif isinstance(mainproc, str):
        mainproc = U.import_type(mainproc)
    U.assert_subclass(mainproc, MainProcess)
    return mainproc