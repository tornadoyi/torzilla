from torzilla.core import utility as U
from .process import MainProcess, Subprocess
from .manager import Manager


def lanuch(
    num_process=0,
    mainproc=None,
    manager=None,
    subproc=None,
    subproc_args=None,
    **shared_args,
):
    # check
    U.assert_type(num_process, int)
    U.assert_type(subproc_args, list, tuple, null=True)
    mainproc_type = _import_module(mainproc, MainProcess)
    subproc_type = _import_module(subproc, Subprocess)
    manager_type = _import_module(manager, Manager)
    if subproc_args is not None and len(subproc_args) != num_process:
        raise Exception(f'Number of subproc args must be equal to num_process, {len(subproc_args)} != {num_process}')
    
    # subproc args
    subproc_args = [{}] * num_process if subproc_args is None else subproc_args
    for args in subproc_args:
        args.update(dict(
            subproc = _import_module(args.get('subproc', subproc_type), Subprocess)
        ))
        args.update(shared_args)

    # mainproc args
    mainproc_args = {}
    mainproc_args.update(dict(
        mainproc = mainproc_type,
        manager = manager_type,
        subproc_args = subproc_args,
    ))

    # start
    proc = mainproc_type(**mainproc_args)
    with proc:
        pass
    return proc

def _import_module(mainproc, dft_type):
    U.assert_type(mainproc, str, type, null=True)
    if mainproc is None:
        mainproc = dft_type
    elif isinstance(mainproc, str):
        mainproc = U.import_type(mainproc)
    U.assert_subclass(mainproc, dft_type)
    return mainproc