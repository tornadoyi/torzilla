import subprocess
import shlex
import sys
import time


class Process(object):
    def __init__(
        self,
        cmd,
        stdin=None,
        stdout=None,
        stderr=None,
        preexec_fn=None,
        shell=False,
        **kwargs
    ):
        cmds = cmd.split('|')
        self._ps = []
        for i, cmd in enumerate(cmds):
            p = subprocess.Popen(
                shlex.split(cmd),
                stdin=stdin,
                stdout=subprocess.PIPE if i < len(cmds)-1 else stdout,
                stderr=stderr,
                preexec_fn=preexec_fn if i == 0 else None,
                shell=shell if i == 0 else False,
                **kwargs
            )
            self._ps.append(p)
            stdin = p.stdout

    @property
    def stdin(self): return self._ps[0].stdin

    @property
    def stdout(self): return self._ps[-1].stdout

    @property
    def stderr(self): return self._ps[-1].stderr

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, value, traceback):
        for p in self._ps:
            p.__exit__(exc_type, value, traceback)

    def poll(self):
        return [p.poll() for p in self._ps]

    def wait(self, timeout=None):
        st = time.time()
        rets = []
        for p in self._ps:
            remain = None if timeout is None else max(st + timeout - time.time(), 0)
            r = p.wait(timeout=remain)
            rets.append(r)
        return rets

    def terminate(self):
        for p in self._ps[-1::-1]:
            p.terminate()

    kill = terminate


def run(cmd, capture_output=False, timeout=None, check=False, **kwargs):
    
    if input is not None:
        if kwargs.get('stdin') is not None:
            raise ValueError('stdin and input arguments may not both be used.')
        kwargs['stdin'] = subprocess.PIPE

    if capture_output:
        if kwargs.get('stdout') is not None or kwargs.get('stderr') is not None:
            raise ValueError('stdout and stderr arguments may not be used '
                             'with capture_output.')
        kwargs['stdout'] = subprocess.PIPE
        kwargs['stderr'] = subprocess.PIPE
    
    with Process(*cmd, **kwargs) as process:
        process.wait(timeout)
        retcodes = process.poll()
        sum_code = sum(retcodes)
        stdout = process.stdout.read()
        stderr = process.stderr.read()

    if check and sum_code:
        raise subprocess.CalledProcessError(
            retcodes, cmd,
            output=stdout, stderr=stderr
        )
    
    return subprocess.CompletedProcess(process.args, retcodes, stdout, stderr)


def check_output(*popenargs, timeout=None, **kwargs):
    if 'stdout' in kwargs:
        raise ValueError('stdout argument not allowed, it will be overridden.')

    return run(*popenargs, stdout=subprocess.PIPE, timeout=timeout, check=True,
               **kwargs).stdout