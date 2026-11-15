"""Launch the native NVFP4 CLI in the validated local GPU runtime, one model at a time."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
CONTROLLER = Path('/home/cap/spark-inference-overnight-20261113')
CONTAINER = 'spark-prototype-quant-agent'
IMAGE = 'sha256:fa7334538384d67c72284edc081c78b1149a36d53aba1fb3c5f7d1d534154ec6'

def output(args):
    return subprocess.check_output(args, text=True, timeout=30).strip()

def main():
    locks = []
    for name in ('controller.lock', 'benchmark.lock'):
        stream = (CONTROLLER / name).open('a+')
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('The inference experiment is busy; wait for it to finish.')
        locks.append(stream)
    if output(['systemctl', '--user', 'show', 'spark-prototype-overnight-20261113.service', '--value', '-p', 'MainPID']) != '0':
        raise RuntimeError('The experiment controller is active.')
    for other in ('spark-prototype-agent', 'spark-prototype-model', 'spark-prototype-vllm-nvfp4'):
        if output(['docker', 'inspect', '--format', '{{.State.Running}}', other]) != 'false':
            raise RuntimeError('Another experiment container is active: ' + other)
    manifest = json.loads((PROJECT / 'benchmarks/model-manifests/qwen38-27b-nvfp4.json').read_text())
    model = Path(manifest['local_path'])
    downloaded = json.loads((model / 'DOWNLOAD_MANIFEST.json').read_text())
    if downloaded['revision'] != manifest['revision']:
        raise RuntimeError('Downloaded checkpoint revision differs from the validated manifest.')
    for item in manifest['verified_files']:
        if (model / item['name']).stat().st_size != item['bytes']:
            raise RuntimeError('Checkpoint file size differs: ' + item['name'])
    existing = subprocess.run(['docker', 'inspect', CONTAINER], text=True, capture_output=True, timeout=30)
    if existing.returncode:
        output(['docker', 'image', 'inspect', IMAGE])  # Require the installed, tested image; never pull implicitly.
        output(['docker', 'create', '--name', CONTAINER, '--init', '--gpus', 'all', '--network', 'none',
                '--shm-size', '1g', '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1',
                '-e', 'OMP_NUM_THREADS=4', '-v', str(PROJECT)+':/workspace:ro', '-v', str(model)+':/model:ro',
                '--entrypoint', '/bin/sh', IMAGE, '-c', 'trap "exit 0" TERM INT; sleep infinity & wait'])
        state = json.loads(output(['docker', 'inspect', CONTAINER]))[0]
    else:
        state = json.loads(existing.stdout)[0]
    if state['Image'] != IMAGE:
        raise RuntimeError('Existing native quantized container uses a different runtime image.')
    started_here = not state['State']['Running']
    try:
        if started_here:
            output(['docker', 'start', CONTAINER])
        processes = output(['docker', 'exec', CONTAINER, 'nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'])
        if processes and 'No running processes' not in processes:
            raise RuntimeError('GPU compute processes are already active:\n' + processes)
        interactive = ['-it'] if sys.stdin.isatty() and sys.stdout.isatty() else []
        bootstrap = ("import sys; sys.argv[0]='spark_engine'; import torch; torch.cuda.init(); "
                     "torch.cuda.set_per_process_memory_fraction(32*1024**3/torch.cuda.get_device_properties(0).total_memory); "
                     "from spark_engine.cli import main; main()")
        command = ['docker', 'exec', *interactive, '-w', '/workspace', CONTAINER,
                   'python3', '-c', bootstrap, '--model', '/model', '--chat', *sys.argv[1:]]
        def host_memory():
            fields = {}
            for line in Path('/proc/meminfo').read_text().splitlines():
                key, _, value = line.partition(':')
                if key in ('MemAvailable', 'SwapTotal', 'SwapFree'):
                    fields[key] = int(value.split()[0]) * 1024
            return fields['MemAvailable'], fields['SwapTotal'] - fields['SwapFree']
        _, initial_swap = host_memory()
        process = subprocess.Popen(command)
        while process.poll() is None:
            available, swap = host_memory()
            if available < 32 * 1024**3 or swap - initial_swap > 256 * 1024**2:
                output(['docker', 'stop', '--timeout', '10', CONTAINER])
                process.wait(timeout=30)
                raise RuntimeError('Native chat stopped by the host-memory/swap guard.')
            try:
                process.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                pass
            except KeyboardInterrupt:
                continue  # The attached CLI handles reply cancellation itself.
        return process.returncode
    finally:
        if started_here:
            subprocess.run(['docker', 'stop', '--timeout', '10', CONTAINER], timeout=30, check=True, stdout=subprocess.DEVNULL)

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
