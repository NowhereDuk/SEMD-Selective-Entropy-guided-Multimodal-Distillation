# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
import shutil
import socket
import sys
import tempfile

from . import USER_CONFIG_DIR
from .torch_utils import TORCH_1_9


def find_free_network_port() -> int:
    """
    Finds a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]  # port


def generate_ddp_file(trainer):
    """Generates a DDP file and returns its file name."""
    module, name = f'{trainer.__class__.__module__}.{trainer.__class__.__name__}'.rsplit('.', 1)
    cwd = os.getcwd()
    extra_overrides = getattr(trainer, 'ddp_extra_overrides', {})

    content = f"""
# Ultralytics Multi-GPU training temp file (should be automatically deleted after use)
import os
import sys
import traceback

cwd = {cwd!r}
os.chdir(cwd)
if cwd not in sys.path:
    sys.path.insert(0, cwd)

overrides = {vars(trainer.args)}
extra_overrides = {extra_overrides}
overrides.update(extra_overrides)

if __name__ == "__main__":
    import inspect

    from {module} import {name}
    from ultralytics.utils import DEFAULT_CFG_DICT

    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')   # handle the extra key 'save_dir'
    trainer_kwargs = {{'overrides': overrides}}
    if 'cfg' in inspect.signature({name}).parameters:
        trainer_kwargs['cfg'] = cfg
    try:
        trainer = {name}(**trainer_kwargs)
        results = trainer.train()
    except Exception:
        traceback.print_exc()
        raise
"""
    (USER_CONFIG_DIR / 'DDP').mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='_temp_',
                                     suffix=f'{id(trainer)}.py',
                                     mode='w+',
                                     encoding='utf-8',
                                     dir=USER_CONFIG_DIR / 'DDP',
                                     delete=False) as file:
        file.write(content)
    return file.name


def generate_ddp_command(world_size, trainer):
    """Generates and returns command for distributed training."""
    import __main__  # noqa local import to avoid https://github.com/Lightning-AI/lightning/issues/15218
    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)  # remove the save_dir
    file = generate_ddp_file(trainer)
    dist_cmd = 'torch.distributed.run' if TORCH_1_9 else 'torch.distributed.launch'
    port = find_free_network_port()
    cmd = [sys.executable, '-m', dist_cmd, '--nproc_per_node', f'{world_size}', '--master_port', f'{port}', file]
    return cmd, file


def ddp_cleanup(trainer, file):
    """Delete temp file if created."""
    if f'{id(trainer)}.py' in file:  # if temp_file suffix in file
        os.remove(file)
