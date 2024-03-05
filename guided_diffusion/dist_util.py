"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist(devices=(0,)):
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    try:
        device_string = ','.join(map(str, devices))
    except TypeError:
        device_string = str(devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = device_string #f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"

    #comm = MPI.COMM_WORLD
  #  print('commworld, 'f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}", comm)
    backend = "gloo" if not th.cuda.is_available() else "nccl"
   # print('commrank', comm.rank)
   # print('commsize', comm.size)

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = '127.0.1.1'#comm.bcast(hostname, root=0)
    os.environ["RANK"] = '0'#str(comm.rank)
    os.environ["WORLD_SIZE"] = '1'#str(comm.size)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    # print('port2', port)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev(device_number=0):
    """
    Get the device to use for torch.distributed.
    """
    if isinstance(device_number, (list, tuple)):  # multiple devices specified
        return [dev(k) for k in device_number]    # recursive call
    if th.cuda.is_available():
        device_count = th.cuda.device_count()
        if device_count == 1:
            return th.device(f"cuda")
        else:
            if device_number < device_count:  # if we specify multiple devices, we have to be specific
                return th.device(f'cuda:{device_number}')
            else:
                raise ValueError(f'requested device number {device_number} (0-indexed) but only {device_count} devices available')
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    #print('mpicommworldgetrank', MPI.COMM_WORLD.Get_rank())
    mpigetrank=0
   # if MPI.COMM_WORLD.Get_rank() == 0:
    if mpigetrank==0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
   # data = MPI.COMM_WORLD.bcast(data)
  #  print('mpibacst', MPI.COMM_WORLD.bcast(data))
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    #for p in params:
    #    with th.no_grad():
    #        dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
