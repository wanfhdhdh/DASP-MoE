import math
import pickle
import torch
from torch import distributed as dist
from torch.utils import data

LOCAL_PROCESS_GROUP = None

def is_primary():
    return get_rank() == 0

def get_rank():
    if not dist.is_available(): return 0
    if not dist.is_initialized(): return 0
    return dist.get_rank()

def get_local_rank():
    if not dist.is_available(): return 0
    if not dist.is_initialized(): return 0
    if LOCAL_PROCESS_GROUP is None:
        # 这里为了防止报错，如果没有初始化组，返回0即可
        # raise ValueError("tensorfn.distributed.LOCAL_PROCESS_GROUP is None")
        return 0
    return dist.get_rank(group=LOCAL_PROCESS_GROUP)

def synchronize():
    if not dist.is_available(): return
    if not dist.is_initialized(): return
    world_size = dist.get_world_size()
    if world_size == 1: return
    dist.barrier()

def get_world_size():
    if not dist.is_available(): return 1
    if not dist.is_initialized(): return 1
    return dist.get_world_size()

def all_reduce(tensor, op=dist.ReduceOp.SUM):
    world_size = get_world_size()
    if world_size == 1: return tensor
    dist.all_reduce(tensor, op=op)
    return tensor

def all_gather(data):
    world_size = get_world_size()
    if world_size == 1: return [data]
    
    # ... (保持原本的序列化逻辑) ...
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")
    
    local_size = torch.IntTensor([tensor.numel()]).to("cuda")
    size_list = [torch.IntTensor([1]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
        
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), 0)
        
    dist.all_gather(tensor_list, tensor)
    
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))
        
    return data_list

# ... (reduce_dict 和 data_sampler 保持不变) ...
def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)