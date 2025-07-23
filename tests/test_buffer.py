import os

import time

import torch
import torch_npu
import torch.distributed as dist

from deep_ep import Buffer

class FakeGroup:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    def size(self):
        return self._size

    def rank(self):
        return self._rank

def test_main():
    print('test_main')
    ascend_path = os.getenv('ASCEND_CUSTOM_OPP_PATH')
    print("ASCEND_CUSTOM_OPP_PATH: ", ascend_path)

    lib_path = os.getenv('LD_LIBRARY_PATH')
    print("LD_LIBRARY_PATH: ", lib_path)

    torch.npu.set_device(0)
    device_group = FakeGroup(0, 1)

    print('----------')
    buffer = Buffer(device_group)
    print(f'low_latency_rdma_size_hint: {buffer.get_low_latency_rdma_size_hint(1,2,3,4)}')

if __name__ == '__main__':
    test_main()