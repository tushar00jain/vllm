# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for split_group in GroupCoordinator.

These tests verify that:
1. split_group is used for both device and CPU group creation.
2. Multiple subgroups work correctly with split_group.
3. Both GPU and CPU all-reduce work on split groups.
"""

import os

import multiprocess as mp
import pytest
import torch
import torch.distributed

from vllm.distributed.parallel_state import (
    GroupCoordinator,
    init_distributed_environment,
)
from vllm.utils.system_utils import update_environment_variables

mp.set_start_method("spawn", force=True)


def distributed_run(fn, world_size):
    number_of_processes = world_size
    processes: list[mp.Process] = []
    for i in range(number_of_processes):
        env: dict[str, str] = {}
        env["RANK"] = str(i)
        env["LOCAL_RANK"] = str(i)
        env["WORLD_SIZE"] = str(number_of_processes)
        env["LOCAL_WORLD_SIZE"] = str(number_of_processes)
        env["MASTER_ADDR"] = "localhost"
        env["MASTER_PORT"] = "12346"
        p = mp.Process(target=fn, args=(env,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for p in processes:
        assert p.exitcode == 0


def worker_fn_wrapper(fn):
    def wrapped_fn(env):
        update_environment_variables(env)
        local_rank = os.environ["LOCAL_RANK"]
        device = torch.device(f"cuda:{local_rank}")
        torch.accelerator.set_device_index(device)
        init_distributed_environment()
        fn()

    return wrapped_fn


def _verify_device_group(coordinator: GroupCoordinator):
    """Verify device group works via all-reduce."""
    local_rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    tensor = torch.ones(16, 16, dtype=torch.float32, device=device)
    torch.distributed.all_reduce(tensor, group=coordinator.device_group)
    torch.accelerator.synchronize()
    expected = coordinator.world_size
    assert torch.all(tensor == expected).cpu().item(), (
        f"Device group all-reduce failed: expected {expected}, "
        f"got {tensor.flatten()[0].item()}"
    )


def _verify_cpu_group(coordinator: GroupCoordinator):
    """Verify CPU group works via all-reduce."""
    tensor = torch.ones(16, dtype=torch.float32)
    torch.distributed.all_reduce(tensor, group=coordinator.cpu_group)
    expected = coordinator.world_size
    assert torch.all(tensor == expected).cpu().item(), (
        f"CPU group all-reduce failed: expected {expected}, "
        f"got {tensor.flatten()[0].item()}"
    )


# ---------------------------------------------------------------------------
# Test 1: Basic split_group path with 2 GPUs
# ---------------------------------------------------------------------------
@worker_fn_wrapper
def split_group_basic_worker():
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    group_ranks = [list(range(world_size))]

    coordinator = GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=rank,
        torch_distributed_backend="nccl",
        use_device_communicator=False,
        group_name="test_split_basic",
    )

    _verify_device_group(coordinator)
    _verify_cpu_group(coordinator)


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run the test.",
)
def test_split_group_basic():
    """Test basic GroupCoordinator creation with split_group."""
    distributed_run(split_group_basic_worker, 2)


# ---------------------------------------------------------------------------
# Test 2: Multiple subgroups with split_group (4 GPUs)
# ---------------------------------------------------------------------------
@worker_fn_wrapper
def split_group_multiple_subgroups_worker():
    rank = torch.distributed.get_rank()
    group_ranks = [[0, 1], [2, 3]]

    coordinator = GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=rank,
        torch_distributed_backend="nccl",
        use_device_communicator=False,
        group_name="test_split_multi",
    )

    assert coordinator.world_size == 2

    _verify_device_group(coordinator)
    _verify_cpu_group(coordinator)

    if rank in [0, 1]:
        assert coordinator.ranks == [0, 1]
    else:
        assert coordinator.ranks == [2, 3]


@pytest.mark.skipif(
    torch.accelerator.device_count() < 4,
    reason="Need at least 4 GPUs to run the test.",
)
def test_split_group_multiple_subgroups():
    """Test GroupCoordinator with multiple independent subgroups."""
    distributed_run(split_group_multiple_subgroups_worker, 4)
