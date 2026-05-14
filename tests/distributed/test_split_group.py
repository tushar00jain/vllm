# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for split_group in GroupCoordinator.

These tests verify that:
1. split_group is used for both device and CPU group creation.
2. Multiple subgroups work correctly with split_group.
3. Both GPU and CPU all-reduce work on split groups.
4. All collective operations work correctly through split groups
   with both use_device_communicator=True and False modes.
"""

import os

import multiprocess as mp
import pytest
import torch
import torch.distributed

import vllm.envs as envs
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    init_distributed_environment,
)
from vllm.utils.system_utils import update_environment_variables

# The whole module exercises the split_group code path, which is opt-in
# behind VLLM_USE_SPLIT_GROUP=1.
pytestmark = pytest.mark.skipif(
    not envs.VLLM_USE_SPLIT_GROUP,
    reason="VLLM_USE_SPLIT_GROUP=1 not set; split_group path is opt-in.",
)

mp.set_start_method("spawn", force=True)


def distributed_run(fn, world_size, extra_env: dict[str, str] | None = None):
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
        # propagate the opt-in flag to the spawned child workers
        env["VLLM_USE_SPLIT_GROUP"] = "1"
        if extra_env:
            env.update(extra_env)
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


def _get_use_device_communicator() -> bool:
    return os.environ.get("USE_DEVICE_COMMUNICATOR", "False") == "True"


def _create_coordinator(
    use_device_communicator: bool, group_name: str
) -> GroupCoordinator:
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    return GroupCoordinator(
        group_ranks=[list(range(world_size))],
        local_rank=rank,
        torch_distributed_backend="nccl",
        use_device_communicator=use_device_communicator,
        group_name=group_name,
    )


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


# ---------------------------------------------------------------------------
# CPU Group Tests (Tests 3-5)
# ---------------------------------------------------------------------------
@worker_fn_wrapper
def broadcast_object_worker():
    use_dc = _get_use_device_communicator()
    coordinator = _create_coordinator(use_dc, "test_broadcast_object")
    obj = {"key": "value", "rank": 0}
    if coordinator.rank_in_group == 0:
        result = coordinator.broadcast_object(obj, src=0)
    else:
        result = coordinator.broadcast_object(src=0)
    assert result == obj, (
        f"broadcast_object failed: expected {obj}, got {result}"
    )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run the test.",
)
@pytest.mark.parametrize("use_device_communicator", [True, False])
def test_broadcast_object(use_device_communicator):
    """Test broadcast_object through split group."""
    distributed_run(
        broadcast_object_worker,
        2,
        extra_env={"USE_DEVICE_COMMUNICATOR": str(use_device_communicator)},
    )


@worker_fn_wrapper
def send_recv_object_worker():
    use_dc = _get_use_device_communicator()
    coordinator = _create_coordinator(use_dc, "test_send_recv_object")
    obj = {"data": [1, 2, 3]}
    if coordinator.rank_in_group == 0:
        coordinator.send_object(obj, dst=1)
    elif coordinator.rank_in_group == 1:
        result = coordinator.recv_object(src=0)
        assert result == obj, (
            f"recv_object failed: expected {obj}, got {result}"
        )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run the test.",
)
@pytest.mark.parametrize("use_device_communicator", [True, False])
def test_send_recv_object(use_device_communicator):
    """Test send_object/recv_object through split group."""
    distributed_run(
        send_recv_object_worker,
        2,
        extra_env={"USE_DEVICE_COMMUNICATOR": str(use_device_communicator)},
    )


@worker_fn_wrapper
def barrier_worker():
    use_dc = _get_use_device_communicator()
    coordinator = _create_coordinator(use_dc, "test_barrier")
    coordinator.barrier()


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run the test.",
)
@pytest.mark.parametrize("use_device_communicator", [True, False])
def test_barrier(use_device_communicator):
    """Test barrier through split group."""
    distributed_run(
        barrier_worker,
        2,
        extra_env={"USE_DEVICE_COMMUNICATOR": str(use_device_communicator)},
    )


# ---------------------------------------------------------------------------
# Device Group Tests (Tests 6-10)
# ---------------------------------------------------------------------------
@worker_fn_wrapper
def broadcast_gpu_tensor_worker():
    use_dc = _get_use_device_communicator()
    coordinator = _create_coordinator(use_dc, "test_broadcast_gpu_tensor")
    local_rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    if coordinator.rank_in_group == 0:
        tensor = torch.ones(16, dtype=torch.float32, device=device)
    else:
        tensor = torch.zeros(16, dtype=torch.float32, device=device)
    coordinator.broadcast(tensor, src=0)
    expected = torch.ones(16, dtype=torch.float32, device=device)
    assert torch.equal(tensor, expected), (
        f"broadcast GPU tensor failed: expected all 1s, "
        f"got {tensor.flatten()[:4].tolist()}"
    )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run the test.",
)
@pytest.mark.parametrize("use_device_communicator", [True, False])
def test_broadcast_gpu_tensor(use_device_communicator):
    """Test broadcast of GPU tensor through split group."""
    distributed_run(
        broadcast_gpu_tensor_worker,
        2,
        extra_env={"USE_DEVICE_COMMUNICATOR": str(use_device_communicator)},
    )


@worker_fn_wrapper
def all_reduce_worker():
    use_dc = _get_use_device_communicator()
    coordinator = _create_coordinator(use_dc, "test_all_reduce")
    local_rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    tensor = torch.ones(16, dtype=torch.float32, device=device)
    if use_dc:
        result = coordinator.all_reduce(tensor)
    else:
        torch.distributed.all_reduce(tensor, group=coordinator.device_group)
        result = tensor
    torch.accelerator.synchronize()
    expected = float(coordinator.world_size)
    assert torch.all(result == expected).cpu().item(), (
        f"all_reduce failed: expected {expected}, "
        f"got {result.flatten()[:4].tolist()}"
    )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run the test.",
)
@pytest.mark.parametrize("use_device_communicator", [True, False])
def test_all_reduce(use_device_communicator):
    """Test all_reduce through split group."""
    distributed_run(
        all_reduce_worker,
        2,
        extra_env={"USE_DEVICE_COMMUNICATOR": str(use_device_communicator)},
    )


@worker_fn_wrapper
def all_gather_worker():
    use_dc = _get_use_device_communicator()
    coordinator = _create_coordinator(use_dc, "test_all_gather")
    local_rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    rank_in_group = coordinator.rank_in_group
    tensor = torch.full((4,), float(rank_in_group), dtype=torch.float32, device=device)
    if use_dc:
        result = coordinator.all_gather(tensor, dim=0)
    else:
        world_size = coordinator.world_size
        output = torch.empty(
            4 * world_size, dtype=torch.float32, device=device
        )
        torch.distributed.all_gather_into_tensor(
            output, tensor, group=coordinator.device_group
        )
        result = output
    torch.accelerator.synchronize()
    expected = torch.cat(
        [
            torch.full((4,), float(r), dtype=torch.float32, device=device)
            for r in range(coordinator.world_size)
        ]
    )
    assert torch.equal(result, expected), (
        f"all_gather failed: expected {expected.tolist()}, "
        f"got {result.tolist()}"
    )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run the test.",
)
@pytest.mark.parametrize("use_device_communicator", [True, False])
def test_all_gather(use_device_communicator):
    """Test all_gather through split group."""
    distributed_run(
        all_gather_worker,
        2,
        extra_env={"USE_DEVICE_COMMUNICATOR": str(use_device_communicator)},
    )


@worker_fn_wrapper
def reduce_scatter_worker():
    use_dc = _get_use_device_communicator()
    coordinator = _create_coordinator(use_dc, "test_reduce_scatter")
    local_rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    world_size = coordinator.world_size
    chunk_size = 4
    tensor = torch.ones(
        world_size * chunk_size, dtype=torch.float32, device=device
    )
    if use_dc:
        result = coordinator.reduce_scatter(tensor, dim=0)
    else:
        output = torch.empty(chunk_size, dtype=torch.float32, device=device)
        torch.distributed.reduce_scatter_tensor(
            output, tensor, group=coordinator.device_group
        )
        result = output
    torch.accelerator.synchronize()
    expected_val = float(world_size)
    assert torch.all(result == expected_val).cpu().item(), (
        f"reduce_scatter failed: expected {expected_val}, "
        f"got {result.tolist()}"
    )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run the test.",
)
@pytest.mark.parametrize("use_device_communicator", [True, False])
def test_reduce_scatter(use_device_communicator):
    """Test reduce_scatter through split group."""
    distributed_run(
        reduce_scatter_worker,
        2,
        extra_env={"USE_DEVICE_COMMUNICATOR": str(use_device_communicator)},
    )


@worker_fn_wrapper
def send_recv_gpu_tensor_worker():
    use_dc = _get_use_device_communicator()
    coordinator = _create_coordinator(use_dc, "test_send_recv_gpu_tensor")
    local_rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    if coordinator.rank_in_group == 0:
        tensor = torch.full((16,), 42.0, dtype=torch.float32, device=device)
        if use_dc:
            coordinator.send(tensor, dst=1)
        else:
            dst_global = coordinator.ranks[1]
            torch.distributed.send(
                tensor, dst=dst_global, group=coordinator.device_group
            )
    elif coordinator.rank_in_group == 1:
        if use_dc:
            result = coordinator.recv(
                size=torch.Size([16]), dtype=torch.float32, src=0
            )
        else:
            result = torch.empty(16, dtype=torch.float32, device=device)
            src_global = coordinator.ranks[0]
            torch.distributed.recv(
                result, src=src_global, group=coordinator.device_group
            )
        torch.accelerator.synchronize()
        expected = torch.full(
            (16,), 42.0, dtype=torch.float32, device=device
        )
        assert torch.equal(result, expected), (
            f"send/recv GPU tensor failed: expected {expected.tolist()}, "
            f"got {result.tolist()}"
        )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run the test.",
)
@pytest.mark.parametrize("use_device_communicator", [True, False])
def test_send_recv_gpu_tensor(use_device_communicator):
    """Test send/recv of GPU tensor through split group."""
    distributed_run(
        send_recv_gpu_tensor_worker,
        2,
        extra_env={"USE_DEVICE_COMMUNICATOR": str(use_device_communicator)},
    )


# ---------------------------------------------------------------------------
# Composite Tests (Tests 11-12)
# ---------------------------------------------------------------------------
@worker_fn_wrapper
def broadcast_tensor_dict_worker():
    use_dc = _get_use_device_communicator()
    coordinator = _create_coordinator(use_dc, "test_broadcast_tensor_dict")
    local_rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    if coordinator.rank_in_group == 0:
        tensor_dict = {
            "gpu_tensor": torch.ones(8, dtype=torch.float32, device=device),
            "scalar": 42,
            "text": "hello",
        }
        result = coordinator.broadcast_tensor_dict(tensor_dict, src=0)
    else:
        result = coordinator.broadcast_tensor_dict(src=0)
    assert result is not None
    expected_tensor = torch.ones(8, dtype=torch.float32, device=device)
    assert torch.equal(result["gpu_tensor"], expected_tensor), (
        f"broadcast_tensor_dict GPU tensor mismatch: "
        f"expected {expected_tensor.tolist()}, got {result['gpu_tensor'].tolist()}"
    )
    assert result["scalar"] == 42, (
        f"broadcast_tensor_dict scalar mismatch: expected 42, "
        f"got {result['scalar']}"
    )
    assert result["text"] == "hello", (
        f"broadcast_tensor_dict text mismatch: expected 'hello', "
        f"got {result['text']}"
    )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run the test.",
)
@pytest.mark.parametrize("use_device_communicator", [True, False])
def test_broadcast_tensor_dict(use_device_communicator):
    """Test broadcast_tensor_dict through split group."""
    distributed_run(
        broadcast_tensor_dict_worker,
        2,
        extra_env={"USE_DEVICE_COMMUNICATOR": str(use_device_communicator)},
    )


@worker_fn_wrapper
def send_recv_tensor_dict_worker():
    use_dc = _get_use_device_communicator()
    coordinator = _create_coordinator(use_dc, "test_send_recv_tensor_dict")
    local_rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    if coordinator.rank_in_group == 0:
        tensor_dict = {
            "gpu_tensor": torch.full(
                (8,), 99.0, dtype=torch.float32, device=device
            ),
            "scalar": 42,
        }
        coordinator.send_tensor_dict(tensor_dict, dst=1)
    elif coordinator.rank_in_group == 1:
        result = coordinator.recv_tensor_dict(src=0)
        assert result is not None
        expected_tensor = torch.full(
            (8,), 99.0, dtype=torch.float32, device=device
        )
        assert torch.equal(result["gpu_tensor"], expected_tensor), (
            f"recv_tensor_dict GPU tensor mismatch: "
            f"expected {expected_tensor.tolist()}, "
            f"got {result['gpu_tensor'].tolist()}"
        )
        assert result["scalar"] == 42, (
            f"recv_tensor_dict scalar mismatch: expected 42, "
            f"got {result['scalar']}"
        )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run the test.",
)
@pytest.mark.parametrize("use_device_communicator", [True, False])
def test_send_recv_tensor_dict(use_device_communicator):
    """Test send_tensor_dict/recv_tensor_dict through split group."""
    distributed_run(
        send_recv_tensor_dict_worker,
        2,
        extra_env={"USE_DEVICE_COMMUNICATOR": str(use_device_communicator)},
    )
# ---------------------------------------------------------------------------
# These tests verify that vllm's workarounds for torchcomms shim bugs work
# correctly. Each test exercises the FIXED code path and should PASS.
# ---------------------------------------------------------------------------

try:
    import flashinfer  # noqa: F401

    _flashinfer_available = True
except ImportError:
    _flashinfer_available = False

try:
    import ninja

    _ninja_bin_dir = ninja.BIN_DIR
except (ImportError, AttributeError):
    _ninja_bin_dir = None


def torchcomms_worker_fn_wrapper(fn):
    """Like worker_fn_wrapper but enables torchcomms before distributed init."""

    def wrapped_fn(env):
        update_environment_variables(env)
        if env.get("USE_TORCHCOMMS") == "True":
            import torch.distributed.config as tc_cfg

            tc_cfg.use_torchcomms = True
        local_rank = os.environ["LOCAL_RANK"]
        device = torch.device(f"cuda:{local_rank}")
        torch.accelerator.set_device_index(device)
        init_distributed_environment()
        fn()

    return wrapped_fn


def distributed_run_with_timeout(
    fn, world_size, extra_env=None, timeout=60
):
    """Run distributed fn with timeout. Returns (hung, exitcodes)."""
    processes: list[mp.Process] = []
    for i in range(world_size):
        env: dict[str, str] = {
            "RANK": str(i),
            "LOCAL_RANK": str(i),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12347",
        }
        if extra_env:
            env.update(extra_env)
        p = mp.Process(target=fn, args=(env,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join(timeout=timeout)

    hung = any(p.is_alive() for p in processes)
    exitcodes = []
    for p in processes:
        if p.is_alive():
            p.kill()
            p.join()
            exitcodes.append(None)
        else:
            exitcodes.append(p.exitcode)

    return hung, exitcodes


@torchcomms_worker_fn_wrapper
def batch_isend_irecv_torchcomms_worker():
    rank = torch.distributed.get_rank()
    coordinator = _create_coordinator(False, "test_batch_p2p_tc")
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        send_tensor = torch.ones(16, dtype=torch.float32, device=device)
        ops = [
            torch.distributed.P2POp(
                torch.distributed.isend,
                send_tensor,
                1,
                group=coordinator.device_group,
            )
        ]
    else:
        recv_tensor = torch.zeros(16, dtype=torch.float32, device=device)
        ops = [
            torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_tensor,
                0,
                group=coordinator.device_group,
            )
        ]

    reqs = torch.distributed.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()

    if rank == 1:
        assert torch.all(recv_tensor == 1.0).item(), (
            f"P2P recv failed: expected all 1s, got {recv_tensor.tolist()}"
        )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run the test.",
)
def test_batch_isend_irecv_torchcomms():
    hung, exitcodes = distributed_run_with_timeout(
        batch_isend_irecv_torchcomms_worker,
        2,
        extra_env={"USE_TORCHCOMMS": "True"},
        timeout=60,
    )
    assert not hung, "P2P ops should not hang with batch_isend_irecv"
    # Exit code -11 (SIGSEGV) is acceptable: torchcomms' NCCL garbage
    # collector thread can crash during Python interpreter shutdown.
    assert all(ec in (0, -11) for ec in exitcodes if ec is not None), (
        f"P2P ops failed with exit codes {exitcodes}"
    )


# BUG-3 workaround: Use all_gather_into_tensor instead of all_gather
@torchcomms_worker_fn_wrapper
def all_gather_into_tensor_torchcomms_worker():
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    coordinator = _create_coordinator(False, "test_ag_into_tensor_tc")
    device = torch.device(f"cuda:{rank}")

    tensor = torch.full(
        (16,), float(rank), dtype=torch.float32, device=device
    )
    output = torch.empty(
        world_size * tensor.numel(),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    torch.distributed.all_gather_into_tensor(
        output, tensor, group=coordinator.device_group
    )
    torch.cuda.synchronize()

    expected = torch.cat(
        [
            torch.full((16,), float(r), dtype=torch.float32, device=device)
            for r in range(world_size)
        ]
    )
    assert torch.equal(output, expected), (
        f"all_gather_into_tensor failed: expected {expected.tolist()}, "
        f"got {output.tolist()}"
    )


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run the test.",
)
def test_all_gather_into_tensor_torchcomms():
    """BUG-3 workaround: all_gather_into_tensor (single output tensor) works
    with torchcomms, unlike all_gather (list of per-rank tensors) which fails
    because _BackendWrapper.allgather expects concatenated output."""
    hung, exitcodes = distributed_run_with_timeout(
        all_gather_into_tensor_torchcomms_worker,
        2,
        extra_env={"USE_TORCHCOMMS": "True"},
        timeout=60,
    )
    assert not hung, "all_gather_into_tensor should not hang"
    assert all(ec in (0, -11) for ec in exitcodes if ec is not None), (
        f"all_gather_into_tensor failed with exit codes {exitcodes}"
    )


# BUG-4 workaround: Skip destroy_process_group when torchcomms is active
@torchcomms_worker_fn_wrapper
def destroy_env_torchcomms_worker():
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
    )

    coordinator = _create_coordinator(False, "test_destroy_env_tc")
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank}")

    tensor = torch.ones(16, dtype=torch.float32, device=device)
    torch.distributed.all_reduce(tensor, group=coordinator.device_group)
    torch.cuda.synchronize()

    destroy_distributed_environment()


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run the test.",
)
def test_destroy_env_torchcomms():
    """BUG-4 workaround: destroy_distributed_environment skips
    destroy_process_group when torchcomms is active, avoiding the
    ncclCommDestroy deadlock."""
    hung, exitcodes = distributed_run_with_timeout(
        destroy_env_torchcomms_worker,
        2,
        extra_env={"USE_TORCHCOMMS": "True"},
        timeout=60,
    )
    assert not hung, "destroy_distributed_environment should not hang"
    assert all(ec in (0, -11) for ec in exitcodes if ec is not None), (
        f"destroy_distributed_environment failed with exit codes {exitcodes}"
    )


# BUG-1 workaround: Use mnnvl backend instead of trtllm with torchcomms
@torchcomms_worker_fn_wrapper
def flashinfer_backend_resolve_torchcomms_worker():
    from vllm.distributed.device_communicators.flashinfer_all_reduce import (
        _resolve_fi_ar_backend,
    )

    backend = _resolve_fi_ar_backend()
    assert backend == "mnnvl", (
        f"Expected mnnvl backend with torchcomms, got {backend}"
    )


@pytest.mark.skipif(
    not _flashinfer_available,
    reason="flashinfer is not installed.",
)
@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run the test.",
)
def test_flashinfer_backend_resolve_torchcomms():
    """BUG-1 workaround: _resolve_fi_ar_backend selects mnnvl instead of
    trtllm when torchcomms is active, avoiding the trtllm TCP socket deadlock."""
    hung, exitcodes = distributed_run_with_timeout(
        flashinfer_backend_resolve_torchcomms_worker,
        2,
        extra_env={"USE_TORCHCOMMS": "True"},
        timeout=60,
    )
    assert not hung, "Backend resolution should not hang"
    assert all(ec == 0 for ec in exitcodes if ec is not None), (
        f"Backend resolution failed with exit codes {exitcodes}"
    )
