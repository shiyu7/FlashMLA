"""Measure the production SGLang MLA DCP merge for BF16/FP32 local states."""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist

from sglang.srt.layers.dcp.comm import cp_lse_ag_out_rs_mla


class TorchGroup:
    def __init__(self):
        self.world_size = dist.get_world_size()
        self.rank_in_group = dist.get_rank()
        self.device_group = dist.group.WORLD

    def all_gather(self, tensor, dim=0):
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor, group=self.device_group)
        return torch.cat(gathered, dim=dim)

    def reduce_scatter_along_dim(self, tensor, dim=0):
        moved = tensor.movedim(dim, 0).contiguous()
        output = torch.empty(
            (moved.shape[0] // self.world_size,) + moved.shape[1:],
            dtype=moved.dtype,
            device=moved.device,
        )
        dist.reduce_scatter_tensor(output, moved, group=self.device_group)
        return output.movedim(0, dim)


def measure(rows, dtype, group):
    heads = 128
    head_dim = 512
    local_out = torch.randn(rows, heads, head_dim, device="cuda", dtype=dtype)
    local_lse = torch.randn(rows, heads, device="cuda", dtype=torch.float32)
    def function():
        output = cp_lse_ag_out_rs_mla(
            local_out,
            local_lse,
            group,
            is_lse_base_on_e=False,
        )
        return output.to(torch.bfloat16)

    for _ in range(10):
        function()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()
    torch.cuda.synchronize()
    dist.barrier()
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(100):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) / 10)
    all_samples = [None] * group.world_size
    dist.all_gather_object(all_samples, samples)
    if group.rank_in_group == 0:
        critical = [max(values) for values in zip(*all_samples)]
        return {
            "rows": rows,
            "dtype": str(dtype).removeprefix("torch."),
            "p50_ms": statistics.median(critical),
            "p99_ms": sorted(critical)[98],
        }
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    group = TorchGroup()
    results = []
    for rows in (8, 64):
        for dtype in (torch.bfloat16, torch.float32):
            result = measure(rows, dtype, group)
            if result is not None:
                results.append(result)
                print(result, flush=True)
    if group.rank_in_group == 0 and args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            json.dump(
                {
                    "kind": "sglang_dcp_merge_dtype_ab",
                    "world_size": group.world_size,
                    "gpu": torch.cuda.get_device_name(),
                    "measurement": {
                        "cuda_graph": True,
                        "warmup": 100,
                        "samples": 100,
                        "calls_per_sample": 10,
                        "includes_final_bfloat16_cast": True,
                    },
                    "results": results,
                },
                output,
                indent=2,
                sort_keys=True,
            )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
