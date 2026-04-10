import asyncio
import random
import time

import numpy as np
import uvloop

from tpu_mini_sglang.entrypoints.engine import launch_subprocesses
from tpu_mini_sglang.managers.io_struct import GenerateRequest
from tpu_mini_sglang.model_config import ModelConfig
from tpu_mini_sglang.sampling.sampling_params import SamplingParams
from tpu_mini_sglang.server_args import ServerArgs
from tpu_mini_sglang.utils import kill_process_tree


def get_generate_req_fn(vocab_size: int):
    def generate_random_test_request(
        input_length: int,
        output_length: int,
        temperature: float = 0.0,
    ) -> GenerateRequest:

        sampling_params = SamplingParams(
            temperature=temperature,
            max_new_tokens=output_length,
            ignore_eos=True,
        )
        input_ids = [random.randint(0, vocab_size - 1) for _ in range(input_length)]
        return GenerateRequest(
            input_ids=input_ids,
            sampling_params=sampling_params,
        )

    return generate_random_test_request


async def main():
    parser = ServerArgs.build_parser(dest="server_args")
    parser.add_argument("--num-iters", type=int, default=3)
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=1)
    args = parser.parse_args()

    num_iters = args.num_iters
    num_seqs = args.num_seqs
    server_args = args.server_args
    model_config = ModelConfig(model_path=server_args.model_path)

    random.seed(0)
    generate_request = get_generate_req_fn(model_config.vocab_size)
    tokenizer_manager = launch_subprocesses(server_args)

    for _ in range(args.warmup_iters):
        # Do one prefill and one decode for each warmup iteration
        await tokenizer_manager.generate_request(
            generate_request(input_length=random.randint(50, 100), output_length=2)
        )

    times = []
    input_tokens = []
    output_tokens = []
    for _ in range(num_iters):
        lengths = [(random.randint(50, 100), random.randint(50, 100)) for _ in range(num_seqs)]
        reqs = [
            generate_request(
                input_length=input_length,
                output_length=output_length,
            )
            for input_length, output_length in lengths
        ]
        tasks = []
        for req in reqs:
            tasks.append(asyncio.create_task(tokenizer_manager.generate_request(req)))
        start_time = time.perf_counter()  # Tasks start running the moment we hit await
        await asyncio.gather(*tasks)
        stop_time = time.perf_counter()
        times.append(stop_time - start_time)
        input_tokens.append(sum(input_length for input_length, _ in lengths))
        output_tokens.append(sum(output_length for _, output_length in lengths))
    print(f"Times for each run: {times}")
    print(f"Num input tokens for each run: {input_tokens}")
    print(f"Num output tokens for each run: {output_tokens}")
    tps = [toks / t for toks, t in zip(output_tokens, times, strict=True)]
    print(f"Toks/sec for each run: {tps}")
    print(f"\n\nOverall Summary:\nMean tps: {np.mean(tps)}\nMedian tps: {np.median(tps)}\n")


if __name__ == "__main__":
    try:
        uvloop.run(main())
    finally:
        # Only kill children to allow exception tracebacks to print
        kill_process_tree(include_parent=False)
