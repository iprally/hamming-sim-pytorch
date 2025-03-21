import argparse
import time

import torch

from hamming_sim import hamming_sim


def time_hamming_sim(
    num_candidate_docs: int,
    num_tests: int,
    num_query_nodes: int,
    num_candidate_nodes: int,
) -> float:
    n_dim = 256
    query_tensor = torch.randint(0, 256, (num_query_nodes, n_dim), dtype=torch.uint8)
    candidate_tensor_list = [
        torch.randint(0, 256, (num_candidate_nodes, n_dim), dtype=torch.uint8)
        for _ in range(num_candidate_docs)
    ]
    start_time = time.time()
    for _ in range(num_tests):
        candidate_tensors = torch.cat(candidate_tensor_list, dim=0)
        _ = hamming_sim.quantized_1bit_tensor_similarity(
            query_tensor, candidate_tensors
        )
    end_time = time.time()

    return (end_time - start_time) / num_tests


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_candidate_docs", type=int, default=100)
    parser.add_argument("--num_tests", type=int, default=10)
    parser.add_argument("--num_candidate_nodes", type=int, default=650)
    parser.add_argument(
        "--num_query_nodes", type=int, nargs="+", default=[22, 150, 650]
    )
    args = parser.parse_args()

    for num_query_nodes in args.num_query_nodes:
        time_taken = time_hamming_sim(
            args.num_candidate_docs,
            args.num_tests,
            num_query_nodes,
            args.num_candidate_nodes,
        )
        print(
            f"Time taken: {time_taken:.4f} seconds for {num_query_nodes} query nodes."
        )
