from itertools import product
from copy import deepcopy
from datetime import datetime
import argparse

from miyajiro_base import execute

def main():
    timestamp = datetime.now().strftime("%Y%m%d%H%M")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_save_path",
        type=str,
        default="outputs",
    )
    parser.add_argument("--llm", type=str, default="alpaca_7B")

    args = parser.parse_args()

    # Experiment settings
    process_title = "agglomerative"
    llm = args.llm
    dataset = "advglue++"
    info_type = "hidden_states"
    extract_block_idx = "31"
    abstraction_methods = ["Agglomerative"]
    abstract_state_nums = [100, 200, 400, 600]
    pca_dims = [512, 1024, 2048]
    probability_models = ["DTMC"]
    hmm_n_comps = [100, 200, 400]
    state_abstract_args = {
        "llm_name": llm,
        "result_save_path": args.result_save_path,
        "dataset": dataset,
        "test_ratio": 0.2,
        "extract_block_idx": extract_block_idx,
        "info_type": info_type,
        "is_attack_success": 1,
    }

    execute(
        process_title=process_title,
        timestamp=timestamp,
        llm=llm,
        dataset=dataset,
        info_type=info_type,
        extract_block_idx=extract_block_idx,
        abstraction_methods=abstraction_methods,
        abstract_state_nums=abstract_state_nums,
        pca_dims=pca_dims,
        probability_models=probability_models,
        hmm_n_comps=hmm_n_comps,
        state_abstract_args=state_abstract_args,
    )


if __name__ == "__main__":
    main()
