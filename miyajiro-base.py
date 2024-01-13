from itertools import product
from copy import deepcopy
from datetime import datetime
import argparse

from RQ23_all_settings import run_experiment, write_result_to_csv, load_data

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
    llm = args.llm
    dataset = "advglue++"
    info_type = "hidden_states"
    extract_block_idx = "31"
    abstraction_methods = ["Grid-based", "Cluster-based"]
    partition_nums = [5, 10, 15]
    abstraction_states = [200, 400, 600]
    cluster_methods = ["GMM", "KMeans"]
    # cluster_methods = ["Birch"]
    pca_dims = [512, 1024, 2048]
    grid_pca_dims = [3, 5, 10]
    probability_models = ["DTMC"]
    hmm_n_comps = [100, 200, 400]
    grid_history_dependency = [1, 2, 3]
    grid_hmm_n_comps = [100, 200, 400]
    state_abstract_args = {
        "llm_name": llm,
        "result_save_path": args.result_save_path,
        "dataset": dataset,
        "test_ratio": 0.2,
        "extract_block_idx": extract_block_idx,
        "info_type": info_type,
        "is_attack_success": 1,
    }

    train_instances_loaded, val_instances_loaded, test_instances_loaded = load_data(
        state_abstract_args
    )
    # Iterate through each abstraction method (Grid-based and Cluster-based)
    for abstraction_method in abstraction_methods:
        # If Grid-based abstraction method is chosen
        if abstraction_method == "Grid-based":
            # Iterate through possible partition numbers for the grid-based method
            for partition_num in partition_nums: # 5, 10, 15
                # Explore the impact of different PCA dimensions
                for pca_dim in grid_pca_dims:  # 3, 5, 10
                    # Explore results for different probability models (DTMC and HMM)
                    for model_type in probability_models: # DTMC, HMM
                        # Explore different numbers of Grid history dependency
                        for grid_history_dependency_num in grid_history_dependency: # 1, 2, 3
                            # If the model is Hidden Markov Model (HMM)
                            if model_type == "HMM":
                                # Iterate over different numbers of HMM components
                                for hmm_n_comp in grid_hmm_n_comps: # 100, 200, 400
                                    train_instances = deepcopy(train_instances_loaded)
                                    val_instances = deepcopy(val_instances_loaded)
                                    test_instances = deepcopy(test_instances_loaded)
                                    result, settings_str = run_experiment(
                                        train_instances,
                                        val_instances,
                                        test_instances,
                                        "Grid",
                                        partition_num,
                                        pca_dim,
                                        model_type,
                                        llm,
                                        dataset,
                                        info_type,
                                        extract_block_idx,
                                        hmm_n_comp=hmm_n_comp,
                                        grid_history_dependency_num=grid_history_dependency_num,
                                    )
                                    print("result", result)
                                    if result:
                                        write_result_to_csv(
                                            timestamp=timestamp,
                                            result=result,
                                            settings_str=settings_str,
                                            dataset=dataset,
                                            extract_block_idx=extract_block_idx,
                                            info_type=info_type,
                                            llm=llm,
                                            abstraction_method=abstraction_method,
                                            partition_num=partition_num,
                                            pca_dim=pca_dim,
                                            model_type=model_type,
                                            hmm_n_comp=hmm_n_comp,
                                            grid_history_dependency_num=grid_history_dependency_num,
                                        )

                            else:
                                train_instances = deepcopy(train_instances_loaded)
                                val_instances = deepcopy(val_instances_loaded)
                                test_instances = deepcopy(test_instances_loaded)
                                result, settings_str = run_experiment(
                                    train_instances,
                                    val_instances,
                                    test_instances,
                                    "Grid",
                                    partition_num,
                                    pca_dim,
                                    model_type,
                                    llm,
                                    dataset,
                                    info_type,
                                    extract_block_idx,
                                    grid_history_dependency_num=grid_history_dependency_num,
                                )
                                if result:
                                    write_result_to_csv(
                                        timestamp=timestamp,
                                        result=result,
                                        settings_str=settings_str,
                                        dataset=dataset,
                                        extract_block_idx=extract_block_idx,
                                        info_type=info_type,
                                        llm=llm,
                                        abstraction_method=abstraction_method,
                                        partition_num=partition_num,
                                        pca_dim=pca_dim,
                                        model_type=model_type,
                                        grid_history_dependency_num=grid_history_dependency_num,
                                    )

        # If Cluster-based abstraction method is chosen
        elif abstraction_method == "Cluster-based":
            # (similar logic as above for cluster-based experiments)
            for abstraction_state, cluster_method in product(
                abstraction_states, cluster_methods
            ):
                for pca_dim in pca_dims:
                    for model_type in probability_models:
                        if model_type == "HMM":
                            for hmm_n_comp in hmm_n_comps:
                                train_instances = deepcopy(train_instances_loaded)
                                val_instances = deepcopy(val_instances_loaded)
                                test_instances = deepcopy(test_instances_loaded)
                                result, settings_str = run_experiment(
                                    train_instances,
                                    val_instances,
                                    test_instances,
                                    cluster_method,
                                    abstraction_state,
                                    pca_dim,
                                    model_type,
                                    llm,
                                    dataset,
                                    info_type,
                                    extract_block_idx,
                                    hmm_n_comp=hmm_n_comp,
                                )
                                if result:
                                    write_result_to_csv(
                                        timestamp=timestamp,
                                        result=result,
                                        settings_str=settings_str,
                                        dataset=dataset,
                                        extract_block_idx=extract_block_idx,
                                        info_type=info_type,
                                        llm=llm,
                                        abstraction_method=abstraction_method,
                                        abstraction_state=abstraction_state,
                                        cluster_method=cluster_method,
                                        pca_dim=pca_dim,
                                        model_type=model_type,
                                    )
                        else:
                            train_instances = deepcopy(train_instances_loaded)
                            val_instances = deepcopy(val_instances_loaded)
                            test_instances = deepcopy(test_instances_loaded)
                            result, settings_str = run_experiment(
                                train_instances,
                                val_instances,
                                test_instances,
                                cluster_method,
                                abstraction_state,
                                pca_dim,
                                model_type,
                                llm,
                                dataset,
                                info_type,
                                extract_block_idx,
                            )
                            if result:
                                write_result_to_csv(
                                    timestamp=timestamp,
                                    result=result,
                                    settings_str=settings_str,
                                    dataset=dataset,
                                    extract_block_idx=extract_block_idx,
                                    info_type=info_type,
                                    llm=llm,
                                    abstraction_method=abstraction_method,
                                    abstraction_state=abstraction_state,
                                    cluster_method=cluster_method,
                                    pca_dim=pca_dim,
                                    model_type=model_type,
                                )

if __name__ == "__main__":
    main()
