from luna.metrics_appeval_collection import MetricsAppEvalCollections
import luna.data_loader as data_loader

from types import SimpleNamespace
import pandas as pd
import os
from copy import deepcopy
from datetime import datetime
import numpy as np
from scipy.stats import mannwhitneyu
import argparse
from time import time

CLUSTER_METHODS = ["GMM", "KMeans", "Birch"]
STATE_DEFINABLE_METHODS = ["GMM", "KMeans", "Birch", "Grid"]
EPSILON_DEFINABLE_METHODS = ["DBSCAN"]

def write_result_to_csv(
    timestamp, result, settings_str, dataset, extract_block_idx, info_type, llm, execution_time,
    process_title='rq23_all_settings',
    abstraction_method=None,
    abstract_state_num=None,
    partition_num=None,
    pca_dim=None,
    epsilon=None,
    model_type=None,
    hmm_n_comp=None,
    grid_history_dependency_num=None,
    p_value=None,
):
    csv_folder = "eval/{}/{}/{}/{}".format(
        dataset, extract_block_idx, info_type, llm
    )
    path = "{}/{}_{}.csv".format(csv_folder, timestamp, process_title)
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)

    result.update({
        "dataset": dataset,
        "abstraction_method": abstraction_method,
        "abstract_state_num": abstract_state_num,
        "epsilon": epsilon,
        "partition_num": partition_num,
        "pca_dim": pca_dim,
        "model_type": model_type,
        "hmm_n_comp": hmm_n_comp,
        "grid_history_dependency_num": grid_history_dependency_num,
        "settings": settings_str,
        "execution_time": execution_time,
        "p_value": p_value,
    })

    columns_order = [
        "dataset",
        "abstraction_method",
        "partition_num",
        "abstract_state_num",
        "epsilon",
        "pca_dim",
        "model_type",
        "hmm_n_comp",
        "grid_history_dependency_num",
        "execution_time",
        "aucroc",
        "p_value",
        "accuracy",
        "f1_score",
        "abnormal_threshold",
        "settings",
        "preciseness_mean",
        "preciseness_max",
        "entropy_val",
        "entropy_test",
        "probabilistic_reasoning_divergence",
        "value_diversity_instant_level_val",
        "value_diversity_instant_level_test",
        "value_diversity_n_gram_level_val",
        "value_diversity_n_gram_level_test",
        "derivative_diversity_n_gram_level_val_increasing",
        "derivative_diversity_n_gram_level_val_decreasing",
        "derivative_diversity_n_gram_level_test_increasing",
        "derivative_diversity_n_gram_level_test_decreasing",
        "second_derivative_diversity_n_gram_level_val_increasing",
        "second_derivative_diversity_n_gram_level_val_decreasing",
        "second_derivative_diversity_n_gram_level_test_increasing",
        "second_derivative_diversity_n_gram_level_test_decreasing",
        "succinctness",
        "coverage",
        "sensitivity",
        "sink_state",
        "source_state",
        "recurrent_state",
        "abstract_model_perplexity_good",
        "abstract_model_perplexity_bad",
        "abstract_model_smoothed_perplexity_2_good",
        "abstract_model_smoothed_perplexity_2_bad",
        "abstract_model_smoothed_perplexity_3_good",
        "abstract_model_smoothed_perplexity_3_bad",
        "abstract_model_smoothed_perplexity_4_good",
        "abstract_model_smoothed_perplexity_4_bad",
    ]

    dict_result = {
        "abstract_model_perplexity_good": result["perplexity_abstract_model"][0],
        "abstract_model_perplexity_bad": result["perplexity_abstract_model"][1],
        "abstract_model_smoothed_perplexity_2_good": result[
            "smoothed_perplexity_abstract_model"
        ][2][0],
        "abstract_model_smoothed_perplexity_2_bad": result[
            "smoothed_perplexity_abstract_model"
        ][2][1],
        "abstract_model_smoothed_perplexity_3_good": result[
            "smoothed_perplexity_abstract_model"
        ][3][0],
        "abstract_model_smoothed_perplexity_3_bad": result[
            "smoothed_perplexity_abstract_model"
        ][3][1],
        "abstract_model_smoothed_perplexity_4_good": result[
            "smoothed_perplexity_abstract_model"
        ][4][0],
        "abstract_model_smoothed_perplexity_4_bad": result[
            "smoothed_perplexity_abstract_model"
        ][4][1],
        "settings": result["settings"],
    }
    for key, value in dict_result.items():
        result[key] = value
    del result["smoothed_perplexity_abstract_model"]
    del result["perplexity_abstract_model"]

    # for key, value in result["stationary_distribution_entropy_dict"].items():
    #     result[key] = value
    # del result["stationary_distribution_entropy_dict"]

    df = pd.DataFrame([result])[columns_order]

    if not os.path.isfile(path):
        print("Creating new file")
        df.to_csv(
            path, mode="w", index=False, header=True
        )
    else:
        print("Appending to existing file")
        df.to_csv(
            path, mode="a", index=False, header=False
        )


def rq3(state_abstract_args, prob_args, train_instances, val_instances, test_instances):
    state_abstract_args_obj = SimpleNamespace(**state_abstract_args)
    prob_args_obj = SimpleNamespace(**prob_args)

    eval_obj = MetricsAppEvalCollections(
        state_abstract_args_obj,
        prob_args_obj,
        train_instances,
        val_instances,
        test_instances,
    )

    (
        aucroc,
        accuracy,
        f1_score,
        _,
        _,
        abnormal_threshold,
    ) = eval_obj.get_eval_result()
    eval_result_dict = {
        "aucroc": aucroc,
        "accuracy": accuracy,
        "f1_score": f1_score,
        "abnormal_threshold": abnormal_threshold,
    }

    preciseness = eval_obj.preciseness()
    entropy = eval_obj.entropy()
    probabilistic_reasoning = eval_obj.probabilistic_reasoning()
    value_diversity_instant_level = eval_obj.value_diversity_instant_level()
    value_diversity_n_gram_level = eval_obj.value_diversity_n_gram_level()
    derivative_diversity_n_gram_level = eval_obj.derivative_diversity_n_gram_level()
    second_derivative_diversity_n_gram_level = (
        eval_obj.second_derivative_diversity_n_gram_level()
    )

    eval_result_dict["transition_matrix_list"] = eval_obj.transition_matrix_list()

    eval_result_dict["preciseness_mean"] = preciseness[0]
    eval_result_dict["preciseness_max"] = preciseness[1]

    eval_result_dict["entropy_val"] = entropy[0]
    eval_result_dict["entropy_test"] = entropy[1]

    eval_result_dict["probabilistic_reasoning_divergence"] = probabilistic_reasoning

    eval_result_dict[
        "value_diversity_instant_level_val"
    ] = value_diversity_instant_level[0]
    eval_result_dict[
        "value_diversity_instant_level_test"
    ] = value_diversity_instant_level[1]

    eval_result_dict["value_diversity_n_gram_level_val"] = value_diversity_n_gram_level[
        0
    ]
    eval_result_dict[
        "value_diversity_n_gram_level_test"
    ] = value_diversity_n_gram_level[1]

    eval_result_dict[
        "derivative_diversity_n_gram_level_val_increasing"
    ] = derivative_diversity_n_gram_level[0]
    eval_result_dict[
        "derivative_diversity_n_gram_level_val_decreasing"
    ] = derivative_diversity_n_gram_level[1]
    eval_result_dict[
        "derivative_diversity_n_gram_level_test_increasing"
    ] = derivative_diversity_n_gram_level[2]
    eval_result_dict[
        "derivative_diversity_n_gram_level_test_decreasing"
    ] = derivative_diversity_n_gram_level[3]

    eval_result_dict[
        "second_derivative_diversity_n_gram_level_val_increasing"
    ] = second_derivative_diversity_n_gram_level[0]
    eval_result_dict[
        "second_derivative_diversity_n_gram_level_val_decreasing"
    ] = second_derivative_diversity_n_gram_level[1]
    eval_result_dict[
        "second_derivative_diversity_n_gram_level_test_increasing"
    ] = second_derivative_diversity_n_gram_level[2]
    eval_result_dict[
        "second_derivative_diversity_n_gram_level_test_decreasing"
    ] = second_derivative_diversity_n_gram_level[3]

    eval_result_dict["succinctness"] = eval_obj.succinctness()
    eval_result_dict["coverage"] = eval_obj.state_coverage()
    eval_result_dict["sensitivity"] = eval_obj.sensitivity()
    eval_result_dict["sink_state"] = eval_obj.sink_state()
    eval_result_dict["source_state"] = eval_obj.source_state()
    eval_result_dict["recurrent_state"] = eval_obj.recurrent_state()
    # eval_result_dict[
    #     "stationary_distribution_entropy_dict"
    # ] = eval_obj.stationary_distribution_entropy()
    eval_result_dict["perplexity_abstract_model"] = eval_obj.perplexity_abstract_model()
    eval_result_dict[
        "smoothed_perplexity_abstract_model"
    ] = eval_obj.smoothed_perplexity_abstract_model()

    eval_result_dict["transition_matrix_list"] = eval_obj.transition_matrix_list()

    return eval_result_dict


def run_experiment(
    train_instances,
    val_instances,
    test_instances,
    abstraction_method,
    abstract_state_num,
    epsilon,
    pca_dim,
    model_type,
    llm,
    dataset,
    info_type,
    extract_block_idx,
    hmm_n_comp=None,
    grid_history_dependency_num=None,
):
    state_abstract_args = {
        "llm_name": llm,
        "result_save_path": "outputs",
        "dataset": dataset,
        "test_ratio": 0.2,
        "extract_block_idx": extract_block_idx,
        "info_type": info_type,
        "is_attack_success": 1,
        "cluster_method": abstraction_method,
        "abstract_state": abstract_state_num,
        "epsilon": epsilon,
        "pca_dim": pca_dim,
        "grid_history_dependency_num": grid_history_dependency_num
        if grid_history_dependency_num
        else "",
    }

    prob_args = {
        "dataset": dataset,
        "test_ratio": 0.2,
        "extract_block_idx": extract_block_idx,
        "info_type": info_type,
        "is_attack_success": 1,
        "iter_num": 100,
        "cluster_method": abstraction_method,
        "abstract_state": abstract_state_num,
        "epsilon": epsilon,
        "pca_dim": pca_dim,
        "model_type": model_type,
        "hmm_components_num": hmm_n_comp if hmm_n_comp else "",
        "grid_history_dependency_num": grid_history_dependency_num
        if grid_history_dependency_num
        else "",
    }

    settings_str = "{}_{}_{}_{}_{}_{}".format(
        abstraction_method,
        abstract_state_num,
        pca_dim,
        model_type,
        hmm_n_comp if hmm_n_comp else "0",
        grid_history_dependency_num if grid_history_dependency_num else "0",
    )

    # Call the rq3 function and return the result
    result = rq3(
        state_abstract_args, prob_args, train_instances, val_instances, test_instances
    )
    train_probs, test_probs, val_probs = result["transition_matrix_list"]
    test_probs = np.array(test_probs)
    val_probs = np.array(val_probs)
    _, p_value = mannwhitneyu(test_probs, val_probs, alternative="two-sided")

    return result, settings_str, p_value


def load_data(state_abstract_args):
    args = SimpleNamespace(**state_abstract_args)
    llm_name = args.llm_name
    result_save_path = args.result_save_path
    dataset = args.dataset
    info_type = args.info_type
    extract_block_idx_str = args.extract_block_idx
    is_attack_success = args.is_attack_success

    dataset_folder_path = "{}/{}/{}".format(
        result_save_path, dataset, extract_block_idx_str
    )
    if not os.path.exists(dataset_folder_path):
        os.makedirs(dataset_folder_path)

    eval_folder_path = "eval/{}/{}".format(dataset, extract_block_idx_str)
    if not os.path.exists(eval_folder_path):
        os.makedirs(eval_folder_path)

    loader = None
    if dataset == "truthful_qa":
        loader = data_loader.TqaDataLoader(dataset_folder_path, llm_name)

    elif dataset == "advglue++":
        loader = data_loader.AdvDataLoader(
            dataset_folder_path, llm_name, is_attack_success
        )

    elif dataset == "sst2":
        loader = data_loader.OodDataLoader(dataset_folder_path, llm_name)

    elif dataset == "humaneval" or dataset == "mbpp":
        loader = data_loader.CodeLoader(dataset_folder_path, llm_name)

    else:
        raise NotImplementedError("Unknown dataset!")

    if info_type == "hidden_states":
        print("Loading hidden states...")
        (
            train_instances,
            val_instances,
            test_instances,
        ) = loader.load_hidden_states()
        print("Finished loading hidden states!")

    elif info_type == "attention_heads" or info_type == "attention_blocks":
        if info_type == "attention_heads":
            print("Loading attention heads...")
            (
                train_instances,
                val_instances,
                test_instances,
            ) = loader.load_attentions(0)
            print("Finished loading attention heads!")
        else:
            print("Loading attention blocks...")
            (
                train_instances,
                val_instances,
                test_instances,
            ) = loader.load_attentions(1)
            print("Finished loading attention blocks!")
    else:
        raise NotImplementedError("Unknown info type!")
    return train_instances, val_instances, test_instances


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
    abstraction_methods = ["Grid", "GMM", "KMeans"]
    partition_nums = [5, 10, 15]
    abstract_state_nums = [200, 400, 600]
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
    # Iterate through each abstraction method (Grid and Cluster-based)
    for abstraction_method in abstraction_methods:
        # If Grid abstraction method is chosen
        if abstraction_method == "Grid":
            # Iterate through possible partition numbers for the Grid method
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
                                    start_time = time()
                                    result, settings_str = run_experiment(
                                        train_instances=train_instances,
                                        val_instances=val_instances,
                                        test_instances=test_instances,
                                        abstraction_method=abstraction_method,
                                        abstract_state_num=partition_num,
                                        epsilon=0,
                                        pca_dim=pca_dim,
                                        model_type=model_type,
                                        llm=llm,
                                        dataset=dataset,
                                        info_type=info_type,
                                        extract_block_idx=extract_block_idx,
                                        hmm_n_comp=hmm_n_comp,
                                        grid_history_dependency_num=grid_history_dependency_num,
                                    )
                                    execution_time = time() - start_time
                                    print("result", result)
                                    if result:
                                        write_result_to_csv(
                                            timestamp=timestamp,
                                            result=result,
                                            execution_time=execution_time,
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
                                start_time = time()
                                result, settings_str = run_experiment(
                                    train_instances=train_instances,
                                    val_instances=val_instances,
                                    test_instances=test_instances,
                                    abstraction_method=abstraction_method,
                                    abstract_state_num=partition_num,
                                    epsilon=0,
                                    pca_dim=pca_dim,
                                    model_type=model_type,
                                    llm=llm,
                                    dataset=dataset,
                                    info_type=info_type,
                                    extract_block_idx=extract_block_idx,
                                    grid_history_dependency_num=grid_history_dependency_num,
                                )
                                execution_time = time() - start_time
                                if result:
                                    write_result_to_csv(
                                        timestamp=timestamp,
                                        result=result,
                                        execution_time=execution_time,
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
        elif abstraction_method in CLUSTER_METHODS:
            # (similar logic as above for cluster-based experiments)
            for abstract_state_num in abstract_state_nums:
                for pca_dim in pca_dims:
                    for model_type in probability_models:
                        if model_type == "HMM":
                            for hmm_n_comp in hmm_n_comps:
                                train_instances = deepcopy(train_instances_loaded)
                                val_instances = deepcopy(val_instances_loaded)
                                test_instances = deepcopy(test_instances_loaded)
                                start_time = time()
                                result, settings_str = run_experiment(
                                    train_instances=train_instances,
                                    val_instances=val_instances,
                                    test_instances=test_instances,
                                    abstraction_method=abstraction_method,
                                    abstract_state_num=abstract_state_num,
                                    epsilon=0,
                                    pca_dim=pca_dim,
                                    model_type=model_type,
                                    llm=llm,
                                    dataset=dataset,
                                    info_type=info_type,
                                    extract_block_idx=extract_block_idx,
                                    hmm_n_comp=hmm_n_comp,
                                )
                                execution_time = time() - start_time
                                if result:
                                    write_result_to_csv(
                                        timestamp=timestamp,
                                        result=result,
                                        execution_time=execution_time,
                                        settings_str=settings_str,
                                        dataset=dataset,
                                        extract_block_idx=extract_block_idx,
                                        info_type=info_type,
                                        llm=llm,
                                        abstraction_method=abstraction_method,
                                        abstract_state_num=abstract_state_num,
                                        pca_dim=pca_dim,
                                        model_type=model_type,
                                    )
                        else:
                            train_instances = deepcopy(train_instances_loaded)
                            val_instances = deepcopy(val_instances_loaded)
                            test_instances = deepcopy(test_instances_loaded)
                            start_time = time()
                            result, settings_str = run_experiment(
                                train_instances=train_instances,
                                val_instances=val_instances,
                                test_instances=test_instances,
                                abstraction_method=abstraction_method,
                                abstract_state_num=abstract_state_num,
                                epsilon=0,
                                pca_dim=pca_dim,
                                model_type=model_type,
                                llm=llm,
                                dataset=dataset,
                                info_type=info_type,
                                extract_block_idx=extract_block_idx,
                            )

                            execution_time = time() - start_time
                            if result:
                                write_result_to_csv(
                                    timestamp=timestamp,
                                    result=result,
                                    execution_time=execution_time,
                                    settings_str=settings_str,
                                    dataset=dataset,
                                    extract_block_idx=extract_block_idx,
                                    info_type=info_type,
                                    llm=llm,
                                    abstraction_method=abstraction_method,
                                    abstract_state_num=abstract_state_num,
                                    pca_dim=pca_dim,
                                    model_type=model_type,
                                )

if __name__ == "__main__":
    main()
