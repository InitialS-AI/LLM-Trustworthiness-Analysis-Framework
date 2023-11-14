from luna.state_abstraction_utils import AbstractStateExtraction
from luna.probabilistic_abstraction_model import (
    HmmModel,
    DtmcModel,
)
from luna.abstraction_model import RegularGrid
from luna.utils.interfaces import Grid

from collections import Counter
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.special import kl_div
import torch

class Metrics:
    def __init__(
        self, abs_args, prob_args, train_instances=None, val_instances=None, test_instances=None
    ):
        self.abstractStateExtraction = AbstractStateExtraction(
            abs_args, train_instances, val_instances, test_instances
        )
        self.dataset = prob_args.dataset
        self.dtmc_model = DtmcModel(
            prob_args.dataset,
            prob_args.extract_block_idx,
            prob_args.info_type,
            prob_args.cluster_method,
            prob_args.abstract_state,
            prob_args.pca_dim,
            prob_args.test_ratio,
            prob_args.is_attack_success,
            prob_args.grid_history_dependency_num,
        )
        (
            self.dtmc_transition_aucroc,
            self.dtmc_transition_fpr,
            self.dtmc_transition_tpr,
        ) = self.dtmc_model.get_aucroc_by_transition_binding()

        if prob_args.model_type == "DTMC":
            self.prob_model = self.dtmc_model
            self.train_abstract_traces = self.dtmc_model.train_traces
            self.test_abstract_traces = self.dtmc_model.test_traces
            self.val_abstract_traces = self.dtmc_model.val_traces

        elif prob_args.model_type == "HMM":
            self.hmm_model = HmmModel(
                prob_args.dataset,
                prob_args.extract_block_idx,
                prob_args.info_type,
                prob_args.cluster_method,
                prob_args.abstract_state,
                prob_args.pca_dim,
                prob_args.test_ratio,
                prob_args.hmm_components_num,
                prob_args.iter_num,
                prob_args.is_attack_success,
                prob_args.grid_history_dependency_num,
            )
            (
                self.hmm_transition_aucroc,
                self.hmm_transition_fpr,
                self.hmm_transition_tpr,
            ) = self.hmm_model.get_aucroc_by_transition_binding()

            self.prob_model = self.hmm_model
            self.train_abstract_traces = [
                self.hmm_model.hmm_model.decode([[x] for x in trace])[1].tolist()
                for trace in self.hmm_model.train_traces
            ]
            self.test_abstract_traces = [
                self.hmm_model.hmm_model.decode([[x] for x in trace])[1].tolist()
                for trace in self.hmm_model.test_traces
            ]
            self.val_abstract_traces = [
                self.hmm_model.hmm_model.decode([[x] for x in trace])[1].tolist()
                for trace in self.hmm_model.val_traces
            ]
        else:
            raise NotImplementedError("Unknown model type!")

    def sliding_window(self, sequence, window_size):
        """Generate sub-sequences of window_size from sequence."""

    def count_subsequences(self, traces, window_size):
        """Count the occurrence of each sub-sequence of window_size in traces."""
        counter = Counter()
        for trace in traces:
            for subsequence in self.sliding_window(trace, window_size):
                # Convert list of integers to a hashable type to count them
                subsequence_tuple = tuple(subsequence)
                counter[subsequence_tuple] += 1
        return counter

    def _calculate_radius(self, cluster_method, data, reference_radius=None):
        print(self.abstractStateExtraction.cluster_model)
        if cluster_method == "KMeans":
            cluster_centers = (
                self.abstractStateExtraction.cluster_model.cluster_centers_
            )
        elif cluster_method == "GMM":
            cluster_centers = self.abstractStateExtraction.cluster_model.means_
        else:
            raise ValueError("Unknown cluster method: %s" % cluster_method)
        max_radius = 0
        exceeding_count = 0
        for i, center in enumerate(cluster_centers):
            cur_cluster = i + 1
            if cur_cluster not in data:
                continue
            distances = pairwise_distances(data[cur_cluster], [center])
            max_radius_cluster = distances.max()
            max_radius = max(max_radius, max_radius_cluster)

            if reference_radius is not None:
                exceeding_count += (distances > reference_radius).sum()
        return max_radius, exceeding_count

    # ===================== Model-aware =====================
    """
    The degree of abstract state and transition compression
    param: num_abstract_states, num_tokens, num_instances
    return: [state_compression, transition_compression]
    """

    def succinctness(self):
        num_abs_states = self.abstractStateExtraction.args.abstract_state
        num_concrete_states = sum(
            [
                len(i["state_trace"])
                for i in self.abstractStateExtraction.train_instances
            ]
        )

        result = num_abs_states / num_concrete_states

        return result

    def transition_MMD_instance_level(self):
        def get_transition_distribution(traces):
            transition_matrix = {}
            for trace in traces:                
                for i in range(len(trace) - 1):
                    state = trace[i]
                    next = trace[i + 1]
                    if state not in transition_matrix:
                        transition_matrix[state] = {}
                    else:
                        if next not in transition_matrix[state]:
                            transition_matrix[state][next] = 1
                        else:
                            transition_matrix[state][next] += 1
                for state in transition_matrix:
                    total = sum(transition_matrix[state].values())
                    for next in transition_matrix[state]:
                        transition_matrix[state][next] /= total
            return transition_matrix

        def MMD(x, y, kernel):
            """Emprical maximum mean discrepancy. The lower the result
            the more evidence that distributions are the same.

            Args:
                x: first sample, distribution P
                y: second sample, distribution Q
                kernel: kernel type such as "multiscale" or "rbf"
            """
            xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
            rx = (xx.diag().unsqueeze(0).expand_as(xx))
            ry = (yy.diag().unsqueeze(0).expand_as(yy))

            dxx = rx.t() + rx - 2. * xx # Used for A in (1)
            dyy = ry.t() + ry - 2. * yy # Used for B in (1)
            dxy = rx.t() + ry - 2. * zz # Used for C in (1)

            XX, YY, XY = (torch.zeros(xx.shape).to(device),
                        torch.zeros(xx.shape).to(device),
                        torch.zeros(xx.shape).to(device))

            if kernel == "multiscale":

                bandwidth_range = [0.2, 0.5, 0.9, 1.3]
                for a in bandwidth_range:
                    XX += a**2 * (a**2 + dxx)**-1
                    YY += a**2 * (a**2 + dyy)**-1
                    XY += a**2 * (a**2 + dxy)**-1

            if kernel == "rbf":

                bandwidth_range = [10, 15, 20, 50]
                for a in bandwidth_range:
                    XX += torch.exp(-0.5*dxx/a)
                    YY += torch.exp(-0.5*dyy/a)
                    XY += torch.exp(-0.5*dxy/a)
            return torch.mean(XX + YY - 2. * XY)
        
        min_length = min(len(self.val_abstract_traces), len(self.test_abstract_traces))
        
        test_traces = self.test_abstract_traces[:min_length]
        val_traces = self.val_abstract_traces[:min_length]

        test_matrix = get_transition_distribution(test_traces)
        val_matrix = get_transition_distribution(val_traces)
        for state in test_matrix:
            if state not in val_matrix:
                val_matrix[state] = {}
            for next in test_matrix[state]:
                if next not in val_matrix[state]:
                    val_matrix[state][next] = 1e-7
        for state in val_matrix:
            if state not in test_matrix:
                test_matrix[state] = {}
            for next in val_matrix[state]:
                if next not in test_matrix[state]:
                    test_matrix[state][next] = 1e-7
        test_distribution = []
        val_distribution = []
        for state in test_matrix:
            for next in test_matrix[state]:
                if test_matrix[state][next] == 1e-7 and val_matrix[state][next] == 1e-7:
                    continue
                test_distribution.append(test_matrix[state][next])
                val_distribution.append(val_matrix[state][next])
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tensor_test = torch.tensor(test_distribution, dtype=torch.float32).to(device)
        tensor_val = torch.tensor(val_distribution, dtype=torch.float32).to(device)

        tensor_val = tensor_val.view(-1, 1)
        tensor_test = tensor_test.view(-1, 1)
        # Calculate the three terms in the MMD formula
        mmd = MMD(tensor_val, tensor_test, kernel="multiscale")
        mmd = mmd.cpu().numpy()
        return mmd
    
    def transition_KL_divergence_instance_level(self):
        def get_transition_distribution_from_traces(traces):
            result = []
            for trace in traces:
                TM = self.prob_model.train_transition_probs
                transition_matrix = {}
                for i in range(len(trace) - 1):
                    state = trace[i]
                    next = trace[i + 1]
                    if state not in transition_matrix:
                        transition_matrix[state] = {}
                    else:
                        if next not in transition_matrix[state]:
                            transition_matrix[state][next] = 1
                        else:
                            transition_matrix[state][next] += 1
                flat_transition = []
                for tm_state in TM.keys():
                    if tm_state not in transition_matrix:
                        transition_matrix[tm_state] = {}
                    for tm_next in TM[tm_state].keys():
                        if tm_next not in transition_matrix[tm_state]:
                            transition_matrix[tm_state][tm_next] = 1e-7
                        else:
                            transition_matrix[tm_state][tm_next] = (
                                transition_matrix[tm_state][tm_next] / len(trace)
                            )
                        flat_transition.append(transition_matrix[tm_state][tm_next])
                
                result.append(flat_transition)
            return result
        
        min_length = min(len(self.val_abstract_traces), len(self.test_abstract_traces))
        
        test_traces = self.test_abstract_traces[:min_length]
        val_traces = self.val_abstract_traces[:min_length]

        test_ditribution = get_transition_distribution_from_traces(test_traces)
        val_distribution = get_transition_distribution_from_traces(val_traces)
        result = []
        for i in range(min_length):
            instance_min_length = min(len(test_ditribution[i]), len(val_distribution[i]))
            test_distribution_instance = test_ditribution[i][:instance_min_length]
            val_distribution_instance = val_distribution[i][:instance_min_length]

            test_instance = np.array(test_distribution_instance)
            val_instance = np.array(val_distribution_instance)
            normal = np.sum(kl_div(test_instance, val_instance))
            abnormal = np.sum(kl_div(val_instance, test_instance))
            avg_div = (normal + abnormal) / 2
            result.append(avg_div)
        return result
    
    def transition_matrix_list(self):
        def get_transition_distribution_by_traces(traces):
            transition_matrix = {}
            for trace in traces:
                for i in range(len(trace) - 1):
                    state = trace[i]
                    next = trace[i + 1]
                    if state not in transition_matrix:
                        transition_matrix[state] = {}
                    else:
                        if next not in transition_matrix[state]:
                            transition_matrix[state][next] = 1
                        else:
                            transition_matrix[state][next] += 1
            for state in transition_matrix:
                total = sum(transition_matrix[state].values())
                for next in transition_matrix[state]:
                    transition_matrix[state][next] /= total
            return transition_matrix
        
        if self.dataset == "truthful_qa":
            test_abnormal_abstract_traces = []
            test_normal_abstract_traces = []
            train_abstract_traces = []
            for i in range(len(self.abstractStateExtraction.train_instances)):
                if self.abstractStateExtraction.train_instances[i]["binary_label"] == 1:
                    train_abstract_traces.append(self.train_abstract_traces[i])
                else:
                    test_abnormal_abstract_traces.append(self.train_abstract_traces[i])
            for i in range(len(self.abstractStateExtraction.test_instances)):
                if self.abstractStateExtraction.test_instances[i]["binary_label"] == 1:
                    test_normal_abstract_traces.append(self.test_abstract_traces[i])
                else:
                    test_abnormal_abstract_traces.append(self.test_abstract_traces[i])
            for i in range(len(self.abstractStateExtraction.val_instances)):
                if self.abstractStateExtraction.val_instances[i]["binary_label"] == 1:
                    test_normal_abstract_traces.append(self.val_abstract_traces[i])
                else:
                    test_abnormal_abstract_traces.append(self.val_abstract_traces[i])
            

        else:
            test_abnormal_abstract_traces = self.test_abstract_traces
            test_normal_abstract_traces = self.val_abstract_traces
            train_abstract_traces = self.train_abstract_traces
        test_distribution = get_transition_distribution_by_traces(test_abnormal_abstract_traces)
        val_distribution = get_transition_distribution_by_traces(test_normal_abstract_traces)
        train_distribution = get_transition_distribution_by_traces(train_abstract_traces)

        # for state in test_distribution:
        #     if state not in val_distribution:
        #         val_distribution[state] = {}
        #     for next in test_distribution[state]:
        #         if next not in val_distribution[state]:
        #             val_distribution[state][next] = 1e-7
        for state in val_distribution:
            if state not in test_distribution:
                test_distribution[state] = {}
            for next in val_distribution[state]:
                if next not in test_distribution[state]:
                    test_distribution[state][next] = 1e-7
        
        for state in val_distribution:
            if state not in train_distribution:
                train_distribution[state] = {}
            for next in val_distribution[state]:
                if next not in train_distribution[state]:
                    train_distribution[state][next] = 1e-7

        for state in train_distribution:
            if state not in val_distribution:
                val_distribution[state] = {}
            for next in train_distribution[state]:
                if next not in val_distribution[state]:
                    val_distribution[state][next] = 1e-7
        
        train_distribution_list = []                
        test_distribution_list = []
        val_distribution_list = []
        for state in train_distribution:
            for next in train_distribution[state]:
                train_distribution_list.append(train_distribution[state][next])

        for state in test_distribution:
            for next in test_distribution[state]:
                test_distribution_list.append(test_distribution[state][next])
        
        for state in val_distribution:
            for next in val_distribution[state]:
                val_distribution_list.append(val_distribution[state][next])
        return train_distribution_list, val_distribution_list, test_distribution_list


    def transition_gain(self):
        def get_transition_coverage_from_traces(traces, state, next, delta):
            num_traces = 0
            for trace in traces:
                cur_transition_count = 0
                for i in range(len(trace) - 1):
                    if state == trace[i] and next == trace[i + 1]:
                        cur_transition_count += 1
                if cur_transition_count / len(trace) >= delta:
                    num_traces += 1
            return num_traces / len(traces)

        transition_gain_dict = {}
        normal_coverage_dict = {}
        abnormal_coverage_dict = {}

        delta = 0.5
        TM = self.prob_model.train_transition_probs
        test_traces = self.test_abstract_traces
        val_traces = self.val_abstract_traces
        for state in TM.keys():
            if state not in transition_gain_dict:
                transition_gain_dict[state] = {}
                normal_coverage_dict[state] = {}
                abnormal_coverage_dict[state] = {}
            for next in TM[state].keys():
                if next not in transition_gain_dict[state]:
                    current_test_transition_coverage = (
                        get_transition_coverage_from_traces(
                            test_traces, state, next, delta
                        )
                    )
                    current_val_transition_coverage = (
                        get_transition_coverage_from_traces(
                            val_traces, state, next, delta
                        )
                    )

                    normal_coverage_dict[state][next] = current_val_transition_coverage
                    abnormal_coverage_dict[state][next] = (
                        current_test_transition_coverage
                    )

                    transition_gain_dict[state][next] = (
                        current_val_transition_coverage
                        - current_test_transition_coverage
                    )
        return transition_gain_dict, normal_coverage_dict, abnormal_coverage_dict

    def state_coverage(self):
        train_pca = self.abstractStateExtraction.pca_train
        test_pca = self.abstractStateExtraction.pca_test
        cluster_labels_train = self.abstractStateExtraction.cluster_train
        cluster_labels_test = self.abstractStateExtraction.cluster_test
        cluster_method = self.abstractStateExtraction.args.cluster_method
        train_set = {}
        test_set = {}

        # Get State Coverage
        if cluster_method == "Grid":
            stacked_pca_traces = np.vstack(train_pca)
            lbd = np.min(stacked_pca_traces, axis=0)
            ubd = np.max(stacked_pca_traces, axis=0)

            # Count how many times pca_test is outside the bounds
            out_of_bounds_count = 0
            total_count = 0
            for test_point in test_pca:
                for p in test_point:
                    if (p < lbd).any() or (p > ubd).any():
                        out_of_bounds_count += 1
                    total_count += 1

            state_coverage = 1 - (out_of_bounds_count / total_count)

        else:
            for i, label in enumerate(cluster_labels_train):
                if label not in train_set:
                    train_set[label] = [train_pca[i]]
                else:
                    train_set[label].append(train_pca[i])
            for i, label in enumerate(cluster_labels_test):
                if label not in test_set:
                    test_set[label] = [test_pca[i]]
                else:
                    test_set[label].append(test_pca[i])
            # Get test count from test_pca np matrix
            test_count = test_pca.shape[0] * test_pca.shape[1]
            train_radius, _ = self._calculate_radius(cluster_method, train_set)
            test_radius, exceeding_count = self._calculate_radius(
                cluster_method, test_set, train_radius
            )
            state_coverage = (test_count - exceeding_count) / test_count

        return state_coverage

    def sensitivity(self):
        epsilon = 0.1
        test_hidden_info = self.abstractStateExtraction.test_hidden_info

        np_test_hidden_info = np.concatenate(test_hidden_info, axis=0)
        np_test_hidden_info_copy = np_test_hidden_info.copy()
        np_test_hidden_info_perturb = np_test_hidden_info_copy + epsilon
        if self.abstractStateExtraction.args.cluster_method == "Grid":
            train_pca = self.abstractStateExtraction.pca_train
            test_hidden_info_perturb = [x + epsilon for x in test_hidden_info]
            pca_test_data_perturb = []
            for i in range(len(test_hidden_info_perturb)):
                pca_test_data_perturb.append(
                    self.abstractStateExtraction.pca_model.transform(
                        test_hidden_info_perturb[i]
                    )
                )
            pca_test_data_original = []
            for i in range(len(test_hidden_info)):
                pca_test_data_original.append(
                    self.abstractStateExtraction.pca_model.transform(
                        test_hidden_info[i]
                    )
                )
            regular_grid = RegularGrid(
                self.abstractStateExtraction.args.abstract_state,
                self.abstractStateExtraction.args.grid_history_dependency_num,
            )

            stacked_pca_traces = np.vstack(train_pca)
            lbd = np.min(stacked_pca_traces, axis=0)
            ubd = np.max(stacked_pca_traces, axis=0)
            grid = Grid(
                lbd, ubd, self.abstractStateExtraction.args.grid_history_dependency_num
            )

            test_perturb_abst_traces = regular_grid.pca_to_abstract_traces(
                grid, pca_test_data_perturb
            )
            test_original_abst_traces = regular_grid.pca_to_abstract_traces(
                grid, pca_test_data_original
            )

            cluster_labels_test_perturb = [
                item
                for sublist2d in test_perturb_abst_traces
                for sublist in sublist2d
                for item in sublist
            ]
            cluster_labels_test_original = [
                item
                for sublist2d in test_original_abst_traces
                for sublist in sublist2d
                for item in sublist
            ]
            different_count = sum(
                a != b
                for a, b in zip(
                    cluster_labels_test_original, cluster_labels_test_perturb
                )
            )
            percentage_difference = different_count / len(cluster_labels_test_perturb)

        else:
            pca_test_data_perturb = self.abstractStateExtraction.pca_model.transform(
                np_test_hidden_info_perturb
            )

            cluster_labels_test_perturb = (
                self.abstractStateExtraction.cluster_model.predict(
                    pca_test_data_perturb
                )
            )
            different_count = sum(
                a != b
                for a, b in zip(
                    self.abstractStateExtraction.cluster_test,
                    cluster_labels_test_perturb,
                )
            )
            percentage_difference = different_count / len(cluster_labels_test_perturb)
        return percentage_difference

    # ===================== State Level =====================
    def sink_state(self):
        train_transition_probs = self.prob_model.train_transition_probs
        sink_state = []
        total_state_num = 0
        for i, row in train_transition_probs.items():
            for j, value in row.items():
                if value == 1:
                    sink_state.append((i, j))
                total_state_num += 1
        return len(sink_state) / total_state_num

    def source_state(self):
        train_transition_probs = self.prob_model.train_transition_probs
        source_state = []
        total_state_num = 0
        for i, row in train_transition_probs.items():
            for j, value in row.items():
                if (
                    value == 0
                    and j in train_transition_probs
                    and i in train_transition_probs[j]
                    and train_transition_probs[j][i] != 0
                ):
                    source_state.append((i, j))
                total_state_num += 1
        return len(source_state) / total_state_num

    def recurrent_state(self):
        train_transition_probs = self.prob_model.train_transition_probs
        recurrent_state = []
        total_state_num = 0
        for i, row in train_transition_probs.items():
            for j, value in row.items():
                if (
                    value != 0
                    and j in train_transition_probs
                    and i in train_transition_probs[j]
                    and train_transition_probs[j][i] != 0
                ):
                    recurrent_state.append((i, j))
                total_state_num += 1
        return len(recurrent_state) / total_state_num

    def _create_low_entropy_matrix(self, n):
        # Create a matrix where the diagonal (same state transition) is 0.95
        # and the remaining probability is spread among other states
        P_matrix = np.full((n, n), 0.05 / (n - 1))
        np.fill_diagonal(P_matrix, 0.95)
        return P_matrix

    def _create_high_entropy_matrix(self, n):
        # Create a matrix where each element is 1/n
        return np.full((n, n), 1 / n)

    def _calculate_entropy(self, P_matrix):
        # Ensure the matrix is square
        assert P_matrix.shape[0] == P_matrix.shape[1], "Matrix must be square"

        # Calculate stationary distribution
        P_matrix_mod = P_matrix.T - np.eye(P_matrix.shape[0])
        P_matrix_mod = np.vstack([P_matrix_mod, np.ones(P_matrix.shape[0])])
        b = np.zeros(P_matrix.shape[0])
        b = np.append(b, 1)
        stationary_vector = np.linalg.lstsq(P_matrix_mod, b, rcond=None)[0]

        # Compute transition entropy for each state and sum them
        total_entropy = 0
        for i in range(P_matrix.shape[0]):
            entropy_i = -np.sum(stationary_vector[i] * P_matrix[i, :] * np.log(P_matrix[i, :]+1e-7))
            total_entropy += entropy_i
            
        return total_entropy

    def stationary_distribution_entropy(self):
        TM = self.prob_model.train_transition_probs
        for key, row in TM.items():
            if sum(row.values()) == 0:
                print(f"Row corresponding to key {key} has a sum of 0")

        states = list(TM.keys())
        for start in states:
            for end in states:
                if end not in TM[start]:
                    TM[start][end] = 0

        P_matrix = np.array([[TM[start][end] for end in states] for start in states])

        stationary_distribution_entropy = self._calculate_entropy(P_matrix)
        lower_bound_entropy = self._calculate_entropy(
            self._create_low_entropy_matrix(len(states))
        )
        upper_bound_entropy = self._calculate_entropy(
            self._create_high_entropy_matrix(len(states))
        )

        result = {
            "stationary_distribution_entropy": stationary_distribution_entropy,
            "lower_bound_entropy": lower_bound_entropy,
            "upper_bound_entropy": upper_bound_entropy,
        }

        return result

    def perplexity_llm(self):
        good_probs_list = [p for i in self.abstractStateExtraction.train_instances for p in i["probs"]]
        bad_probs_list = [p for i in self.abstractStateExtraction.test_instances for p in i["probs"]]
        good_probs = []
        bad_probs = []
        for l in good_probs_list:
            for p in l:
                if p == 0:
                    p = 1e-7
                good_probs.append(p)
        for l in bad_probs_list:
            for p in l:
                if p == 0:
                    p = 1e-7
                bad_probs.append(p)

        print(len(good_probs))
        print(len(bad_probs))

        good_perplexity = np.exp(-np.sum(np.log(good_probs)) / len(good_probs))
        bad_perplexity = np.exp(-np.sum(np.log(bad_probs)) / len(bad_probs))
        return good_perplexity, bad_perplexity

    def smoothed_perplexity_llm(self):
        alpha = 0.5
        vocab_size = [2, 3, 4]
        result = {}
        for size in vocab_size:
            good_probs = []
            bad_probs = []
            for i in self.abstractStateExtraction.train_instances:
                for l in i["probs"]:
                    for p in l:
                        prob = (p + alpha) / (1 + alpha * size)
                        good_probs.append(prob)

            for i in self.abstractStateExtraction.test_instances:
                for l in i["probs"]:
                    for p in l:
                        prob = (p + alpha) / (1 + alpha * size)
                        bad_probs.append(prob)
            
            good_perplexity = np.exp(-np.sum(np.log(good_probs)) / len(good_probs))
            bad_perplexity = np.exp(-np.sum(np.log(bad_probs)) / len(bad_probs))
            
            result[size] = (good_perplexity, bad_perplexity)
        return result
    
    def perplexity_abstract_model(self):
        def get_trace_transition_probs(traces):
            one_set_score_list = []
            for i, one_trace in enumerate(traces):
                if len(one_trace) < 2:
                    continue
                for j, start_state in enumerate(one_trace):
                    if j + 1 < len(one_trace):
                        end_state = one_trace[j + 1]
                    if (start_state in self.prob_model.train_transition_probs) and (
                        end_state in self.prob_model.train_transition_probs[start_state]
                    ):
                        score = self.prob_model.train_transition_probs[start_state][end_state]
                        one_set_score_list.append(score)
                    else:
                        one_set_score_list.append(0.0)
            return one_set_score_list
        val_traces = self.val_abstract_traces
        test_traces = self.test_abstract_traces

        val_probs = get_trace_transition_probs(val_traces)
        test_probs = get_trace_transition_probs(test_traces)

        val_perplexity = np.exp(-np.sum(np.log(val_probs)) / len(val_probs))
        test_perplexity = np.exp(-np.sum(np.log(test_probs)) / len(test_probs))
        return val_perplexity, test_perplexity
        

    def smoothed_perplexity_abstract_model(self):
        def get_trace_transition_probs(traces):
            one_set_score_list = []
            for i, one_trace in enumerate(traces):
                if len(one_trace) < 2:
                    continue
                for j, start_state in enumerate(one_trace):
                    if j + 1 < len(one_trace):
                        end_state = one_trace[j + 1]
                    if (start_state in self.prob_model.train_transition_probs) and (
                        end_state in self.prob_model.train_transition_probs[start_state]
                    ):
                        score = self.prob_model.train_transition_probs[start_state][end_state]
                        one_set_score_list.append(score)
                    else:
                        one_set_score_list.append(0.0)
            return one_set_score_list
        
        val_traces = self.val_abstract_traces
        test_traces = self.test_abstract_traces
        alpha = 0.5
        vocab_size = [2, 3, 4]
        result = {}
        val_probs = get_trace_transition_probs(val_traces)
        test_probs = get_trace_transition_probs(test_traces)
        for size in vocab_size:
            good_probs = []
            bad_probs = []
            for p in val_probs:
                prob = (p + alpha) / (1 + alpha * size)
                good_probs.append(prob)

            for p in test_probs:
                prob = (p + alpha) / (1 + alpha * size)
                bad_probs.append(prob)
            
            good_perplexity = np.exp(-np.sum(np.log(good_probs)) / len(good_probs))
            bad_perplexity = np.exp(-np.sum(np.log(bad_probs)) / len(bad_probs))
            
            result[size] = (good_perplexity, bad_perplexity)
        return result
