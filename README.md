# 🚀🦸 LUNA: A Model-based LLM-Oriented Universal Analysis Framework

A comprehensive framework to construct abstract models for analyzing various tasks.
## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [API Overview](#api-overview)
- [License](#license)

## Installation

### Setting up Python Environment

1. Ensure you have Python 3.8+ installed.
2. Clone this repository:
   ```bash
   git clone <repository-link>
3. Navigate to the project directory and set up a virtual environment:
   ```bash
   cd LUNA
   conda create -n env_name python=3.8
4. Activate the virtual environment:
   ```bash
   conda activate env_name
5. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt

## Usage

### LUNA Framework

The `luna` folder contains all the essential APIs for model abstraction and metrics calculation.

1. Navigate to the `luna` directory:

2. Use the API as needed. (Provide specific instructions or code examples here.)

### 🚀 **Example**

### **Initializing the MetricsAppEvalCollections Object**

To analyze your data, you first need to initialize the `MetricsAppEvalCollections` class. This class is responsible for collecting various metrics on your data based on abstract model representations:

```python
eval_obj = MetricsAppEvalCollections(
    state_abstract_args_obj,
    prob_args_obj,
    train_instances,
    val_instances,
    test_instances,
)
```

Where:

- `state_abstract_args_obj`: A namespace object containing arguments related to state abstraction (e.g., dataset name, block index, info type).
- `prob_args_obj`: A namespace object containing arguments related to probability calculations (e.g., dataset, PCA dimension, model type).
- `train_instances`, `val_instances`, `test_instances`: The data instances for training, validation, and testing.

### **Collecting Metrics**

Once the `MetricsAppEvalCollections` object is initialized, you can then collect various metrics. Here are some examples:

- Evaluating the model:

  ```python
  aucroc, accuracy, f1_score, _, _, abnormal_threshold = eval_obj.get_eval_result()
  ```

- Calculating preciseness:

  ```python
  preciseness = eval_obj.preciseness()
  ```

- Calculating entropy:

  ```python
  entropy = eval_obj.entropy()
  ```

... and many other metrics as shown in the `rq3` function.

---


## Datasets
We've conducted experiments on the following datasets:

Certainly! Here's a clearer version of the Markdown text, with enhanced structure and readability:

---

### TruthfulQA Dataset Overview

- **Purpose**: TruthfulQA is designed to evaluate the truthfulness of Large Language Models (LLMs) in generating answers to questions.
- **Composition**: The dataset contains 817 questions across 38 categories of potential falsehoods, such as misconceptions and fiction.
- **Truth Assessment**: Answers' truthfulness is judged using fine-tuned GPT-3-13B models, classifying each response as true or false.

### Pre-requisites for Using the TruthfulQA Dataset

Before utilizing the TruthfulQA dataset, certain preparatory steps are required:

1. **OpenAI API Account**:
   - Ensure you have an active account with OpenAI's API.

2. **Model Fine-Tuning**:
   - Follow the guide on [Inference-Time Intervention: Eliciting Truthful Answers from a Language Model](https://github.com/likenneth/honest_llama#truthfulqa-evaluation) to create GPT-JUDGE, a fine-tuned GPT-3 model.
   - We provided a READY-TO-USE GPT-Judge.
     ```bash
     judge_model_key = "curie:ft-momentum-lab-2023-07-07-11-31-31"
     info_model_key = "curie:ft-momentum-lab-2023-07-07-14-15-29"
     ```

3. **Dataset Preparation**:
   - Run the `add_scores_to_truthful_qa.py` script to process the dataset. 
   - Make sure to update the `file_name` and `file_with_score` variables in the script with the correct file paths.

   Execute the following command in your terminal:
   ```bash
   python add_scores_to_truthful_qa.py
   ```

---

Sure, here's a revised version of the descriptions for the SST-2 and AdvGLUE++ datasets:

---

### SST-2 Dataset Overview

- **Context**: SST-2 is adapted for Out-of-Distribution (OOD) analysis, based on the sentiment analysis dataset by Wang et al. [2].
- **Composition**: This dataset builds upon the original SST-2 dataset [3] and includes word-level and sentence-level style transferred data. This involves transforming original sentences into a different style.
- **Size and Distribution**:
   - Total Sentences: 9,603.
   - In-Distribution (ID) Data: 873 sentences.
   - Out-of-Distribution (OOD) Data: 8,730 sentences.

### AdvGLUE++ Dataset Overview

- **Purpose**: AdvGLUE++ is used for evaluating model performance against adversarial attacks.
- **Tasks and Methods**:
   - Encompasses three types of tasks: Sentiment Classification, Duplicate Question Detection, and Multi-Genre Natural Language Inference.
   - Incorporates five word-level attack methods.
- **Dataset Size**: Consists of a total of 11,484 data points.

---


The dataset can be found in LUNA/datasets folder. You can also check generate.py file about how to use it.


## API Overview**

### **Model Abstraction**

The process of abstracting the behavior and properties of a system into a simplified representation that retains only the essential characteristics of the original system. In the context of this framework, model abstraction is done based on state and probabilistic models.

#### **1. ProbabilisticModel (from probabilistic_abstraction_model.py)**
- **Purpose**: Provides a base for creating probabilistic models based on abstracted states.
  
- **Usage Examples**:
  ```python
  # Initialize the ProbabilisticModel
  prob_model = ProbabilisticModel(args)
  
  # Evaluate LLM performance on a dataset task
  prob_model.eval_llm_performance_on_dataset_task()
  
  # Compose scores with ground truths
  prob_model.compose_scores_with_groundtruths_pair()
  ```

#### **2. AbstractStateExtraction (from state_abstraction_utils.py)**
- **Purpose**: Extracts abstract states from provided data instances.
  
- **Usage Examples**:
  ```python
  # Initialize the AbstractStateExtraction
  state_extractor = AbstractStateExtraction(args)
  
  # Perform PCA on data
  state_extractor.perform_pca()
  
  # (Additional method usage examples would be included if available in the file)
  ```

### **Metrics Calculation**

Metrics provide a quantitative measure to evaluate the performance and characteristics of models. In this framework, metrics evaluate the quality and behavior of abstracted models.

#### **1. MetricsAppEvalCollections (from metrics_appeval_collection.py)**
- **Purpose**: Acts as a central utility for metric evaluations based on state abstractions.
  
- **Usage Examples**:
  ```python
  # Initialize the MetricsAppEvalCollections
  metrics_evaluator = MetricsAppEvalCollections(args_obj1, args_obj2, train_data, val_data, test_data)
  
  # Retrieve evaluation results
  aucroc, accuracy, f1_score, _, _, _ = metrics_evaluator.get_eval_result()
  
  # Calculate the preciseness of predictions
  preciseness_mean, preciseness_max = metrics_evaluator.preciseness()
  ```

### **📚 Metrics Categories**

#### **Model-wise Metrics**:
- **Succinctness (SUC)**
- **Stationary Distribution Entropy (SDE)**
- **Sink State (SS)**
- **Sensitivity (SEN)**
- **Coverage (COV)**
- **Perplexity (PERP)**

#### **Semantic Metrics**:
- **Preciseness (PRE)**
- **Entropy (ENT)**
- **Surprise Level (SL)**
- **N-gram Derivative Trend (NDT)**
- **Instance Value Trend (IVT)**
- **N-gram Value Trend (NVT)**

---

### **Usage Overview**

### Detailed Experiment Setup Example with LUNA Framework

To demonstrate the practical application of the LUNA framework, here is a detailed example based on the provided Python script `RQ23-all-settings.py`.

#### Step 1: Import Necessary Modules

Start by importing the required modules from the LUNA framework and other dependencies.

```python
from luna.metrics_appeval_collection import MetricsAppEvalCollections
import luna.data_loader as data_loader
from types import SimpleNamespace
import pandas as pd
from itertools import product
from collections import defaultdict
import os
from copy import deepcopy
import numpy as np
from scipy.stats import mannwhitneyu
```

#### Step 2: Set Experiment Parameters

Configure the parameters for your experiment, including the language model, dataset, and various settings for abstraction and analysis.

```python
llm = "code_llama_7B_python"  # Language model
dataset = "humaneval"         # Dataset
info_type = "hidden_states"
extract_block_idx = "31"
abstraction_methods = ["Grid", "Cluster-based"]
partition_nums = [5, 10, 15]
abstract_state_nums = [200, 400, 600]
# ... other configurations
```

#### Step 3: Initialize Results Storage

Prepare a structure to store the results of your experiments.

```python
results = defaultdict(list)
```

Step 4: Running the Experiment with Command-line Argument
To run the experiment with a specified result save path, use the command-line interface. The script RQ23-all-settings.py accepts a --result_save_path argument to determine where to save the results. Here's how to run the experiment:

```bash
python RQ23-all-settings.py --result_save_path <path-to-save-results>
```
This command will execute the script with the provided result save path. Ensure the path is correctly specified relative to the script's location. The script will internally handle the experiment setup, execution, and result storage based on the configurations set within the file.



```python
# Iterate through each abstraction method (Grid and Cluster-based)
for abstraction_method in abstraction_methods:
    # If Grid abstraction method is chosen
    if abstraction_method == "Grid":
        # Iterate through possible partition numbers for the Grid method
        for partition_num in partition_nums:
            # Explore the impact of different PCA dimensions
            for pca_dim in grid_pca_dims:
                # Explore results for different probability models (DTMC and HMM)
                for model_type in probability_models:
                    # Explore different numbers of Grid history dependency
                    for grid_history_dependency_num in grid_history_dependency:
                        # If the model is Hidden Markov Model (HMM)
                        if model_type == "HMM":
                            # Iterate over different numbers of HMM components
                            for hmm_n_comp in grid_hmm_n_comps:
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
                                    hmm_n_comp=hmm_n_comp,
                                    grid_history_dependency_num=grid_history_dependency_num,
                                )
                                print("result", result)
                                if result:
                                    write_result_to_csv(result, settings_str)
                               
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
                                grid_history_dependency_num=grid_history_dependency_num,
                            )
                            if result:
                                write_result_to_csv(result, settings_str)

    # If Cluster-based abstraction method is chosen
    elif abstraction_method in CLUSTER_METHODS:
        # (similar logic as above for cluster-based experiments)
        for abstract_state_num, cluster_method in product(
            abstract_state_nums, cluster_methods
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
                                abstract_state_num,
                                pca_dim,
                                model_type,
                                hmm_n_comp=hmm_n_comp,
                            )
                            if result:
                                write_result_to_csv(result, settings_str)
                    else:
                        train_instances = deepcopy(train_instances_loaded)
                        val_instances = deepcopy(val_instances_loaded)
                        test_instances = deepcopy(test_instances_loaded)
                        result, settings_str = run_experiment(
                            train_instances,
                            val_instances,
                            test_instances,
                            cluster_method,
                            abstract_state_num,
                            pca_dim,
                            model_type,
                        )
                        if result:
                            write_result_to_csv(result, settings_str)

```

#### Step 5: Analyzing and Storing Results

Finally, analyze the collected results and store them for further evaluation.

```python
# Example of analyzing results
for metric, values in results.items():
    print(f"Results for {metric}: {values}")

# Storing results in a CSV file
results_df = pd.DataFrame.from_dict(results)
results_df.to_csv('experiment_results.csv', index=False)

```

This detailed example provides a clear guide on how to set up and run a comprehensive experiment using the LUNA framework, demonstrating its capability to handle complex configurations and analyses.

---



## License

[FPA](LICENSE)

---
