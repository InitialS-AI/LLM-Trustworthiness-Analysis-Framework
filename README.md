# 🚀🌙 LUNA: A Model-based LLM-Oriented Universal Analysis Framework

A comprehensive framework to construct abstract models for analyzing various tasks. The current experiments are conducted on OOD Detection, Adversarial Attacks, and Hallucination.

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

\```python
eval_obj = MetricsAppEvalCollections(
    state_abstract_args_obj,
    prob_args_obj,
    train_instances,
    val_instances,
    test_instances,
)
\```

Where:

- `state_abstract_args_obj`: A namespace object containing arguments related to state abstraction (e.g., dataset name, block index, info type).
- `prob_args_obj`: A namespace object containing arguments related to probability calculations (e.g., dataset, PCA dimension, model type).
- `train_instances`, `val_instances`, `test_instances`: The data instances for training, validation, and testing.

### **Collecting Metrics**

Once the `MetricsAppEvalCollections` object is initialized, you can then collect various metrics. Here are some examples:

- Evaluating the model:

  \```python
  aucroc, accuracy, f1_score, _, _, abnormal_threshold = eval_obj.get_eval_result()
  \```

- Calculating preciseness:

  \```python
  preciseness = eval_obj.preciseness()
  \```

- Calculating entropy:

  \```python
  entropy = eval_obj.entropy()
  \```

... and many other metrics as shown in the `rq3` function.

---


## Datasets
We've conducted experiments on the following datasets:

- **TruthfulQA** - For the hallucination dataset, we choose
TruthfulQA [1], which is designed for measuring the truth-
fulness of LLM in generating answers to questions. It consists
of 817 questions, with 38 categories of falsehood, e.g., mis-
conceptions and fiction. The ground truth of the answers is
judged by fine-tuned GPT-3-13B models [1] to classify each
answer as true or false.
- **SST-2** - For Out-of-Distribution, we adapt the sentiment
analysis dataset created by Wang et al. [2]. It is based on
SST-2 dataset [3], and contains word-level and sentence-
level style transferred data, where the original sentences
are transformed to another style. It contains a total of 9,603
sentences, with 873 in-distribution (ID) data and 8,730 OOD
data.
- **AdvGLUE++** - For the adversarial attack dataset, we
use AdvGLUE++ [4], which consists of three types of tasks
(sentiment classification, duplicate question detection, and
multi-genre natural language inference) and five word-level
attack methods. It contains 11,484 data in total.

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




## License

[MIT](LICENSE)

---
