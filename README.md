# 🚀🦸 LUNA: A Model-based LLM-Oriented Universal Analysis Framework

A comprehensive framework to construct abstract models for analyzing various tasks. The current experiments are conducted on OOD Detection, Adversarial Attacks, and Hallucination.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [API Overview](#api-overview)
- [Contributing](#contributing)
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


## API Overview

- **Model Abstraction**: Explain briefly what this does, and possibly give a small example.

- **Metrics Calculation**: Briefly explain and provide a small example.


## License

[MIT](LICENSE)

---
