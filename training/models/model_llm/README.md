---
library_name: setfit
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
base_model: sentence-transformers/all-MiniLM-L6-v2
metrics:
- accuracy
widget:
- text: Language models (LMs) have become ubiquitous in both NLP research and in commercial
    product offerings. As their commercial importance has surged, the most powerful
    models have become closed off, gated behind proprietary interfaces, with important
    details of their training data, architectures, and development undisclosed. Given
    the importance of these details in scientifically studying these models, including
    their biases and potential risks, we believe it is essential for the research
    community to have access to powerful, truly open LMs. To this end, we have built
    OLMo, a competitive, truly Open Language Model, to enable the scientific study
    of language models. Unlike most prior efforts that have only released model weights
    and inference code, we release OLMo alongside open training data and training
    and evaluation code. We hope this release will empower the open research community
    and inspire a new wave of innovation.
- text: Language is essentially a complex, intricate system of human expressions governed
    by grammatical rules. It poses a significant challenge to develop capable AI algorithms
    for comprehending and grasping a language. As a major approach, language modeling
    has been widely studied for language understanding and generation in the past
    two decades, evolving from statistical language models to neural language models.
    Recently, pre-trained language models (PLMs) have been proposed by pre-training
    Transformer models over large-scale corpora, showing strong capabilities in solving
    various NLP tasks. Since researchers have found that model scaling can lead to
    performance improvement, they further study the scaling effect by increasing the
    model size to an even larger size. Interestingly, when the parameter scale exceeds
    a certain level, these enlarged language models not only achieve a significant
    performance improvement but also show some special abilities that are not present
    in small-scale language models. To discriminate the difference in parameter scale,
    the research community has coined the term large language models (LLM) for the
    PLMs of significant size. Recently, the research on LLMs has been largely advanced
    by both academia and industry, and a remarkable progress is the launch of ChatGPT,
    which has attracted widespread attention from society. The technical evolution
    of LLMs has been making an important impact on the entire AI community, which
    would revolutionize the way how we develop and use AI algorithms. In this survey,
    we review the recent advances of LLMs by introducing the background, key findings,
    and mainstream techniques. In particular, we focus on four major aspects of LLMs,
    namely pre-training, adaptation tuning, utilization, and capacity evaluation.
    Besides, we also summarize the available resources for developing LLMs and discuss
    the remaining issues for future directions.
pipeline_tag: text-classification
inference: true
model-index:
- name: SetFit with sentence-transformers/all-MiniLM-L6-v2
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: Unknown
      type: unknown
      split: test
    metrics:
    - type: accuracy
      value: 1.0
      name: Accuracy
---

# SetFit with sentence-transformers/all-MiniLM-L6-v2

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 256 tokens
- **Number of Classes:** 2 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label | Examples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|:------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0     | <ul><li>'This book aims to provide an introduction to the topic of deep learning algorithms. We review essential components of deep learning algorithms in full mathematical detail including different artificial neural network (ANN) architectures (such as fully-connected feedforward ANNs, convolutional ANNs, recurrent ANNs, residual ANNs, and ANNs with batch normalization) and different optimization algorithms (such as the basic stochastic gradient descent (SGD) method, accelerated methods, and adaptive methods). We also cover several theoretical aspects of deep learning algorithms such as approximation capacities of ANNs (including a calculus for ANNs), optimization theory (including Kurdyka-Łojasiewicz inequalities), and generalization errors. In the last part of the book some deep learning approximation methods for PDEs are reviewed including physics-informed neural networks (PINNs) and deep Galerkin methods. We hope that this book will be useful for students and scientists who do not yet have any background in deep learning at all and would like to gain a solid foundation as well as for practitioners who would like to obtain a firmer mathematical understanding of the objects and methods considered in deep learning.'</li><li>'We introduce a framework for generating, organizing, and reasoning with computational knowledge. It is motivated by the observation that most problems in Computational Sciences and Engineering (CSE) can be formulated as that of completing (from data) a computational graph (or hypergraph) representing dependencies between functions and variables. Nodes represent variables, and edges represent functions. Functions and variables may be known, unknown, or random. Data comes in the form of observations of distinct values of a finite number of subsets of the variables of the graph (satisfying its functional dependencies). The underlying problem combines a regression problem (approximating unknown functions) with a matrix completion problem (recovering unobserved variables in the data). Replacing unknown functions by Gaussian Processes (GPs) and conditioning on observed data provides a simple but efficient approach to completing such graphs. Since this completion process can be reduced to an algorithm, as one solves 2–√ on a pocket calculator without thinking about it, one could, with the automation of the proposed framework, solve a complex CSE problem by drawing a diagram. Compared to traditional kriging, the proposed framework can be used to recover unknown functions with much scarcer data by exploiting interdependencies between multiple functions and variables. The underlying problem could therefore also be interpreted as a generalization of that of solving linear systems of equations to that of approximating unknown variables and functions with noisy, incomplete, and nonlinear dependencies. Numerous examples illustrate the flexibility, scope, efficacy, and robustness of the proposed framework and show how it can be used as a pathway to identifying simple solutions to classical CSE problems (digital twin modeling, dimension reduction, mode decomposition, etc.).'</li><li>'The Language of Thought Hypothesis suggests that human cognition operates on a structured, language-like system of mental representations. While neural language models can naturally benefit from the compositional structure inherently and explicitly expressed in language data, learning such representations from non-linguistic general observations, like images, remains a challenge. In this work, we introduce the Neural Language of Thought Model (NLoTM), a novel approach for unsupervised learning of LoTH-inspired representation and generation. NLoTM comprises two key components: (1) the Semantic Vector-Quantized Variational Autoencoder, which learns hierarchical, composable discrete representations aligned with objects and their properties, and (2) the Autoregressive LoT Prior, an autoregressive transformer that learns to generate semantic concept tokens compositionally, capturing the underlying data distribution. We evaluate NLoTM on several 2D and 3D image datasets, demonstrating superior performance in downstream tasks, out-of-distribution generalization, and image generation quality compared to patch-based VQ-VAE and continuous object-centric representations. Our work presents a significant step towards creating neural networks exhibiting more human-like understanding by developing LoT-like representations and offers insights into the intersection of cognitive science and machine learning'</li></ul> |
| 1     | <ul><li>'The fields of generative AI and transfer learning have experienced remarkable advancements in recent years especially in the domain of Natural Language Processing (NLP). Transformers have been at the heart of these advancements where the cutting-edge transformer-based Language Models (LMs) have led to new state-of-the-art results in a wide spectrum of applications. While the number of research works involving neural LMs is exponentially increasing, their vast majority are high-level and far from self-contained. Consequently, a deep understanding of the literature in this area is a tough task especially in the absence of a unified mathematical framework explaining the main types of neural LMs. We address the aforementioned problem in this tutorial where the objective is to explain neural LMs in a detailed, simplified and unambiguous mathematical framework accompanied by clear graphical illustrations. Concrete examples on widely used models like BERT and GPT2 are explored. Finally, since transformers pretrained on language-modeling-like tasks have been widely adopted in computer vision and time series applications, we briefly explore some examples of such solutions in order to enable readers to understand how transformers work in the aforementioned domains and compare this use with the original one in NLP'</li><li>'Multilingual Large Language Models are capable of using powerful Large Language Models to handle and respond to queries in multiple languages, which achieves remarkable success in multilingual natural language processing tasks. Despite these breakthroughs, there still remains a lack of a comprehensive survey to summarize existing approaches and recent developments in this field. To this end, in this paper, we present a thorough review and provide a unified perspective to summarize the recent progress as well as emerging trends in multilingual large language models (MLLMs) literature. The contributions of this paper can be summarized: (1) First survey: to our knowledge, we take the first step and present a thorough review in MLLMs research field according to multi-lingual alignment; (2) New taxonomy: we offer a new and unified perspective to summarize the current progress of MLLMs; (3) New frontiers: we highlight several emerging frontiers and discuss the corresponding challenges; (4) Abundant resources: we collect abundant open-source resources, including relevant papers, data corpora, and leaderboards. We hope our work can provide the community with quick access and spur breakthrough research in MLLMs.'</li><li>'While large language models (LLMs) like ChatGPT have shown impressive capabilities in Natural Language Processing (NLP) tasks, a systematic investigation of their potential in this field remains largely unexplored. This study aims to address this gap by exploring the following questions: (1) How are LLMs currently applied to NLP tasks in the literature? (2) Have traditional NLP tasks already been solved with LLMs? (3) What is the future of the LLMs for NLP? To answer these questions, we take the first step to provide a comprehensive overview of LLMs in NLP. Specifically, we first introduce a unified taxonomy including (1) parameter-frozen application and (2) parameter-tuning application to offer a unified perspective for understanding the current progress of LLMs in NLP. Furthermore, we summarize the new frontiers and the associated challenges, aiming to inspire further groundbreaking advancements. We hope this work offers valuable insights into the {potential and limitations} of LLMs in NLP, while also serving as a practical guide for building effective LLMs in NLP.'</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |

## Evaluation

### Metrics
| Label   | Accuracy |
|:--------|:---------|
| **all** | 1.0      |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("Language models (LMs) have become ubiquitous in both NLP research and in commercial product offerings. As their commercial importance has surged, the most powerful models have become closed off, gated behind proprietary interfaces, with important details of their training data, architectures, and development undisclosed. Given the importance of these details in scientifically studying these models, including their biases and potential risks, we believe it is essential for the research community to have access to powerful, truly open LMs. To this end, we have built OLMo, a competitive, truly Open Language Model, to enable the scientific study of language models. Unlike most prior efforts that have only released model weights and inference code, we release OLMo alongside open training data and training and evaluation code. We hope this release will empower the open research community and inspire a new wave of innovation.")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median  | Max |
|:-------------|:----|:--------|:----|
| Word count   | 90  | 188.875 | 270 |

| Label | Training Sample Count |
|:------|:----------------------|
| 0     | 5                     |
| 1     | 3                     |

### Training Hyperparameters
- batch_size: (16, 16)
- num_epochs: (5, 5)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 200
- body_learning_rate: (2e-05, 2e-05)
- head_learning_rate: 2e-05
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch | Step | Training Loss | Validation Loss |
|:-----:|:----:|:-------------:|:---------------:|
| 0.01  | 1    | 0.1826        | -               |
| 0.5   | 50   | 0.0764        | -               |
| 1.0   | 100  | 0.0011        | -               |
| 1.5   | 150  | 0.0006        | -               |
| 2.0   | 200  | 0.0004        | -               |
| 2.5   | 250  | 0.0003        | -               |
| 3.0   | 300  | 0.0003        | -               |
| 3.5   | 350  | 0.0003        | -               |
| 4.0   | 400  | 0.0002        | -               |
| 4.5   | 450  | 0.0002        | -               |
| 5.0   | 500  | 0.0002        | -               |

### Framework Versions
- Python: 3.8.16
- SetFit: 1.1.0
- Sentence Transformers: 3.2.1
- Transformers: 4.41.2
- PyTorch: 2.0.0
- Datasets: 2.19.2
- Tokenizers: 0.19.1

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->