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
- text: Document retrieval systems have experienced a revitalized interest with the
    advent of retrieval-augmented generation (RAG). RAG architecture offers a lower
    hallucination rate than LLM-only applications. However, the accuracy of the retrieval
    mechanism is known to be a bottleneck in the efficiency of these applications.
    A particular case of subpar retrieval performance is observed in situations where
    multiple documents from several different but related topics are in the corpus.
    We have devised a new vectorization method that takes into account the topic information
    of the document. The paper introduces this new method for text vectorization and
    evaluates it in the context of RAG. Furthermore, we discuss the challenge of evaluating
    RAG systems, which pertains to the case at hand
- text: Decision trees (DTs) epitomize what have become to be known as interpretable
    machine learning (ML) models. This is informally motivated by paths in DTs being
    often much smaller than the total number of features. This paper shows that in
    some settings DTs can hardly be deemed interpretable, with paths in a DT being
    arbitrarily larger than a PI-explanation, i.e. a subset-minimal set of feature
    values that entails the prediction. As a result, the paper proposes a novel model
    for computing PI-explanations of DTs, which enables computing one PI-explanation
    in polynomial time. Moreover, it is shown that enumeration of PI-explanations
    can be reduced to the enumeration of minimal hitting sets. Experimental results
    were obtained on a wide range of publicly available datasets with well-known DT-learning
    tools, and confirm that in most cases DTs have paths that are proper supersets
    of PI-explanations.
- text: Retrieval-Augmented Generation (RAG) allows overcoming the limited knowledge
    of LLMs by extending the input with external information. As a consequence, the
    contextual inputs to the model become much longer which slows down decoding time
    directly translating to the time a user has to wait for an answer. We address
    this challenge by presenting COCOM, an effective context compression method, reducing
    long contexts to only a handful of Context Embeddings speeding up the generation
    time by a large margin. Our method allows for different compression rates trading
    off decoding time for answer quality. Compared to earlier methods, COCOM allows
    for handling multiple contexts more effectively, significantly reducing decoding
    time for long inputs. Our method demonstrates a speed-up of up to 5.69 × while
    achieving higher performance compared to existing efficient context compression
    methods
- text: Evaluating large language models (LLMs) on their linguistic reasoning capabilities
    is an important task to understand the gaps in their skills that may surface during
    large-scale adoption. In this work, we investigate the abilities of such models
    to perform abstract multilingual reasoning through the lens of linguistic puzzles
    on extremely low-resource languages. As these translation tasks involve inductive
    and deductive reasoning from reference instances, we examine whether diverse auxiliary
    demonstrations can be automatically induced from seed exemplars, through analogical
    prompting. We employ a two-stage procedure, first generating analogical exemplars
    with a language model, and then applying them in-context along with provided target
    language exemplars. Our results on the modeLing dataset show that analogical prompting
    is effective in eliciting models’ knowledge of language grammar similarities,
    boosting the performance of GPT-4o by as much as 8.1% and Llama-3.1-405B-Instruct
    by 5.9% over chain-of-thought approaches. These gains are attributable to the
    analogical demonstrations, both when self-generated as well as when produced by
    weaker multilingual models. Furthermore, we demonstrate that our method generalizes
    to other tasks present in Linguistics Olympiad competitions, achieving sizable
    improvements across all problem types and difficulty levels included in the LINGOLY
    dataset with GPT-4o. We also report several findings about interesting phenomena
    which drive linguistic reasoning performance, suggesting that such puzzles are
    a valuable benchmark for new reasoning methods.
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
      value: 0.75
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
| Label | Examples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|:------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1     | <ul><li>"There is no limit to how much a robot might explore and learn, but all of that knowledge needs to be searchable and actionable. Within language research, retrieval augmented generation (RAG) has become the workhouse of large-scale non-parametric knowledge, however existing techniques do not directly transfer to the embodied domain, which is multimodal, data is highly correlated, and perception requires abstraction.To address these challenges, we introduce Embodied-RAG, a framework that enhances the foundational model of an embodied agent with a non-parametric memory system capable of autonomously constructing hierarchical knowledge for both navigation and language generation. Embodied-RAG handles a full range of spatial and semantic resolutions across diverse environments and query types, whether for a specific object or a holistic description of ambiance. At its core, Embodied-RAG's memory is structured as a semantic forest, storing language descriptions at varying levels of detail. This hierarchical organization allows the system to efficiently generate context-sensitive outputs across different robotic platforms. We demonstrate that Embodied-RAG effectively bridges RAG to the robotics domain, successfully handling over 200 explanation and navigation queries across 19 environments, highlighting its promise for general-purpose non-parametric system for embodied agents."</li><li>"In the age of mobile internet, user data, often referred to as memories, is continuously generated on personal devices. Effectively managing and utilizing this data to deliver services to users is a compelling research topic. In this paper, we introduce a novel task of crafting personalized agents powered by large language models (LLMs), which utilize a user's smartphone memories to enhance downstream applications with advanced LLM capabilities. To achieve this goal, we introduce EMG-RAG, a solution that combines Retrieval-Augmented Generation (RAG) techniques with an Editable Memory Graph (EMG). This approach is further optimized using Reinforcement Learning to address three distinct challenges: data collection, editability, and selectability. Extensive experiments on a real-world dataset validate the effectiveness of EMG-RAG, achieving an improvement of approximately 10% over the best existing approach. Additionally, the personalized agents have been transferred into a real smartphone AI assistant, which leads to enhanced usability."</li><li>'The choice of embedding model is a crucial step in the design of Retrieval Augmented Generation (RAG) systems. Given the sheer volume of available options, identifying clusters of similar models streamlines this model selection process. Relying solely on benchmark performance scores only allows for a weak assessment of model similarity. Thus, in this study, we evaluate the similarity of embedding models within the context of RAG systems. Our assessment is two-fold: We use Centered Kernel Alignment to compare embeddings on a pair-wise level. Additionally, as it is especially pertinent to RAG systems, we evaluate the similarity of retrieval results between these models using Jaccard and rank similarity. We compare different families of embedding models, including proprietary ones, across five datasets from the popular Benchmark Information Retrieval (BEIR). Through our experiments we identify clusters of models corresponding to model families, but interestingly, also some inter-family clusters. Furthermore, our analysis of top-k retrieval similarity reveals high-variance at low k values. We also identify possible open-source alternatives to proprietary models, with Mistral exhibiting the highest similarity to OpenAI models.'</li></ul> |
| 0     | <ul><li>'Today, intelligent systems that offer artificial intelligence capabilities often rely on machine learning. Machine learning describes the capacity of systems to learn from problem-specific training data to automate the process of analytical model building and solve associated tasks. Deep learning is a machine learning concept based on artificial neural networks. For many applications, deep learning models outperform shallow machine learning models and traditional data analysis approaches. In this article, we summarize the fundamentals of machine learning and deep learning to generate a broader understanding of the methodical underpinning of current intelligent systems. In particular, we provide a conceptual distinction between relevant terms and concepts, explain the process of automated analytical model building through machine learning and deep learning, and discuss the challenges that arise when implementing such intelligent systems in the field of electronic markets and networked business. These naturally go beyond technological aspects and highlight issues in human-machine interaction and artificial intelligence servitization'</li><li>'This article provides an introduction to surface code quantum computing. We first estimate the size and speed of a surface code quantum computer. We then introduce the concept of the stabilizer, using two qubits, and extend this concept to stabilizers acting on a two-dimensional array of physical qubits, on which we implement the surface code. We next describe how logical qubits are formed in the surface code array and give numerical estimates of their fault-tolerance. We outline how logical qubits are physically moved on the array, how qubit braid transformations are constructed, and how a braid between two logical qubits is equivalent to a controlled-NOT. We then describe the single-qubit Hadamard, S and T operators, completing the set of required gates for a universal quantum computer. We conclude by briefly discussing physical implementations of the surface code. We include a number of appendices in which we provide supplementary information to the main text.'</li><li>'Tree boosting is a highly effective and widely used machine learning method. In this paper, we describe a scalable end-to-end tree boosting system called XGBoost, which is used widely by data scientists to achieve state-of-the-art results on many machine learning challenges. We propose a novel sparsity-aware algorithm for sparse data and weighted quantile sketch for approximate tree learning. More importantly, we provide insights on cache access patterns, data compression and sharding to build a scalable tree boosting system. By combining these insights, XGBoost scales beyond billions of examples using far fewer resources than existing systems.'</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

## Evaluation

### Metrics
| Label   | Accuracy |
|:--------|:---------|
| **all** | 0.75     |

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
preds = model("Document retrieval systems have experienced a revitalized interest with the advent of retrieval-augmented generation (RAG). RAG architecture offers a lower hallucination rate than LLM-only applications. However, the accuracy of the retrieval mechanism is known to be a bottleneck in the efficiency of these applications. A particular case of subpar retrieval performance is observed in situations where multiple documents from several different but related topics are in the corpus. We have devised a new vectorization method that takes into account the topic information of the document. The paper introduces this new method for text vectorization and evaluates it in the context of RAG. Furthermore, we discuss the challenge of evaluating RAG systems, which pertains to the case at hand")
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
| Training set | Min | Median | Max |
|:-------------|:----|:-------|:----|
| Word count   | 95  | 154.0  | 216 |

| Label | Training Sample Count |
|:------|:----------------------|
| 0     | 5                     |
| 1     | 8                     |

### Training Hyperparameters
- batch_size: (16, 16)
- num_epochs: (5, 5)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 500
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
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0025 | 1    | 0.1168        | -               |
| 0.1229 | 50   | 0.1239        | -               |
| 0.2457 | 100  | 0.0285        | -               |
| 0.3686 | 150  | 0.0035        | -               |
| 0.4914 | 200  | 0.0015        | -               |
| 0.6143 | 250  | 0.0009        | -               |
| 0.7371 | 300  | 0.0006        | -               |
| 0.8600 | 350  | 0.0005        | -               |
| 0.9828 | 400  | 0.0004        | -               |
| 1.1057 | 450  | 0.0003        | -               |
| 1.2285 | 500  | 0.0003        | -               |
| 1.3514 | 550  | 0.0003        | -               |
| 1.4742 | 600  | 0.0002        | -               |
| 1.5971 | 650  | 0.0002        | -               |
| 1.7199 | 700  | 0.0002        | -               |
| 1.8428 | 750  | 0.0002        | -               |
| 1.9656 | 800  | 0.0002        | -               |
| 2.0885 | 850  | 0.0002        | -               |
| 2.2113 | 900  | 0.0001        | -               |
| 2.3342 | 950  | 0.0001        | -               |
| 2.4570 | 1000 | 0.0001        | -               |
| 2.5799 | 1050 | 0.0001        | -               |
| 2.7027 | 1100 | 0.0001        | -               |
| 2.8256 | 1150 | 0.0001        | -               |
| 2.9484 | 1200 | 0.0001        | -               |
| 3.0713 | 1250 | 0.0001        | -               |
| 3.1941 | 1300 | 0.0001        | -               |
| 3.3170 | 1350 | 0.0001        | -               |
| 3.4398 | 1400 | 0.0001        | -               |
| 3.5627 | 1450 | 0.0001        | -               |
| 3.6855 | 1500 | 0.0001        | -               |
| 3.8084 | 1550 | 0.0001        | -               |
| 3.9312 | 1600 | 0.0001        | -               |
| 4.0541 | 1650 | 0.0001        | -               |
| 4.1769 | 1700 | 0.0001        | -               |
| 4.2998 | 1750 | 0.0001        | -               |
| 4.4226 | 1800 | 0.0001        | -               |
| 4.5455 | 1850 | 0.0001        | -               |
| 4.6683 | 1900 | 0.0001        | -               |
| 4.7912 | 1950 | 0.0001        | -               |
| 4.9140 | 2000 | 0.0001        | -               |

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