## Mindful: NLP-based Journal Summarization and Sentiment Analysis

Our machine learning system integrates directly into existing psychological counseling practices. Psychologists regularly ask patients to maintain therapeutic journals, manually reviewing these entries to detect trends, emotions, and possible risk factors. This manual process is labor-intensive, subjective, and difficult to scale. Our proposed ML assistant will automatically summarize journal entries, perform sentiment and emotional analysis, and highlight important or concerning themes. This allows therapists to quickly identify areas requiring attention, track patient progress objectively, and increase the number of patients they can effectively support.

### Non-ML Status Quo:
#### Currently, psychologists manually review patient journal entries, which is time-consuming, subjective, and prone to missing subtle emotional patterns or risks.

### Business Metrics:
* Reduction in time spent reviewing patient journal entries.
* Improved accuracy in detecting emotional patterns and therapeutic risks.
* Psychologist-reported satisfaction and usability scores.

### Contributors

<!-- Table of contributors and their roles. 
First row: define responsibilities that are shared by the team. 
Then, each row after that is: name of contributor, their role, and in the third column, 
you will link to their contributions. If your project involves multiple repos, you will 
link to their contributions in all repos here. -->

| Name                            | Responsible for| Link to their commits in this repo |
|---------------------------------|-----------------|------------------------------------|
| All team members                |Idea formulation, value proposition, ML problem setup, integration|     NA                               |
| Arnab Bhowal                   |Model training (Units 4 & 5)|        https://github.com/Sanjeevan1998/mindful/commits/main/?author=arnabbhowal                            |
| Sanjeevan Adhikari                   |Model serving (Unit 6) and monitoring (Unit 7)|             https://github.com/Sanjeevan1998/mindful/commits/main/?author=Sanjeevan1998                       |
| Divya Chougule                   |Data pipeline (Unit 8)|                 https://github.com/Sanjeevan1998/mindful/commits/main/?author=divyaa25                   |
| Vishwas Karale                   |CICD pipeline|                   https://github.com/Sanjeevan1998/mindful/commits/main/?author=vishwaskarale83                 |



### System diagram

![System Diagram](https://raw.githubusercontent.com/Sanjeevan1998/mindful/refs/heads/main/SystemDiagram.jpeg)



### Summary of outside materials

<!-- In a table, a row for each dataset, foundation model. 
Name of data/model, conditions under which it was created (ideally with links/references), 
conditions under which it may be used. -->

# Mental Health Analysis Datasets and Foundation Models

## Datasets for Mental Health Analysis

| Name | How it was created | Conditions of use |  
|------------------------------------|--------------------|-------------------|  
| [**Sentiment Analysis Dataset**](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset) | Labeled dataset for sentiment analysis tasks, useful for training models on positive/negative sentiment detection | Available for research and commercial use under Kaggle's dataset terms |  
| [**Sentiment Analysis for Mental Health**](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health) | Focuses on sentiment classification specific to mental health discussions | Available for research purposes, subject to Kaggle's data use policies |  
| [**Suicidal Tweet Detection Dataset**](https://www.kaggle.com/datasets/aunanya875/suicidal-tweet-detection-dataset) | Tweets labeled based on suicidal intent, useful for NLP-based risk detection | Requires ethical considerations and adherence to platform policies |  
| [**Suicidal Mental Health Dataset**](https://www.kaggle.com/datasets/aradhakkandhari/suicidal-mental-health-dataset) | Text dataset focused on mental health and suicide risk classification | Available for academic and research purposes |  
| [**Students Anxiety and Depression Dataset**](https://www.kaggle.com/datasets/sahasourav17/students-anxiety-and-depression-dataset) | Survey-based dataset on student mental health indicators | Available for research, subject to dataset creators’ conditions |  
| [**Reddit Mental Health Data**](https://www.kaggle.com/datasets/neelghoshal/reddit-mental-health-data) | Collected from mental health-related subreddits, covering topics like depression and anxiety | Available for research, subject to Reddit’s data use policies |  
| [**Mental Health Dataset (Bipolar)**](https://www.kaggle.com/datasets/michellevp/mental-health-dataset-bipolar) | Data on individuals with bipolar disorder, curated for mental health analysis | Available for research purposes, ethical considerations required |  
| [**Human Stress Prediction**](https://www.kaggle.com/datasets/kreeshrajani/human-stress-prediction) | Dataset for stress prediction based on various mental health indicators | Available for research and analysis |  
| [**Predicting Anxiety in Mental Health Data**](https://www.kaggle.com/datasets/michellevp/predicting-anxiety-in-mental-health-data) | Dataset designed for anxiety prediction in mental health patients | Available for research under dataset-specific terms |  
| [**Depression Reddit Cleaned**](https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned) | Preprocessed dataset for studying depression discussions on Reddit | Available for research, subject to Reddit's data policies |  
| [**Sentiment140**](https://www.kaggle.com/datasets/kazanova/sentiment140) | Large-scale sentiment analysis dataset based on Twitter data | Publicly available for sentiment analysis research |  
| [**DAIC-WOZ (Distress Analysis Interview Corpus)**](https://dcapswoz.ict.usc.edu/) | Contains clinical interviews designed to assess mental distress using speech and text data | Requires application and approval for access, focused on ethical research |  
| [**Yelp Review Full**](https://huggingface.co/datasets/Yelp/yelp_review_full) | Yelp reviews dataset labeled for sentiment analysis | Publicly available for NLP tasks |  
| [**TweetEval**](https://huggingface.co/datasets/cardiffnlp/tweet_eval) | Benchmark dataset for sentiment and emotion classification on Twitter data | Available for research under dataset license |  
| [**Sentiment Analysis for Mental Health (Hugging Face)**](https://huggingface.co/datasets/btwitssayan/sentiment-analysis-for-mental-health) | NLP dataset focused on sentiment classification for mental health discussions | Available for research under dataset-specific terms |  
| [**Sentiment Analysis Preprocessed Dataset**](https://huggingface.co/datasets/prasadsawant7/sentiment_analysis_preprocessed_dataset) | Preprocessed dataset for sentiment analysis | Available for research and analysis |  
| [**Social Media Sentiments Analysis Dataset**](https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset) | Analyzed social media posts for sentiment trends | Available for research and sentiment analysis tasks |  

## Foundation Models

| Name      | How it was created | Conditions of use |
|----------|--------------------|-------------------|
| [**BART**](https://huggingface.co/docs/transformers/v4.46.0/model_doc/bart)      | Sequence-to-sequence model with absolute position embeddings, designed for text generation and summarization | MIT License - allows free use, modification, and distribution with inclusion of original license and copyright notice |
| [**RoBERTa**](https://huggingface.co/docs/transformers/en/model_doc/roberta)   | Robustly optimized BERT approach, trained with more data and longer sequences than BERT | MIT License - similar to BART, allowing wide usage and modification |
| [**DistilBERT**](https://huggingface.co/docs/transformers/en/model_doc/distilbert) | Distilled version of BERT, combining language modeling, distillation, and cosine-distance losses | Apache 2.0 License - permits use, distribution, and modification with license copy inclusion |



### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), 
how much/when, justification. Include compute, floating IPs, persistent storage. 
The table below shows an example, it is not a recommendation. -->

## Resource Requirements  

| Requirement         | Quantity/Duration         | Justification  |
|---------------------|-------------------------|---------------|
| **m1.medium VMs**  | 3 for entire project duration | These general-purpose VMs will host Kubernetes clusters, run data pipelines (Kafka, Airflow), serve APIs (FastAPI), and manage CI/CD pipelines (GitHub Actions, ArgoCD). They provide sufficient CPU and memory resources for these workloads without unnecessary cost. |
| **Any available GPU** | 2 GPUs, 8-hour blocks, twice weekly | GPUs are required for fine-tuning transformer-based NLP models (BART, RoBERTa, DistilBERT). Transformer models are GPU-intensive, and distributed training (Ray Train) and hyperparameter tuning (Ray Tune) require GPUs with ample memory and compute power. |
| **Persistent Storage** | 200 GB for entire project duration | Persistent storage is necessary to securely store raw and processed datasets, trained model artifacts, container images, and experiment tracking data (MLFlow). This ensures data persistence, reproducibility, and easy access across different stages of the pipeline. |
| **Floating IPs** | 2 for entire project duration | One floating IP is required for public access to the model-serving API endpoints and psychologist dashboard. The second floating IP is needed for accessing internal services such as MLFlow experiment tracking and monitoring dashboards (Prometheus/Grafana). |


### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the 
diagram, (3) justification for your strategy, (4) relate back to lecture material, 
(5) include specific numbers. -->

#### Model training and training platforms


We will fine-tune three pre-trained transformer models to build our ML assistant for therapists:
- **Summarization Model**: BART-base (for summarizing patient journal entries)
- **Sentiment Analysis Model**: RoBERTa-base (for detecting emotions and sentiment)
- **Risk Detection Model**: DistilBERT-base (for identifying concerning or risky content)

We will perform initial fine-tuning of these models using publicly available mental health datasets (e.g., Mental Health Reddit Dataset, GoEmotions). After initial training, we will periodically retrain these models using updated labeled data collected from simulated production usage.

To efficiently train these transformer models, we will use distributed training strategies (Ray Train with Distributed Data Parallel - DDP) on GPU nodes (RTX6000 GPUs). Additionally, we will perform hyperparameter tuning using Ray Tune to optimize model performance. All experiments, hyperparameters, metrics, and model artifacts will be tracked using MLFlow, hosted on Chameleon infrastructure.

### Relevant parts of the diagram:
- **Kubernetes Cluster → Model Training & Experiment Tracking:**
  - Ray Cluster (Distributed Training & Hyperparameter Tuning)
  - MLFlow Server (Experiment Tracking)
  - GPU Nodes (RTX6000 GPUs)

### Justification for our strategy:
- **Transformer Models**: BART, RoBERTa, and DistilBERT are state-of-the-art NLP models widely recognized for their effectiveness in summarization, sentiment analysis, and text classification tasks, respectively. Fine-tuning these models on domain-specific mental health data ensures high accuracy and relevance for therapists.
- **Distributed Training (Ray Train with DDP)**: Transformer models are computationally intensive and require significant GPU resources. Distributed training allows us to significantly reduce training time, enabling faster experimentation and iteration.
- **Hyperparameter Tuning (Ray Tune)**: Transformer models' performance heavily depends on hyperparameters (learning rate, batch size, epochs, etc.). Ray Tune efficiently explores hyperparameter spaces, ensuring optimal model performance.
- **Experiment Tracking (MLFlow)**: MLFlow provides reproducibility, experiment management, and easy comparison of model performance across experiments, essential for systematic model development.

### Relation back to lecture material (Units 4 & 5):

#### Unit 4 (Model training at scale):
- We satisfy the requirement to train and retrain at least one model (we train three).
- We make reasonable, appropriate, and well-justified modeling choices (transformer models fine-tuned on relevant datasets).
- **Difficulty point attempted**: Distributed training (Ray Train with DDP) to increase training velocity. We will conduct experiments comparing training times using one GPU vs. two GPUs, clearly documenting speedups and efficiency gains.

#### Unit 5 (Model training infrastructure and platform):
- We satisfy the requirement to host an experiment tracking server (MLFlow) on Chameleon and instrument our training code with logging calls.
- We satisfy the requirement to run a Ray cluster and submit training jobs as part of our continuous training pipeline.
- **Difficulty point attempted**: Hyperparameter tuning using Ray Tune, leveraging advanced tuning algorithms (e.g., Bayesian optimization, HyperBand) to efficiently optimize model performance.

### Specific numbers and details:
- **GPU Resources**: 2 RTX6000 GPUs (24GB VRAM each), scheduled in 8-hour blocks, twice weekly.
- **Models**: 3 transformer models (BART-base, RoBERTa-base, DistilBERT-base).
- **Distributed Training**: Ray Train with Distributed Data Parallel (DDP), comparing training times with 1 GPU vs. 2 GPUs.
- **Hyperparameter Tuning**: Ray Tune with Bayesian optimization or HyperBand, exploring at least 10-20 hyperparameter configurations per model.
- **Experiment Tracking**: MLFlow server hosted on Chameleon, logging all experiments, hyperparameters, metrics, and artifacts.

### Summary of Difficulty Points Attempted (Units 4 & 5):

| Unit | Difficulty Point Selected      | Explanation |
|------|--------------------------------|-------------|
| 4    | Distributed training (Ray Train with DDP) | We will demonstrate training speedups by comparing single-GPU vs. multi-GPU training times. |
| 5    | Hyperparameter tuning (Ray Tune) | We will use advanced tuning algorithms to efficiently optimize model performance. |



<!-- Make sure to clarify how you will satisfy the Unit 4 and Unit 5 requirements, 
and which optional "difficulty" points you are attempting. -->



#### Model serving and monitoring platforms

<!-- Make sure to clarify how you will satisfy the Unit 6 and Unit 7 requirements, 
and which optional "difficulty" points you are attempting. -->

#### Data pipeline

<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which 
optional "difficulty" points you are attempting. -->

#### Continuous X

<!-- Make sure to clarify how you will satisfy the Unit 3 requirements,  and which 
optional "difficulty" points you are attempting. -->


