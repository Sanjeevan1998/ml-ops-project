## Mindful: NLP-based Journal Summarization and Sentiment Analysis

Our machine learning system integrates directly into existing psychological counseling practices. Psychologists regularly ask patients to maintain therapeutic journals, manually reviewing these entries to detect trends, emotions, and possible risk factors. This manual process is labor-intensive, subjective, and difficult to scale. Our proposed ML assistant will automatically summarize journal entries, perform sentiment and emotional analysis, and highlight important or concerning themes. This allows therapists to quickly identify areas requiring attention, track patient progress objectively, and increase the number of patients they can effectively support.


### Non-ML Status Quo:

#### Currently, psychologists manually review patient journal entries, which is:
* Time-consuming: Reviewing and analyzing large volumes of text takes significant time and effort.
* Subjective: Interpretations of emotional patterns and risks can vary between psychologists, leading to inconsistencies.
* Prone to oversight: Subtle emotional cues or trends may be missed, especially when dealing with extensive or complex journal entries.
* Difficult to scale: As the number of patients increases, the manual process becomes unsustainable, limiting the number of patients a psychologist can effectively support.

### Business Metrics:

#### The success of the "Mindful" system will be judged on the following business metrics:
* Reduction in Time Spent Reviewing Journals: The system should significantly decrease the time psychologists spend on manual journal reviews.
* Improved Detection Accuracy: The system's ability to identify emotional patterns, risks, and concerning themes should outperform manual reviews.
* Psychologist Satisfaction and Usability: Feedback from psychologists on the system's ease of use, reliability, and effectiveness will be critical.
* Patient Outcomes: Indirectly, the system's impact on patient progress and therapeutic success can also serve as a measure of its value.


### Value Proposition

#### Who is the customer?
The primary customers for the "Mindful" system are psychologists, therapists, and mental health professionals working in private practices, counseling centers, or hospitals. For example, a specific business that could benefit from this system is BetterHelp, an online therapy platform that connects patients with licensed therapists. BetterHelp therapists often rely on written communication (e.g., patient journals or messages) to understand their clients' emotional states, making this system a perfect fit for their workflow.

#### Where will it be used?

The system will be used in psychological counseling practices or online therapy platforms like BetterHelp or Talkspace. It can also be integrated into hospital mental health departments or university counseling centers, where therapists manage large caseloads and need tools to streamline their workflows.


#### Benefits of our system over the current system:
* Time Savings: Automates the labor-intensive process of manually reviewing journal entries, allowing psychologists to focus on patient care.
* Improved Accuracy: Detects subtle emotional patterns and therapeutic risks that may be missed in manual reviews, ensuring better patient outcomes.
* Scalability: Enables psychologists to support more patients effectively by reducing the time spent on administrative tasks.
* Objective Insights: Provides consistent and unbiased analysis, reducing subjectivity in interpreting journal entries.
* Confidence Scores: The system provides confidence scores for each inference (e.g., "This entry is flagged with 85% confidence"), helping psychologists make informed decisions.



### Privacy, Fairness, and Ethics Concerns
#### Privacy Concerns:
##### Sensitive Data: Patient journals contain highly sensitive personal information. Ensuring data privacy is paramount.
##### Preventive Measures:
* All data will be anonymized and encrypted during processing and storage.
* Personally identifiable information (PII) such as names, locations, and other sensitive details will be removed during preprocessing.


#### Fairness Concerns:
##### Bias in Sentiment and Emotion Analysis: The system should avoid biases that could arise from training data, such as cultural or linguistic differences in journal entries leading to inaccurate or unfair assessments.
##### Preventive Measures:
* Use diverse and representative datasets during training to minimize bias.
* Regularly evaluate the system for fairness using fairness metrics and domain-specific tests.
* Allow psychologists to provide feedback on flagged entries to improve the system over time.


#### Ethical Concerns:
##### Incorrect Flagging: There is a risk that the system may incorrectly flag or fail to flag critical emotional patterns or risks.
##### Preventive Measures:
* The system will provide confidence scores for each inference, indicating how certain it is about its predictions (e.g., "This entry is flagged with 85% confidence").
* Psychologists will retain full control over the final interpretation of the analysis, ensuring that the system acts as a tool to assist, not replace, human judgment.
* Regular retraining and evaluation of the models will ensure continuous improvement in accuracy and reliability.

#### Legal Use of Data:
#### Informed Consent: Patients must provide informed consent for their journal entries to be analyzed by the system.
#### Preventive Measures:
* Ensure that all data is collected and used under appropriate legal agreements and licenses.
* Clearly communicate to patients how their data will be used and the benefits of the system.


### Contributors

<!-- Table of contributors and their roles. 
First row: define responsibilities that are shared by the team. 
Then, each row after that is: name of contributor, their role, and in the third column, 
you will link to their contributions. If your project involves multiple repos, you will 
link to their contributions in all repos here. -->

| Name                            | Responsible for| Link to their commits in this repo |
|---------------------------------|-----------------|------------------------------------|
| All team members                |Idea formulation, value proposition, ML problem setup, integration|     NA                               |
| Arnab Bhowal                   |Model Training|        https://github.com/Sanjeevan1998/mindful/commits/main/?author=arnabbhowal                            |
| Sanjeevan Adhikari                   |Model Serving and Monitoring|             https://github.com/Sanjeevan1998/mindful/commits/main/?author=Sanjeevan1998                       |
| Divya Chougule                   |Data Pipeline (Unit 8)|                 https://github.com/Sanjeevan1998/mindful/commits/main/?author=divyaa25                   |
| Vishwas Karale                   |Continous X Pipeline|                   https://github.com/Sanjeevan1998/mindful/commits/main/?author=vishwaskarale83                 |



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
| [**GoEmotions**](https://www.kaggle.com/datasets/debarshichanda/goemotions) | GoEmotions is a corpus of 58k carefully curated comments extracted from Reddit, with human annotations to 27 emotion categories or Neutral. | Available for research and commercial use under Kaggle's dataset terms |
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
| **Persistent Storage** | 65-70 GB for entire project duration | Persistent storage is necessary to securely store raw and processed datasets, trained model artifacts, container images, and experiment tracking data (MLFlow). This ensures data persistence, reproducibility, and easy access across different stages of the pipeline. |
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

To efficiently train these transformer models, we will use distributed training strategies (Ray Train with Distributed Data Parallel - DDP) on GPU nodes. Additionally, we will perform hyperparameter tuning using Ray Tune to optimize model performance. All experiments, hyperparameters, metrics, and model artifacts will be tracked using MLFlow, hosted on Chameleon infrastructure.

### Relevant parts of the diagram:
- **Kubernetes Cluster → Model Training & Experiment Tracking:**
  - Ray Cluster (Distributed Training & Hyperparameter Tuning)
  - MLFlow Server (Experiment Tracking)
  - GPU Nodes

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
- **GPU Resources**: 2 GPUs, scheduled in 8-hour blocks, twice weekly.
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

We will deploy our fine-tuned models as APIs using FastAPI, which is known for its speed and ease of use. The models will be containerized using Docker and orchestrated with Kubernetes on Chameleon Cloud. This setup allows for efficient scaling and management of our model-serving infrastructure.

To ensure optimal performance, we will implement model-level optimizations such as quantization and ONNX conversion, which reduce model size and improve inference speed. We will also compare CPU and GPU serving options to determine the most cost-effective and performant solution for our application.

For monitoring, we will utilize Prometheus for collecting metrics and Grafana for visualizing these metrics in real-time. This monitoring setup will help us track model performance, latency, and resource utilization. Additionally, we will implement automated load testing to simulate user interactions and evaluate the system's performance under different loads.

To address potential issues with model performance, we will set up data and model drift monitoring. If significant drift is detected, our system will trigger automatic retraining of the models to maintain accuracy and relevance.

### Relevant parts of the diagram:
- **Kubernetes Cluster → Model Serving API (FastAPI, Models)**
- **Kubernetes Cluster → Monitoring & Evaluation (Prometheus, Grafana, Load Tests, Drift Monitoring)**
- **Staged Deployment (Staging → Canary → Production)**

### Justification for our strategy:
- **FastAPI** is an excellent choice for serving machine learning models due to its high performance and ease of integration with Python-based ML frameworks. It allows us to create RESTful APIs quickly, enabling therapists to access model predictions seamlessly.
- **Docker** ensures that our models and their dependencies are packaged consistently, making deployment straightforward across different environments. This containerization also simplifies scaling and management within Kubernetes.
- **Kubernetes** provides robust orchestration capabilities, allowing us to manage multiple instances of our model-serving APIs efficiently. It also supports automatic scaling based on demand, ensuring that we can handle varying loads effectively.
- **Prometheus** and **Grafana** are industry-standard tools for monitoring and visualization. They allow us to track key performance metrics, ensuring that we can respond quickly to any issues that arise in production.
- Implementing **automated load testing** and **drift monitoring** ensures that our system remains reliable and accurate over time, addressing potential performance degradation proactively.

### Relation back to lecture material (Units 6 & 7):

#### Unit 6 (Model serving):
- We satisfy the requirement to serve models via APIs using FastAPI, ensuring low-latency access for therapists.
- We will compare CPU vs. GPU serving options to optimize performance and cost.
- **Difficulty point attempted**: Develop multiple serving options (CPU vs. GPU).

#### Unit 7 (Monitoring):
- We satisfy the requirement for monitoring model performance and resource utilization using Prometheus and Grafana.
- We will implement automated load tests to evaluate system performance under various conditions.
- We will monitor for data and model drift, triggering automatic retraining when necessary.
- **Difficulty point attempted**: Monitor data drift/model degradation and implement automatic retraining.

### Specific numbers and details:
- **Serving Requirements**:
  - **Latency requirement**: <100ms per inference
  - **Concurrency requirement**: ~50 simultaneous requests
- **Monitoring Metrics**:
  - Track latency, throughput, and resource utilization (CPU/GPU usage).
- **Load Testing**:
  - Simulate realistic user interactions with at least 100 concurrent users during load tests.

### Summary of Difficulty Points Attempted (Units 6 & 7):

| Unit | Difficulty Point Selected      | Explanation |
|------|--------------------------------|-------------|
| 6    | CPU vs. GPU serving comparison | We will evaluate performance and cost-effectiveness of serving models on CPU vs. GPU. |
| 7    | Data drift/model degradation monitoring & automatic retraining | We will implement monitoring to detect drift and trigger retraining to maintain model accuracy. |



#### Data pipeline

<!-- Make sure to clarify how you will satisfy the Unit 8 requirements,  and which 
optional "difficulty" points you are attempting. -->


We will implement two types of data pipelines to handle the ingestion, processing, and storage of data for our ML Assistant for Therapists:

**Offline ETL Pipeline (Batch Processing)**:
- Built using **Apache Airflow**, this pipeline will handle batch processing of raw datasets (e.g., Mental Health Reddit Dataset, GoEmotions).
- The pipeline will extract, clean, preprocess, and transform the data into a format suitable for training our models.
- Processed datasets will be stored in persistent storage (200GB) on **Chameleon Cloud** for use in model training and retraining.

**Online Streaming Pipeline (Real-Time Processing)**:
- Built using **Kafka** and **Zookeeper**, this pipeline will simulate real-time patient journal entries being ingested into the system.
- The streaming data will be used for inference (model predictions) and monitoring (e.g., detecting data drift or anomalies).
- The pipeline will also store incoming data in persistent storage for further analysis and retraining.

### Relevant parts of the diagram:
- **Kubernetes Cluster → Data Pipelines**:
  - **Kafka/Zookeeper** (Online streaming pipeline)
  - **Apache Airflow** (Offline ETL pipeline)
- **Persistent Storage (200GB)**:
  - Stores raw and processed datasets, model artifacts, and inference data.

### Justification for our strategy:
- **Offline ETL Pipeline**: Batch processing is ideal for preparing large datasets for training and retraining. **Apache Airflow** is a widely-used workflow orchestration tool that allows us to define, schedule, and monitor ETL tasks efficiently. It ensures that our data is clean, consistent, and ready for use in model training.
- **Online Streaming Pipeline**: Real-time data ingestion is critical for simulating patient journal entries and testing our system's inference and monitoring capabilities. **Kafka** is a robust, scalable, and fault-tolerant platform for real-time data streaming, making it an excellent choice for this purpose.
- **Persistent Storage**: Ensures that all raw and processed data, as well as model artifacts, are securely stored and easily accessible for training, inference, and monitoring.

### Relation back to lecture material (Unit 8):
- **Persistent Storage**: We satisfy the requirement to provision persistent storage for datasets, models, and artifacts.
- **Offline Data Pipeline**: We satisfy the requirement to set up an ETL pipeline for batch processing of training data.
- **Online Data Pipeline**: We satisfy the requirement to simulate realistic online data streams for inference and monitoring.

### Specific numbers and details:
- **Persistent Storage**: 200GB on Chameleon Cloud.
- **Offline Pipeline**: Processes datasets (e.g., Mental Health Reddit Dataset, GoEmotions) with approximately 10,000-50,000 entries.
- **Online Pipeline**: Simulates real-time journal entries at a rate of ~100 entries per hour.

### Optional Difficulty Point Attempted:
- **Interactive Data Dashboard**: We will implement an interactive dashboard (using **Grafana**) to provide insights into data quality, such as missing values, outliers, and distribution changes over time. This dashboard will help us monitor and improve the quality of our training and inference data.



### Continuous X

<!-- Make sure to clarify how you will satisfy the Unit 3 requirements,  and which 
optional "difficulty" points you are attempting. -->

To ensure the smooth deployment procedure for updates, patch fixes for robustness, reliability, reproducibility, and scalability of our ML Assistant for Therapists, we will implement a robust Continuous X (CI/CD) pipeline using standard tools and practices.

**Infrastructure-as-Code**:
- We will use **Terraform** to define and provision our infrastructure on **Chameleon Cloud**, including the Kubernetes cluster, persistent storage, and networking resources.
- Additionally, we will use **Ansible** to automate the configuration and setup of these infrastructure components, ensuring a consistent and reliable environment.

**Automation and CI/CD**:
- Our CI/CD pipeline will be implemented using **GitHub Actions** and **ArgoCD**. Whenever changes are made to our codebase (e.g., model updates, API changes, infrastructure modifications), the CI/CD pipeline will be triggered to:
  - Automatically build and test the changes.
  - Package the application components (models, APIs, data pipelines) into **Docker containers**.
  - Deploy the containerized components to our **Kubernetes cluster** in a staged manner (**staging → canary → production**).

**Staged Deployment**:
- We will implement a staged deployment strategy to ensure the reliability and safety of our production environment. New model versions, API updates, and infrastructure changes will first be deployed to a **staging** environment for initial testing and evaluation. Once the changes have been validated, they will be gradually rolled out to a **canary** environment for further testing with simulated user interactions. Finally, the changes will be promoted to the **production** environment after successful canary evaluation.

**Cloud-Native Principles**:
- Throughout our Continuous X pipeline, we will adhere to cloud-native principles, such as:
  - **Immutable Infrastructure**: Infrastructure components will be defined as code and provisioned in a repeatable manner, ensuring consistency and reliability.
  - **Microservices Architecture**: Our application will be designed as a collection of loosely coupled, independently deployable services (e.g., model serving, data pipelines, monitoring).
  - **Containerization**: All application components will be packaged as **Docker containers**, enabling consistent deployment across different environments.

### Justification for our strategy:
- **Infrastructure-as-Code (Terraform, Ansible)**: Defining infrastructure as code ensures consistency, reproducibility, and easy management of our Chameleon Cloud resources.
- **Automation and CI/CD (GitHub Actions, ArgoCD)**: Automating the build, test, and deployment processes reduces manual effort, minimizes errors, and enables rapid iteration.
- **Staged Deployment**: Gradually rolling out changes to different environments (staging, canary, production) helps us identify and mitigate issues before impacting end-users.
- **Cloud-Native Principles**: These principles improve the scalability, reliability, and maintainability of our application, aligning with industry best practices.

### Relation back to lecture material (Unit 3):
- **Infrastructure-as-Code**: We satisfy the requirement to use **Terraform** and **Ansible** for infrastructure provisioning and configuration.
- **Automation and CI/CD**: We satisfy the requirement to implement a CI/CD pipeline using **GitHub Actions** and **ArgoCD**.
- **Staged Deployment**: We satisfy the requirement to implement a staged deployment strategy (**staging → canary → production**).
- **Cloud-Native Principles**: We satisfy the requirement to develop our application using cloud-native principles.

### Specific numbers and details:
- **Infrastructure-as-Code**: We will define all **Chameleon Cloud** resources (VMs, storage, networking) using **Terraform** scripts.
- **Automation and CI/CD**: **GitHub Actions** will be triggered on code commits, running tests and building **Docker** images. **ArgoCD** will then deploy the new versions to our **Kubernetes cluster**.
- **Staged Deployment**: We will have three distinct environments (**staging, canary, production**) for gradual rollout of changes.
- **Cloud-Native Principles**: Our application will be designed as a collection of **Docker containers**, with each component (models, APIs, data pipelines) deployed as a separate **microservice**.


