## LegalAI: NLP-based Case Law Identification and Summarization Assistant

Our machine learning system integrates directly into existing legal research and case preparation workflows. Currently, lawyers and paralegals manually review extensive legal documents, case transcripts, and previous rulings to identify relevant precedents, arguments, and key details. This manual review process is labor-intensive and time-consuming. 

Our proposed AI assistant, LegalAI, will function as a digital paralegal, enabling legal professionals to upload details and transcripts of their current cases. Leveraging advanced NLP techniques and large language models, the system automatically identifies, and extracts essential information from relevant case laws. The tool will generate concise summaries and provides essential metadata, allowing lawyers and paralegals to rapidly review pertinent case details with direct links to access comprehensive documents online.

This approach not only streamlines and accelerates legal research but significantly enhances objectivity and accuracy in identifying relevant precedents, allowing legal teams to handle a greater volume of cases effectively and strategically.


### Non-ML Status Quo:

#### Currently, lawyers and paralegals manually review extensive legal documents and case transcripts, which is:
* Time-consuming: Reviewing extensive legal documents, transcripts, and prior rulings demands substantial time, limiting productivity.
* Prone to human error: Critical precedents, arguments, or subtle legal nuances can easily be overlooked due to fatigue or oversight.
* Inefficient: Manual searching through vast legal resources is inefficient, particularly when rapid response and thorough analysis are required.
* Difficult to scale: As the volume of cases grows, manually managing comprehensive reviews becomes unsustainable, restricting the volume of cases a legal professional can effectively handle.

### Business Metrics:

#### The success of the LegalAI system will be evaluated using the following business metrics:
* Reduction in Time Spent Reviewing Case Documents: Currently, legal professionals typically spend several hours per case manually reviewing relevant documents and case laws. With LegalAI, automated summarization and extraction reduce review time significantly—potentially from hours down to minutes—freeing up valuable time for strategic preparation.
* Improved Identification Accuracy: The system’s capability to pinpoint relevant precedents, critical arguments, and essential metadata should consistently surpass manual review accuracy.
* User Satisfaction and Usability: Feedback from lawyers and paralegals regarding system intuitiveness, reliability, and practical utility will serve as a critical measure of success.
* Enhanced Case Outcomes: Indirectly, the system’s impact on improved case preparation and positive legal outcomes will serve as a core indicator of its effectiveness.

### Value Proposition

#### Who is the customer?
The primary customers for the LegalAI system are lawyers, paralegals, and legal research professionals working in law firms, in-house legal departments, judicial systems, and legal consulting services. For instance, law firms handling complex litigation cases with extensive documentation, such as corporate legal teams and legal research firms (e.g., LexisNexis, Westlaw), would significantly benefit from the efficiency and precision offered by the system.

#### Where will it be used?

The LegalAI assistant will be deployed in law firms, corporate legal departments, court systems, and academic institutions that engage heavily in legal research. The system can also be integrated with online legal research databases and platforms such as LexisNexis, Westlaw, or public judiciary databases, enhancing their existing search and summarization capabilities.


#### Benefits of our system over the current system:
* Time Savings: Automates the manual and labor-intensive task of reviewing legal documents, allowing legal professionals to dedicate more time to strategic case preparation.
* Improved Accuracy: Detects critical legal precedents and arguments that might otherwise be overlooked, ensuring more thorough and precise legal analyses.
* Scalability: Allows law firms and legal professionals to efficiently handle a larger number of cases by significantly reducing time and effort spent on administrative and review tasks.
* Objective Insights: Delivers consistent, unbiased summaries and metadata extraction, reducing variability inherent in manual analysis.
* Confidence Scores: Recommendation or summarized precedent is accompanied by a confidence score if its below certain threshold (e.g., "This precedent matches your case context with 72% confidence"), supporting informed decision-making.


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

![System Diagram](https://raw.githubusercontent.com/Sanjeevan1998/mindful/refs/heads/main/LegalAI_system_diagram.png)

### Summary of outside materials

<!-- In a table, a row for each dataset, foundation model. 
Name of data/model, conditions under which it was created (ideally with links/references), 
conditions under which it may be used. -->

# Legal Case Analysis Datasets and Foundation Models

## Datasets for Legal Case Analysis 

| Name | How it was created | Conditions of use |  
|------------------------------------|--------------------|-------------------|  
| [**Lexis Nexis Dataset**](https://www.lexisnexis.com/) | LexisNexis offers extensive legal databases with case law, statutes, regulations, and other legal documents. They also provide access to news articles and other legal resources |  

## Foundation Models

| Name      | How it was created | Conditions of use |
|----------|--------------------|-------------------|
| [**Legal-BERT**](https://huggingface.co/nlpaueb/legal-bert-base-uncased?utm_source=chatgpt.com)     | Fine-tuned on legal corpora (e.g., contracts, court decisions) for legal language understanding and metadata tagging. | Available for academic research; non-commercial use advised |

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), 
how much/when, justification. Include compute, floating IPs, persistent storage. 
The table below shows an example, it is not a recommendation. -->


## Deployment and Execution

The Product is a Cloud Native application which can be bootstapped in multiple phases. The Development & Operations Lifecycle of the Product is distributed as follows. The Project can be launched on Trovi as [Legal AI - NLP CaseLaw Identifier](https://chameleoncloud.org/experiment/share/a49b98c4-fb07-41f0-a045-dd68735b7dc3)

- **Boostrapping Phase** : In this Phase we will be setting up the initial Infrastructure which is defined as IaaC using Terraform Scripts and Ansible Playbooks
- **Development Phase** : In this Phase we train the initial Model and also we evaluate our base performance we use GPU Machines and train Models using Ray Clusters.
- **Runtime Phase** : In Runtime we Monitor the Performance of our Model where we Check for the Performance of the Model also monitor the model for Data drift
- **Update Phase** : Once in the Runtime Phase if we observe issues or have bugs with our model or need to add new features we start back in the retraining phase where we make modifications to the existing model and then take the model till Production following a Staged Deployment
- **End Phase** : In End Phase we shut down all our systems gracefully ensuring all consumed Chameleon Cloud Resources are released


### Development and Operations Flow

| Stage | Notebook Pipelines | Operations Executed |
|-------|--------------------|---------------------|
| Bootstrapping | [0_create_persistent_storage.ipnyb](https://github.com/Sanjeevan1998/ml-ops-project/blob/main/notebooks/bootstrap/0_create_persistent_storage.ipynb) | This Notebook will initially clone this Repo from Github which contains all the Code help in Setting up the Persistent Storage Block on CHI@UC which is used throughout using terraform |
| Bootstrapping | [1_key_setup.ipynb](https://github.com/Sanjeevan1998/ml-ops-project/blob/main/notebooks/bootstrap/1_key_setup.ipynb) | This will Setup the initial Key Pair with KVM@TACC, it is separated as it runs with Python Kernal and other run with Bash |
| Bootstrapping | [2_k8s_setup.ipynb](https://github.com/Sanjeevan1998/ml-ops-project/blob/main/notebooks/bootstrap/2_k8s_setup.ipynb) | This will create the intial Infrastructure on KVM@TACC which includes Compute Node, Floating IP, Attached Storage. After Node creation this notebook will further Setup K8S using Ansible and KubeSpray and all required tools (MinIO, MLFlow, Postgres, Grafana, Prometheus, ArgoCD, ArgoWorkflows) all are running on single K8S |
| Development | [1_create_initial_deployment.ipynb](https://github.com/Sanjeevan1998/ml-ops-project/blob/main/notebooks/development/1_create_initial_deployment.ipynb) | This will create the Argo Workflows needed for building, training, promoting model as well as will setup the Helm Charts for different Environments |
| End | [0_delete_persistent_storage.ipynb](https://github.com/Sanjeevan1998/ml-ops-project/blob/main/notebooks/delete/0_delete_persistent_storage.ipynb) | This will delete the Persistent Storage on CHI@UC |
| End | [1_delete_kvm_deployment.ipynb](https://github.com/Sanjeevan1998/ml-ops-project/blob/main/notebooks/delete/1_delete_kvm_resources.ipynb) | This will delete all the resources from KVM@TACC |



## Summary of infrastructure requirements

### Resource Requirements  

| Requirement         | Quantity/Duration         | Justification  |
|---------------------|-------------------------|---------------|
| **m1.xlarge VMs**  | 1 for entire project duration | These general-purpose VMs will host Kubernetes clusters, run data pipelines (Kafka, Airflow), serve APIs (FastAPI), and manage CI/CD pipelines (ArgoWorkflow, ArgoCD). They provide sufficient CPU and memory resources for these workloads without unnecessary cost. |
| **Any available GPU** | 2 GPUs (V100 or equivalent), 8-hour blocks, twice weekly | Required for fine-tuning Legal-BERT on large case law data. Also needed for distributed training via Ray. |
| **Persistent Storage** | 20GB for entire project duration, we will scale down after monitoring the disk usage | Persistent storage is necessary to securely store raw and processed datasets, trained model artifacts, container images, and experiment tracking data (MLFlow). This ensures data persistence, reproducibility, and easy access across different stages of the pipeline. |
| **Floating IPs** | 2 for entire project duration | One floating IP is required for public access to the model-serving API endpoints and monitoring dashboard. The second floating IP is needed for accessing internal services such as MLFlow experiment tracking and monitoring dashboards (Prometheus/Grafana). |


### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the 
diagram, (3) justification for your strategy, (4) relate back to lecture material, 
(5) include specific numbers. -->

#### Model training and training platforms


We will fine-tune and integrate pre-trained open-source model to build the LegalAI system:
- **Metadata Extractor**: Legal-BERT, used to extract structured fields such as involved parties, jurisdictions, legal principles, and outcomes from unstructured text.

Training and hyperparameter tuning will be done using Ray Train and Ray Tune respectively, on Chameleon Cloud GPU nodes. All training experiments, metrics, and models will be versioned and tracked using MLFlow.

### Relevant parts of the diagram:
- Kubernetes Cluster → Monitoring & Experiment Tracking
- Ray Cluster (Distributed Training & Hyperparameter Tuning)
- MLFlow Server (Experiment Tracking)
- GPU Nodes

### Justification for our strategy:
- **Models**: Legal-BERT brings domain-specific awareness, making it ideal for extracting metadata and understanding legal language.
- **Distributed Training (Ray Train with DDP)**: Transformer models are computationally intensive and require significant GPU resources. Distributed training allows us to significantly reduce training time, enabling faster experimentation and iteration.
- **Hyperparameter Tuning (Ray Tune)**: Legal text is complex and varies across jurisdictions. Hyperparameter optimization ensures model robustness across diverse document types.
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
- Processed datasets will be stored in persistent storage (65-70GB) on **Chameleon Cloud** for use in model training and retraining.

**Online Streaming Pipeline (Real-Time Processing)**:
- Built using **Kafka** and **Zookeeper**, this pipeline will simulate real-time patient journal entries being ingested into the system. We might use Kafka (or a lightweight queue like Redis Streams, if complexity is too high) to simulate real-time entry streaming.
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
- **Automation and CI/CD**: We satisfy the requirement to implement a CI/CD pipeline using **ArgoWorkflows** and **ArgoCD**.
- **Staged Deployment**: We satisfy the requirement to implement a staged deployment strategy (**staging → canary → production**).
- **Cloud-Native Principles**: We satisfy the requirement to develop our application using cloud-native principles.

### Specific numbers and details:
- **Infrastructure-as-Code**: We will define all **Chameleon Cloud** resources (VMs, storage, networking) using **Terraform** scripts.
- **Automation and CI/CD**: **GitHub Actions** will be triggered on code commits, running tests and building **Docker** images. **ArgoCD** will then deploy the new versions to our **Kubernetes cluster**.
- **Staged Deployment**: We will have three distinct environments (**staging, canary, production**) for gradual rollout of changes.
- **Cloud-Native Principles**: Our application will be designed as a collection of **Docker containers**, with each component (models, APIs, data pipelines) deployed as a separate **microservice**.
   

## Deployment and Execution

The Product is a Cloud Native application which can be bootstapped in multiple phases. The Development & Operations Lifecycle of the Product is distributed as follows. The Project can be launched on Trovi as [Legal AI - NLP CaseLaw Identifier](https://chameleoncloud.org/experiment/share/a49b98c4-fb07-41f0-a045-dd68735b7dc3)

- **Boostrapping Phase** : In this Phase we will be setting up the initial Infrastructure which is defined as IaaC using Terraform Scripts and Ansible Playbooks
- **Development Phase** : In this Phase we train the initial Model and also we evaluate our base performance we use GPU Machines and train Models using Ray Clusters.
- **Runtime Phase** : In Runtime we Monitor the Performance of our Model where we Check for the Performance of the Model also monitor the model for Data drift
- **Update Phase** : Once in the Runtime Phase if we observe issues or have bugs with our model or need to add new features we start back in the retraining phase where we make modifications to the existing model and then take the model till Production following a Staged Deployment
- **End Phase** : In End Phase we shut down all our systems gracefully ensuring all consumed Chameleon Cloud Resources are released

## Data pipeline
1. Create persistent storage and Use rclone and authenticate to object store from a compute instance
   - [1_create_server.ipynb](https://github.com/Sanjeevan1998/ml-ops-project/data-pipeline/1_create_server.ipynb)
   - [2_object.ipynb](https://github.com/Sanjeevan1998/ml-ops-project/data-pipeline/2_object.ipynb)

2. ETL
  - Extract - unzips data from the direct download link
  - Transform - Extracts metadata from raw case law files
  - Load - loads the data in object storage
    [docker-compose-etl.yaml](https://github.com/Sanjeevan1998/ml-ops-project/blob/main/data-pipeline/docker/docker-compose-etl.yaml)

3. Create labeled data
