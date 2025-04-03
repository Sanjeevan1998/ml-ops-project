## Mindful: NLP-based Journal Summarization and Sentiment Analysis

Our machine learning system integrates directly into existing psychological counseling practices. Psychologists regularly ask patients to maintain therapeutic journals, manually reviewing these entries to detect trends, emotions, and possible risk factors. This manual process is labor-intensive, subjective, and difficult to scale. Our proposed ML assistant will automatically summarize journal entries, perform sentiment and emotional analysis, and highlight important or concerning themes. This allows therapists to quickly identify areas requiring attention, track patient progress objectively, and increase the number of patients they can effectively support.

Non-ML Status Quo:<br>
Currently, psychologists manually review patient journal entries, which is time-consuming, subjective, and prone to missing subtle emotional patterns or risks.

Business Metrics:<br>
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
| All team members                |Idea formulation, value proposition, ML problem setup, integration|                                    |
| Arnab Bhowal                   |Model training (Units 4 & 5)|                                    |
| Sanjeevan Adhikari                   |Model serving (Unit 6) and monitoring (Unit 7)|                                    |
| Divya Chougule                   |Data pipeline (Unit 8)|                                    |
| Vishwas Karale                   |CICD pipeline|                                    |



### System diagram

Chameleon Cloud Infrastructure
│
├── Persistent Storage (200GB)
│   ├── Raw Journal Data
│   ├── Processed Training/Inference Datasets
│   └── Model artifacts, container images
│
└── Kubernetes Cluster
    ├── Data Pipelines (Containers)
    │   ├── Kafka (Online streaming)
    │   └── Apache Airflow (Offline ETL)
    │
    ├── Model Training & Experiment Tracking (Containers)
    │   ├── Ray Cluster (Distributed Training & Hyperparameter Tuning)
    │   ├── MLFlow Server (Experiment Tracking)
    │   └── GPU Nodes
    │
    ├── Model Serving API (Containers, FastAPI)
    │   ├── Summarization Model (BART)
    │   ├── Sentiment Analysis Model (RoBERTa)
    │   └── Risk Detection Model (DistilBERT)
    │
    ├── CI/CD & Continuous Training (Containers)
    │   ├── GitHub Actions, ArgoCD/Argo Workflows
    │   ├── Docker Registry
    │   └── Terraform & Ansible
    │
    └── Monitoring & Evaluation (Containers)
        ├── Prometheus & Grafana Dashboard
        ├── Automated Load Tests
        └── Data/Model Drift Monitoring

Staged Deployment: Staging → Canary → Production → Psychologist's Interface & Dashboard

### Summary of outside materials

<!-- In a table, a row for each dataset, foundation model. 
Name of data/model, conditions under which it was created (ideally with links/references), 
conditions under which it may be used. -->

|              | How it was created | Conditions of use |
|--------------|--------------------|-------------------|
| Data set 1   |                    |                   |
| Data set 2   |                    |                   |
| Base model 1 |                    |                   |
| etc          |                    |                   |


### Summary of infrastructure requirements

<!-- Itemize all your anticipated requirements: What (`m1.medium` VM, `gpu_mi100`), 
how much/when, justification. Include compute, floating IPs, persistent storage. 
The table below shows an example, it is not a recommendation. -->

| Requirement     | How many/when                                     | Justification |
|-----------------|---------------------------------------------------|---------------|
| `m1.medium` VMs | 3 for entire project duration                     | ...           |
| `gpu_mi100`     | 4 hour block twice a week                         |               |
| Floating IPs    | 1 for entire project duration, 1 for sporadic use |               |
| etc             |                                                   |               |

### Detailed design plan

<!-- In each section, you should describe (1) your strategy, (2) the relevant parts of the 
diagram, (3) justification for your strategy, (4) relate back to lecture material, 
(5) include specific numbers. -->

#### Model training and training platforms

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


