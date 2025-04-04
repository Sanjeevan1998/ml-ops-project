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
| All team members                |Idea formulation, value proposition, ML problem setup, integration|                                    |
| Arnab Bhowal                   |Model training (Units 4 & 5)|                                    |
| Sanjeevan Adhikari                   |Model serving (Unit 6) and monitoring (Unit 7)|                                    |
| Divya Chougule                   |Data pipeline (Unit 8)|                                    |
| Vishwas Karale                   |CICD pipeline|                                    |



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


