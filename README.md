# Break Through AI Final Project - Job Title Clustering

A machine learning pipeline for analyzing and clustering job descriptions, using a dataset of over 20,000 web-scraped job postings. The project classifies job posting sentences, trains local classifiers, and performs clustering and topic modeling on job descriptions.

## Features

- Sentence-level classification of job descriptions using GPT-4
- Local ML model training (Logistic Regression, Random Forest, XGBoost) for sentence categorization
- K-means clustering with optimal cluster selection
- LDA topic modeling for job description analysis
- Visualization of clusters and topics

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd job-title-clustering
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your OpenAI API key: `OPENAI_API_KEY=your_api_key_here`

## Usage

### 1. Classify Job Descriptions with GPT

Classify job posting sentences into categories (Marketing, Description, Requirements, Legal):

```bash
python src/gpt_classifier.py
```

This processes job descriptions and saves results to `data/gpt_classified_job_descriptions.csv`.

### 2. Train Local Classifier

Train ML models on GPT-classified data and predict on remaining job postings:

```bash
python src/train_local_classifier.py
```

This trains multiple models, selects the best one, and saves it to `models/best_classifier.pkl`.

### 3. Recombine Job Descriptions

Combine classified sentences back into full job descriptions:

```bash
python src/recombine_job_descriptions.py
```

This creates `data/recombined_descriptions.csv` with filtered job descriptions.

### 4. K-means Clustering

Cluster job descriptions using K-means:

```bash
python src/kmeans.py
```

This finds optimal clusters, generates visualizations, and saves results to `data/k-means-clustered_jobs.csv`.

### 5. LDA Topic Modeling

Run Latent Dirichlet Allocation topic modeling:

```bash
python src/run_lda.py
```

This trains an LDA model and generates an interactive visualization saved to `models/lda_vis.html`.

## Project Structure

```
job-title-clustering/
├── data/                    # Input/output data files
├── models/                   # Trained models and visualizations
├── notebooks/               # Jupyter notebooks for EDA
├── src/                     # Source code
│   ├── config.py           # Configuration settings
│   ├── preprocessing.py    # Text cleaning utilities
│   ├── gpt_classifier.py   # GPT-based sentence classification
│   ├── train_local_classifier.py  # ML model training
│   ├── recombine_job_descriptions.py  # Sentence recombination
│   ├── kmeans.py           # K-means clustering
│   └── run_lda.py          # LDA topic modeling
└── requirements.txt        # Python dependencies
```

## Data Requirements

The pipeline expects input data in CSV format with the following columns:
- `RequisitionID`: Unique identifier for each job posting
- `JobTitle`: Job title
- `JobDescription`: Full job description text
- `OrigJobTitle`: Original job title (optional)

## Output Files

- `data/gpt_classified_job_descriptions.csv`: GPT-classified sentences
- `data/model_classified_job_descriptions.csv`: Model-classified sentences
- `data/recombined_descriptions.csv`: Recombined job descriptions
- `data/k-means-clustered_jobs.csv`: Clustered job postings
- `data/k-means-cluster_labels.csv`: Cluster labels and keywords
- `models/best_classifier.pkl`: Trained classifier model
- `models/lda_vis.html`: Interactive LDA visualization
