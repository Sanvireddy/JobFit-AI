import re
import sqlite3
import spacy
from spacy.matcher import PhraseMatcher

conn = sqlite3.connect("jobs.db")
cursor = conn.cursor()

def extract_experience():
    sql_query = "SELECT description, job_id FROM scraped_jobs"
    cursor.execute(sql_query)
    job_desc_list = cursor.fetchall()

    pattern = re.compile(
        r"(?<!\d)(\d+)\s*(?:-\s*(\d+)\s*\+?|\+)?\s*(years?|yrs?|yoe)\b",
        re.IGNORECASE
    )

    for job_desc in job_desc_list:
        description = job_desc[0]
        job_id = job_desc[1]

        results = []
        matches = pattern.finditer(description)

        for match in matches:
            min_years = int(match.group(1))
            max_years = match.group(2)
            max_years = int(max_years) if max_years else None

            results.append({
                "text": match.group(0),
                "min_years": min_years,
                "max_years": max_years
            })

        print(results)

        if len(results) ==0 or len(results)>1:
            final_result = 2
        else:
            exp = results[0]['min_years']
            final_result = 1 if exp < 4 else 0

        print(job_id, final_result)

def get_required_skills():
    sql_query = "SELECT description, job_id FROM scraped_jobs"
    cursor.execute(sql_query)
    job_desc_list = cursor.fetchall()
    conn.commit()
    skills = ['python', 'r', 'sql', 'java', 'scala', 'julia', 'matlab', 'c++', 'bash', 'linux', 'git', 'github', 'gitlab', 'machine learning', 'ml', 'deep learning', 'dl', 'supervised learning', 'unsupervised learning', 'reinforcement learning', 'rl', 'feature engineering', 'feature selection', 'model evaluation', 'model validation', 'hyperparameter tuning', 'cross validation', 'regularization', 'bias variance tradeoff', 'linear regression', 'logistic regression', 'decision tree', 'decision trees', 'random forest', 'gradient boosting', 'xgboost', 'lightgbm', 'catboost', 'support vector machine', 'svm', 'k nearest neighbors', 'knn', 'naive bayes', 'k means', 'hierarchical clustering', 'dbscan', 'gaussian mixture model', 'gmm', 'neural networks', 'artificial neural networks', 'convolutional neural networks', 'cnn', 'recurrent neural networks', 'rnn', 'lstm', 'gru', 'transformers', 'attention mechanism', 'transfer learning', 'fine tuning', 'model training', 'model optimization', 'tensorflow', 'keras', 'pytorch', 'jax', 'scikit learn', 'sklearn', 'natural language processing', 'nlp', 'text processing', 'text mining', 'tokenization', 'stemming', 'lemmatization', 'tf idf', 'word embeddings', 'word2vec', 'glove', 'bert', 'gpt', 'sentence transformers', 'topic modeling', 'latent dirichlet allocation', 'lda', 'named entity recognition', 'ner', 'text classification', 'sentiment analysis', 'semantic search', 'information retrieval', 'retrieval augmented generation', 'rag', 'prompt engineering', 'prompt design', 'large language models', 'llm', 'llms', 'llm fine tuning', 'embeddings', 'vector embeddings', 'vector search', 'vector databases', 'pinecone', 'weaviate', 'faiss', 'chroma', 'milvus', 'langchain', 'llamaindex', 'huggingface transformers', 'spacy', 'nltk', 'haystack', 'pandas', 'numpy', 'scipy', 'polars', 'dask', 'exploratory data analysis', 'eda', 'statistical analysis', 'hypothesis testing', 'a/b testing', 'experiment design', 'experimentation', 'causal inference', 'time series analysis', 'time series forecasting', 'forecasting', 'anomaly detection', 'recommendation systems', 'recommender systems', 'clustering', 'dimensionality reduction', 'principal component analysis', 'pca', 't sne', 'umap', 'matplotlib', 'seaborn', 'plotly', 'tableau', 'power bi', 'looker', 'apache superset', 'ggplot', 'spark', 'pyspark', 'hadoop', 'hive', 'presto', 'trino', 'kafka', 'airflow', 'apache airflow', 'flink', 'databricks', 'data pipelines', 'etl', 'elt', 'data engineering', 'postgresql', 'mysql', 'sqlite', 'snowflake', 'bigquery', 'amazon redshift', 'mongodb', 'elasticsearch', 'aws', 'amazon web services', 'gcp', 'google cloud platform', 'azure', 's3', 'sagemaker', 'aws lambda', 'ec2', 'aws glue', 'athena', 'vertex ai', 'azure machine learning', 'azure databricks', 'mlflow', 'kubeflow', 'docker', 'kubernetes', 'ci cd', 'continuous integration', 'continuous deployment', 'model deployment', 'model monitoring', 'model serving', 'model versioning', 'feature store', 'tensorflow serving', 'torchserve', 'bentoml', 'fastapi', 'flask', 'rest api', 'rest apis', 'microservices', 'software engineering', 'unit testing', 'software architecture', 'probability', 'statistics', 'linear algebra', 'optimization', 'bayesian statistics', 'information theory', 'stochastic processes', 'data visualization', 'data storytelling', 'stakeholder communication', 'business analytics', 'product analytics', 'kpi analysis', 'credit risk modeling', 'fraud detection', 'customer segmentation', 'churn prediction']

    for job_desc in job_desc_list:
        description = job_desc[0]
        job_id = job_desc[1]
        nlp = spacy.load("en_core_web_sm")
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        patterns = [nlp.make_doc(skill) for skill in skills]
        matcher.add("SKILLS",patterns)
        doc = nlp(description)
        matches = matcher(doc)
        allmatches = []
        for match_id, start, end in matches:
            allmatches.append(doc[start:end].text)
        if len(allmatches)==0: print(job_id)
get_required_skills()