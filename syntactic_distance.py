import pandas as pd
import numpy as np
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
DATASET_FILE = "humaneval_java_plus_no_outliers_correct.csv"

# Tokenizer function (to be used for different tokenization strategies)
def tokenize_code(code, tokens='nltk'):
    if tokens == "nltk":
        return nltk.word_tokenize(code)
    elif tokens == "words":
        return code.split()
    else:
        raise Exception("Not a valid tokens type")

def compute_bleu_score(reference_code, candidate_code, tokens):
    smoothing_function = SmoothingFunction().method4
    reference_code_tokens = tokenize_code(reference_code, tokens)
    candidate_code_tokens = tokenize_code(candidate_code, tokens)
    return sentence_bleu([reference_code_tokens], candidate_code_tokens, smoothing_function=smoothing_function)

def compute_cosine_distance(reference_code, candidate_code, tokens):
    count_vectorizer = CountVectorizer(tokenizer=lambda code: tokenize_code(code, tokens))
    sparse_matrix = count_vectorizer.fit_transform([reference_code, candidate_code])
    cosine_sim_matrix = 1 - cosine_similarity(sparse_matrix, sparse_matrix)
    return cosine_sim_matrix[0][1]

def compute_jaccard_sim(reference_code, candidate_code, tokens):
    reference_code_tokens = tokenize_code(reference_code, tokens)
    candidate_code_tokens = tokenize_code(candidate_code, tokens)
    reference_code_tokens_set = set(reference_code_tokens)
    candidate_code_tokens_set = set(candidate_code_tokens)
    intersection = len(reference_code_tokens_set.intersection(candidate_code_tokens_set))
    union = len(reference_code_tokens_set) + len(candidate_code_tokens_set) - len(reference_code_tokens_set)
    return intersection / union

def sequence_based_distance(reference_code, candidate_code, tokens):
    reference_code_tokens = tokenize_code(reference_code, tokens)
    candidate_code_tokens = tokenize_code(candidate_code, tokens)
    matcher = SequenceMatcher(None, reference_code_tokens, candidate_code_tokens)
    tokens_differences = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            tokens_differences += len(reference_code_tokens[i1:i2])
    return tokens_differences

def compute_tf_idf_distance(reference_code, candidate_code, tokens):
    tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda code: tokenize_code(code, tokens))
    sparse_matrix = tfidf_vectorizer.fit_transform([reference_code, candidate_code])
    cosine_sim_matrix = 1 - cosine_similarity(sparse_matrix, sparse_matrix)
    return cosine_sim_matrix[0][1]

# Function to compute distances and return results
def report_distance(intent_file_path, reference_file_path, metric='bleu', tasks=[]):
    mutated = pd.read_csv(intent_file_path)
    original = pd.read_csv(reference_file_path)
    
    overall_distances = []
    
    for task_id in tasks:
        task_original = original[original['task_id'] == task_id]
        task_mutated = mutated[mutated['task_id'] == task_id]

        baseline_code = task_original.iloc[0]['code']  # Assuming the canonical solution is the first one in the list
        task_distances = []

        for _, mutant in task_mutated.iterrows():
            mutant_code = mutant['code']
            if metric == 'bleu':
                distance = 1 - compute_bleu_score(baseline_code, mutant_code, tokens='nltk')
            elif metric == 'cosine':
                distance = compute_cosine_distance(baseline_code, mutant_code, tokens='nltk')
            elif metric == 'jaccard':
                distance = 1 - compute_jaccard_sim(baseline_code, mutant_code, tokens='nltk')
            elif metric == 'tokens':
                distance = sequence_based_distance(baseline_code, mutant_code, tokens='nltk')
            elif metric == 'tf':
                distance = compute_tf_idf_distance(baseline_code, mutant_code, tokens='nltk')
            else:
                raise Exception("Invalid metric. Choose one from: bleu, cosine, jaccard, tokens, tf.")

            task_distances.append({"task_id": task_id, "mutant_id": mutant['mutant_id'], "distance": distance})

        overall_distances.extend(task_distances)
    
    return overall_distances

# Function to save distances to JSONL and create boxplots
def save_and_plot_distances(distances, metric):
    df = pd.DataFrame(distances)
    
    # Save results to JSONL
    df.to_json(f"distance_results/{metric}.jsonl", orient='records', lines=True)

    # Plot distance distribution
    plt.figure(figsize=(16, 10))
    sns.boxplot(data=df, x="task_id", y="distance")
    plt.title(f"Syntactic Distance Distribution - {metric}")
    plt.ylabel(f"Distance (1 - {metric.upper()})")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"distance_results/{metric}.png", dpi=300)

# Main function to compare distances for different metrics
def main():
    intent_file_path = "/path/to/your/mutants.csv"  # Path to the mutants CSV
    original_file_path = "/path/to/your/original.csv"  # Path to the original solutions CSV
    tasks = ['Java/119', 'Java/154', 'Java/120']  # List of tasks to process
    metrics = ['tokens', 'bleu', 'cosine', 'jaccard', 'tf']  # Metrics to use

    results = []
    for metric in metrics:
        print(f"Computing distances using {metric} metric...")
        distances = report_distance(intent_file_path, original_file_path, metric=metric, tasks=tasks)
        save_and_plot_distances(distances, metric)
        mean_distance = np.mean([d['distance'] for d in distances])
        results.append([metric, mean_distance])
    
    print("Distance results (metric, mean distance):", results)

if __name__ == "__main__":
    main()
