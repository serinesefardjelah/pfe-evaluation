import pandas as pd
import numpy as np
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
from nltk.tokenize import word_tokenize


nltk.download("punkt")

repos = [
    "https://github.com/PantherPy/pantherdb",
    "https://github.com/akoumjian/datefinder",
    "https://github.com/alexmojaki/executing",
    "https://github.com/andialbrecht/sqlparse",
    "https://github.com/antonagestam/phantom-types",
    "https://github.com/cpburnz/python-pathspec"
]


# Tokenizer function
# def tokenize_code(code, tokens='nltk'):
#     if tokens == "nltk":
#         return word_tokenize(code)
#     elif tokens == "words":
#         return code.split()
#     else:
#         raise Exception("Not a valid tokens type")

import re

def tokenize_code(code, tokens='nltk'):
    if tokens == "nltk":
        return word_tokenize(code)
    elif tokens == "words":
        return code.split()
    elif tokens == "simple":
        return re.findall(r"\w+|[^\w\s]", code)
    else:
        raise Exception("Not a valid tokens type")

# Distance metrics
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
    return intersection / union if union > 0 else 0

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

# Distance computation function
def report_distance(data_file_path, metric='bleu'):
    df = pd.read_csv(data_file_path)
    overall_distances = []

    grouped = df.groupby(['Repo Name', 'Function Name'])

    for (repo_name, function_name), group in grouped:
        baseline_code = group.iloc[0]['original_code']  # Assume original_code is the same across group
        for _, row in group.iterrows():
            mutant_code = row['Mutant Code']
            if pd.isna(mutant_code):
                continue

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

            overall_distances.append({
                "repo": repo_name,
                "function": function_name,
                "mutant_index": row["Mutant Index"],
                "distance": distance
            })

    return overall_distances

# Save results and plot boxplots
def save_and_plot_distances(distances, label):
    df = pd.DataFrame(distances)
    os.makedirs("distance_results", exist_ok=True)
    
    # Save JSONL
    jsonl_path = f"distance_results/{label}.jsonl"
    df.to_json(jsonl_path, orient='records', lines=True)

    # Plot
    plt.figure(figsize=(16, 10))
    sns.boxplot(data=df, x="function", y="distance")
    plt.title(f"Syntactic Distance Distribution - {label}")
    plt.ylabel(f"Distance")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"distance_results/{label}.png", dpi=300)
    plt.close()

# Helper to extract repo name from URL
def extract_repo_basename(repo_url):
    return repo_url.rstrip("/").split("/")[-1]

# Main batch-processing logic
def main():
    metrics = ['tokens', 'bleu', 'cosine', 'jaccard', 'tf']
    all_results = []

    for repo_url in repos:
        repo_name = extract_repo_basename(repo_url)
        file_path = f"repo_stats_with_code/{repo_name}_mutation_results.csv"
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}. Skipping...")
            continue

        print(f"\nProcessing repository: {repo_name}")
        for metric in metrics:
            print(f"  Computing distances using {metric} metric...")
            distances = report_distance(file_path, metric=metric)
            if distances:
                label = f"{repo_name}_{metric}"
                save_and_plot_distances(distances, label)
                mean_distance = np.mean([d['distance'] for d in distances])
                all_results.append({
                    "repo": repo_name,
                    "metric": metric,
                    "mean_distance": mean_distance
                })

    # Save overall summary
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv("syntactic_distance_results/distance_summary.csv", index=False)
    print("\n✅ Summary saved to syntactic_distance_results/distance_summary.csv")

if __name__ == "__main__":
    main()
