import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for Windows

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

# Your data
data = """
repo,metric,mean_distance
pantherdb,tokens,39.916666666666664
pantherdb,bleu,0.8647924490596737
pantherdb,cosine,0.3542210566991961
pantherdb,jaccard,0.3788228390645639
pantherdb,tf,0.4884208400208638
datefinder,tokens,150.5897435897436
datefinder,bleu,0.9223862641400351
datefinder,cosine,0.3755072067829989
datefinder,jaccard,0.4262303027823042
datefinder,tf,0.47299157012237114
executing,tokens,206.57894736842104
executing,bleu,0.8781113212993507
executing,cosine,0.327241151714444
executing,jaccard,0.3257786958813068
executing,tf,0.42876969147832794
sqlparse,tokens,106.95679012345678
sqlparse,bleu,0.8833463793912236
sqlparse,cosine,0.37364536753088323
sqlparse,jaccard,0.4526384181933748
sqlparse,tf,0.4797509644141861
phantom-types,tokens,72.33333333333333
phantom-types,bleu,0.9124267690409811
phantom-types,cosine,0.33293297058534943
phantom-types,jaccard,0.3298243713898854
phantom-types,tf,0.4484111422463234
python-pathspec,tokens,241.63492063492063
python-pathspec,bleu,0.9816954936875137
python-pathspec,cosine,0.5027819560473892
python-pathspec,jaccard,0.3968825382418306
python-pathspec,tf,0.602442299135103
"""

# Load data
df = pd.read_csv(StringIO(data))
pivot_df = df.pivot(index="repo", columns="metric", values="mean_distance")

repos = pivot_df.index.tolist()
x = np.arange(len(repos))
width = 0.15

# Define metrics
token_metric = 'tokens'
other_metrics = ['bleu', 'cosine', 'jaccard', 'tf']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Plotting
fig, ax1 = plt.subplots(figsize=(14, 6))
ax2 = ax1.twinx()

# Plot 'tokens' on ax1 (left y-axis)
ax1.bar(x, pivot_df[token_metric], width, label=token_metric, color=colors[0])

# Plot other metrics on ax2 (right y-axis)
for i, metric in enumerate(other_metrics):
    ax2.bar(x + (i + 1) * width, pivot_df[metric], width, label=metric, color=colors[i + 1])

# Axis and label formatting
ax1.set_ylabel('Token Distance', color=colors[0])
ax2.set_ylabel('Other Distances (bleu, cosine, jaccard, tf)')
ax1.set_xlabel('Repository')
ax1.set_xticks(x + 2 * width)
ax1.set_xticklabels(repos, rotation=45)

# Combine legends
bars_labels = [token_metric] + other_metrics
handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(bars_labels))]
ax1.legend(handles, bars_labels, title="Metric")

plt.title("Mean Distance Metrics per Repository (Dual Y-Axis)")
plt.tight_layout()
plt.savefig("distance_dual_axis_plot.png")
print("✅ Dual-axis plot saved as 'distance_dual_axis_plot.png'")
