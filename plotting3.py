import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend to avoid Tkinter issues on Windows

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

# Your distance metrics data
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

# Load the data into a DataFrame
df = pd.read_csv(StringIO(data))

# Pivot to get metrics as columns
pivot_df = df.pivot(index="repo", columns="metric", values="mean_distance")

# Set up bar positions
repos = pivot_df.index.tolist()
metrics = pivot_df.columns.tolist()
x = np.arange(len(repos))
width = 0.15

# Colors for each metric
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 5 distinct colors

# Plotting
plt.figure(figsize=(14, 6))
for i, metric in enumerate(metrics):
    plt.bar(x + i * width, pivot_df[metric], width, label=metric, color=colors[i])

plt.xlabel('Repository')
plt.ylabel('Mean Distance')
plt.title('Mean Distance Metrics per Repository')
plt.xticks(x + width * 2, repos, rotation=45)
plt.legend(title="Metric")
plt.tight_layout()
plt.savefig("distance_metrics_per_repo.png")
print("✅ Plot saved as 'distance_metrics_per_repo.png'")

