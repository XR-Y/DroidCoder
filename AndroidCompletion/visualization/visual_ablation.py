import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

data = {
    'Repo': ['A18', 'A20', 'A19', 'A29', 'A28', 'A30',
             'A38', 'A39', 'A40', 'A9', 'A8', 'A10', 'Avg.'],
    'EM DroidCoder': [22.17, 11.18, 24.46, 18.29, 21.43, 26.63, 23.75, 23.75, 24.68, 27.97, 20.38, 22.38, 22.26],
    'ES DroidCoder': [46.87, 45.34, 48.10, 41.72, 43.18, 51.66, 46.06, 49.54, 50.20, 51.07, 44.70, 43.20, 46.80],
    'EM Without Retriever': [18.26, 8.82, 21.47, 11.79, 14.29, 20.65, 18.62, 21.85, 14.94, 23.87, 13.08, 16.57, 17.02],
    'ES Without Retriever': [42.15, 39.55, 45.28, 33.49, 36.73, 43.99, 39.92, 47.47, 38.81, 46.23, 36.65, 36.74, 40.58],
    'EM Without Reranker': [14.78, 11.76, 19.41, 9.23, 11.69, 16.58, 15.65, 19.19, 16.34, 21.45, 12.50, 11.34, 14.99],
    'ES Without Reranker': [39.81, 39.11, 42.99, 32.01, 34.85, 41.01, 38.12, 45.84, 39.51, 43.46, 37.33, 38.23, 39.36],
    'EM Without Context\nEnhancement': [23.04, 11.18, 22.13, 14.02, 16.23, 22.83, 20.65, 21.14, 21.43, 25.92, 15.00, 22.38, 19.66],
    'ES Without Context\nEnhancement': [46.87, 43.08, 45.63, 36.85, 38.59, 46.09, 42.55, 47.28, 46.84, 49.55, 38.53, 40.59, 43.54],
    'EM Pure Fine-Tune': [10.00, 7.65, 17.69, 6.67, 8.77, 19.57, 8.37, 21.56, 8.44, 20.46, 8.27, 10.76, 12.35],
    'ES Pure Fine-Tune': [34.83, 31.59, 41.19, 27.06, 31.33, 42.41, 33.08, 44.33, 33.33, 44.72, 33.87, 44.19, 36.83],
    'M-EM-Line MDroidCoder': [49.32, 51.61, 51.33, 50.60, 51.40, 48.85, 53.17, 49.67, 49.92, 53.96, 50.47, 54.34, 51.22],
    'M-ES MDroidCoder': [40.59, 42.51, 43.88, 38.98, 39.21, 45.74, 42.17, 44.90, 42.63, 46.12, 41.01, 41.72, 42.46],
    'M-EM-Line Without Retriever': [45.78, 41.69, 48.62, 43.08, 44.01, 44.48, 46.30, 45.82, 43.78, 49.52, 43.07, 49.93, 45.51],
    'M-ES Without Retriever': [36.68, 35.00, 39.19, 32.13, 30.72, 38.97, 35.06, 40.70, 36.17, 40.42, 33.05, 35.04, 36.09],
    'M-EM-Line Without Reranker': [41.02, 39.15, 45.43, 37.06, 40.04, 42.21, 40.41, 42.65, 42.08, 43.98, 41.29, 38.02, 41.11],
    'M-ES Without Reranker': [37.23, 31.81, 34.85, 29.90, 27.00, 36.44, 32.87, 37.59, 34.03, 36.78, 31.43, 30.78, 33.39],
    'M-EM-Line Without Context\nEnhancement': [44.99, 42.01, 47.09, 42.50, 44.86, 44.66, 48.50, 43.94, 46.36, 48.10, 42.78, 51.86, 45.64],
    'M-ES Without Context\nEnhancement': [38.43, 37.98, 39.64, 33.82, 33.29, 40.44, 39.30, 40.62, 39.50, 41.56, 35.73, 39.72, 38.34],
    'M-EM-Line Pure Fine-Tune': [34.14, 38.40, 43.21, 32.91, 38.55, 41.65, 35.37, 42.34, 33.52, 46.81, 35.61, 36.28, 38.23],
    'M-ES Pure Fine-Tune': [30.10, 31.49, 37.85, 27.50, 29.30, 39.07, 31.88, 39.17, 31.17, 40.91, 31.85, 32.09, 33.53]
}

df = pd.DataFrame(data)

# Melt the DataFrame to a long format
df_long = df.melt(id_vars=['Repo'], var_name='Metric', value_name='Score')

# Split the 'Metric' into two columns: 'Type' and 'Category'
df_long[['Type', 'Category']] = df_long['Metric'].str.split(' ', n=1, expand=True)

# Filter the DataFrame for separate plots
df_em_line = df_long[df_long['Type'] == 'M-EM-Line']
df_em = df_long[df_long['Type'] == 'EM']
df_es_line = df_long[df_long['Type'] == 'M-ES']
df_es = df_long[df_long['Type'] == 'ES']


# Setting up the figure
rcParams['font.family'] = 'DejaVu Sans'
rcParams['pdf.fonttype'] = 42
palette = sns.color_palette("colorblind", 5)

# Plot for EM-Line
fig, axes = plt.subplots(1, 1, figsize=(6, 6))
sns.despine()
sns.lineplot(data=df_em, x='Repo', y='Score', hue='Category', style='Category', markers=True, dashes=False, ax=axes, palette=palette)
axes.set_title('')
axes.set_xlabel('')  # Hide x-label for the top plot
axes.set_ylabel('EM Score(%)')
axes.legend(title='', frameon=False, prop={'size': 9.5}, ncol=2, bbox_to_anchor=(0, 1.055), loc='upper left')
axes.tick_params(axis='x', labelsize=8)
# Improve layout and display
plt.tight_layout()

# Plot for ES
fig, axes = plt.subplots(1, 1, figsize=(6, 6))
sns.despine()
sns.lineplot(data=df_es, x='Repo', y='Score', hue='Category', style='Category', markers=True, dashes=False, ax=axes, palette=palette)
axes.set_title('')
axes.set_xlabel('')
axes.set_ylabel('ES Score(%)')
axes.legend(title='', frameon=False, prop={'size': 9.5}, ncol=2, bbox_to_anchor=(1.05, 0), loc='lower right')
axes.tick_params(axis='x', labelsize=8)
# Improve layout and display
plt.tight_layout()

# Duplicate Plot for EM-Line
fig, axes = plt.subplots(1, 1, figsize=(6, 6))
sns.despine()
sns.lineplot(data=df_em_line, x='Repo', y='Score', hue='Category', style='Category', markers=True, dashes=False, ax=axes, palette=palette)
axes.set_title('')
axes.set_xlabel('')  # Hide x-label for the bottom plot
axes.set_ylabel('EM-Line Score(%)')
axes.legend(title='', frameon=False, prop={'size': 9.5}, ncol=2, bbox_to_anchor=(0, 1.055), loc='upper left')
axes.tick_params(axis='x', labelsize=8)
# Improve layout and display
plt.tight_layout()

# Duplicate Plot for ES
fig, axes = plt.subplots(1, 1, figsize=(6, 6))
sns.despine()
sns.lineplot(data=df_es_line, x='Repo', y='Score', hue='Category', style='Category', markers=True, dashes=False, ax=axes, palette=palette)
axes.set_title('')
axes.set_xlabel('')
axes.set_ylabel('ES Score(%)')
axes.legend(title='', frameon=False, prop={'size': 9.5}, ncol=2, bbox_to_anchor=(1.05, 0), loc='lower right')
axes.tick_params(axis='x', labelsize=8)
# Improve layout and display
plt.tight_layout()
