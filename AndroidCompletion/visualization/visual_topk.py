import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

data = {
    'Repo': ['A18', 'A20', 'A19', 'A29', 'A28', 'A30',
             'A38', 'A39', 'A40', 'A9', 'A8', 'A10', 'Avg.'],
    'EM TOPK=10': [22.17, 11.18, 24.46, 18.29, 21.43, 26.63, 23.75, 23.75, 24.68, 27.97, 20.38, 22.38, 22.26],
    'ES TOPK=10': [46.87, 45.34, 48.10, 41.72, 43.18, 51.66, 46.06, 49.54, 50.20, 51.07, 44.70, 43.20, 46.80],
    'EM TOPK=5': [21.74, 9.41, 23.31, 16.58, 20.45, 23.91, 20.92, 21.80, 22.08, 27.56, 18.85, 22.97, 20.80],
    'ES TOPK=5': [47.96, 41.38, 47.17, 38.82, 43.39, 48.19, 43.92, 48.59, 49.80, 50.39, 43.38, 41.81, 45.40],
    'EM TOPK=20': [18.26, 11.76, 20.60, 17.26, 21.43, 22.01, 19.49, 20.29, 22.73, 28.38, 19.04, 25.29, 20.55],
    'ES TOPK=20': [45.82, 44.85, 45.73, 40.06, 42.55, 48.88, 41.86, 50.13, 49.31, 51.93, 44.01, 43.90, 45.75],
    'EM TOPK=50': [20.43, 12.94, 21.76, 16.58, 20.78, 22.34, 20.19, 21.48, 18.18, 27.56, 17.69, 20.64, 20.05],
    'ES TOPK=50': [47.02, 44.88, 45.77, 39.72, 43.43, 49.20, 43.37, 50.75, 45.38, 50.98, 43.11, 43.50, 45.59]
}

df = pd.DataFrame(data)

# Melt the DataFrame to a long format
df_long = df.melt(id_vars=['Repo'], var_name='Metric', value_name='Score')

# Split the 'Metric' into two columns: 'Type' and 'Category'
df_long[['Type', 'Category']] = df_long['Metric'].str.split(' ', n=1, expand=True)

# Filter the DataFrame for separate plots
df_em = df_long[df_long['Type'] == 'EM']
df_es = df_long[df_long['Type'] == 'ES']


# Setting up the figure
fig, axes = plt.subplots(1, 1, figsize=(6, 6))
rcParams['font.family'] = 'DejaVu Sans'
rcParams['pdf.fonttype'] = 42
palette = sns.color_palette("colorblind", 4)
sns.despine()

# Plot for EM-Line
sns.lineplot(data=df_em, x='Repo', y='Score', hue='Category', style='Category', markers=True, dashes=False, ax=axes, palette=palette)
axes.set_title('')
axes.set_xlabel('')  # Hide x-label for the top plot
axes.set_ylabel('EM Score(%)')
axes.legend(title='', frameon=False, prop={'size': 10}, loc='lower right')
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
axes.legend(title='', frameon=False, prop={'size': 10}, loc='lower right')
axes.tick_params(axis='x', labelsize=8)
# Improve layout and display
plt.tight_layout()
