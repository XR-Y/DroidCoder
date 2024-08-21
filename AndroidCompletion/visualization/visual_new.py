import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Manually create the DataFrame
data = {
    'Repo': ['Password-Store', 'BCR', 'fcitx5', 'Feeder', 'Iconify', 'MaterialFiles', 'Neo-Backup', 'QuickNovel', 'Trail-Sense', 'VinylMusicPlayer', 'Avg.'],
    'CodeT5+ 770M\nDroidCoder EM-Line': [60.31, 56.21, 52.58, 55.49, 60.20, 58.40, 40.86, 58.27, 56.65, 59.41, 55.84],
    'CodeT5+ 770M\nDroidCoder ES': [52.09, 44.62, 36.58, 45.28, 60.25, 47.77, 34.95, 46.58, 44.03, 55.90, 46.81],
    'CodeT5+ 220M\nDroidCoder EM-Line': [55.62, 52.17, 49.12, 52.26, 59.42, 54.24, 39.81, 54.78, 54.69, 56.97, 52.91],
    'CodeT5+ 220M\nDroidCoder ES': [43.28, 40.27, 33.48, 41.48, 58.94, 42.67, 35.86, 40.27, 41.36, 55.48, 43.31],
    'CodeT5+ 220M\nPure Fine-Tune EM-Line': [45.37, 48.46, 35.52, 39.15, 44.77, 44.69, 31.38, 41.81, 43.10, 44.07, 41.83],
    'CodeT5+ 220M\nPure Fine-Tune ES': [35.77, 38.84, 28.03, 33.51, 44.16, 35.39, 30.20, 33.09, 33.49, 40.08, 35.26],
    'StarCoderBase-1B\nDroidCoder EM-Line': [49.94, 53.76, 43.68, 53.44, 54.69, 53.20, 35.95, 53.43, 52.21, 52.21, 50.40],
    'StarCoderBase-1B\nDroidCoder ES': [43.81, 45.34, 31.68, 44.44, 53.91, 43.90, 34.48, 43.36, 41.08, 51.87, 43.39],
    'CodeGPT\nDroidCoder EM-Line': [32.64, 37.84, 34.01, 36.27, 40.07, 34.22, 25.05, 38.89, 37.54, 46.74, 36.33],
    'CodeGPT\nDroidCoder ES': [24.55, 19.71, 19.43, 22.42, 33.40, 21.00, 20.16, 22.91, 22.07, 43.54, 24.92],
    'CodeGPT\nFT2Ra EM-Line': [9.55, 16.71, 19.45, 16.20, 7.73, 10.08, 4.65, 14.91, 19.79, 13.29, 13.24],
    'CodeGPT\nFT2Ra ES': [13.11, 13.50, 10.58, 12.07, 28.02, 11.13, 12.13, 12.49, 13.91, 19.64, 14.66],
    'CodeT5+ 770M\nOriginal EM-Line': [22.84, 27.29, 19.06, 27.71, 31.49, 24.85, 16.22, 24.69, 25.49, 23.47, 24.31],
    'CodeT5+ 770M\nOriginal ES': [22.03, 25.94, 17.91, 25.63, 33.08, 24.11, 18.00, 22.58, 23.91, 24.88, 23.81],
    'ChatGPT3.5\nRepoCoder EM-Line': [19.08, 18.59, 22.41, 17.70, 17.42, 19.42, 15.22, 19.53, 19.39, 30.00, 19.88],
    'ChatGPT3.5\nRepoCoder ES': [14.70, 12.60, 13.22, 12.23, 15.24, 12.84, 12.03, 11.46, 11.26, 27.00, 14.26]
}

df = pd.DataFrame(data)

# Separate EM-Line and ES data
em_line_data = df.set_index('Repo').filter(like='EM-Line')
es_data = df.set_index('Repo').filter(like='ES')

# Update column names for simplicity
em_line_data.columns = [col.replace(' EM-Line', '') for col in em_line_data.columns]
es_data.columns = [col.replace(' ES', '') for col in es_data.columns]

# Create a single figure with two subplots for the heatmaps with consistent width
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5), gridspec_kw={'width_ratios': [0.82, 1]})
rcParams['font.family'] = 'DejaVu Sans'
rcParams['pdf.fonttype'] = 42

# Define a larger font size for annotations
annot_font_size = 16

# Plot EM-Line heatmap without the right color bar
sns.heatmap(em_line_data, annot=True, fmt=".2f", cmap='vlag', ax=axes[0], cbar=False, annot_kws={"size": annot_font_size})
axes[0].set_title('EM* Heatmap', fontsize=20)
axes[0].set_ylabel('')
axes[0].set_yticklabels(axes[0].get_yticklabels(), size=annot_font_size)
xticklabels = axes[0].get_xticklabels()
for i, label in enumerate(axes[0].get_xticklabels()):
    if "MDroidCoder" in str(label):
        label.set_fontweight('bold')
axes[0].set_xticklabels(xticklabels, size=10)

# Plot ES heatmap without the y-axis labels
heatmap = sns.heatmap(es_data, annot=True, fmt=".2f", cmap='vlag', ax=axes[1], annot_kws={"size": annot_font_size})
axes[1].set_title('ES Heatmap', fontsize=20)
axes[1].yaxis.set_visible(False)
xticklabels = axes[1].get_xticklabels()
for i, label in enumerate(axes[1].get_xticklabels()):
    if "MDroidCoder" in str(label):
        label.set_fontweight('bold')
axes[1].set_xticklabels(xticklabels, size=10)

# Increase font size of color bar ticks
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=annot_font_size)

# Rotate x-axis labels
for ax in axes:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='center')

# Adjust layout to make space for titles
plt.tight_layout()
plt.show()