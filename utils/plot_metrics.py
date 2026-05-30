import json
import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

file_patterns = {
    'NLLB-200-distilled-600M': '../experiments/nllb-200-distilled-600m-kjh-ru-finetune/dev_metrics/*metrics.json',
    'Hy-MT2-1.8B': '../experiments/hy-mt2-1.8b-kjh-ru-lora-finetune/dev_metrics/*metrics.json'
}

data_list = []

for model_name, file_pattern in file_patterns.items():
    files = glob.glob(file_pattern)

    assert len(files) > 0

    for file_path in files:
        file_name = os.path.basename(file_path)
        step_match = re.search(r'(\d+)', file_name)
        if step_match:
            step = int(step_match.group(1))

            if model_name == 'Hy-MT2-1.8B':
                step *= 2  # the batch was 2 times larger, which means 2 times more data

            with open(file_path, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
                for direction, metrics in metrics_data.items():
                    if direction.startswith('rus') or direction.startswith('Russian'):
                        clean_direction = 'ru → kjh'
                    else:
                        clean_direction = 'kjh → ru'

                    data_list.append({
                        'model': model_name,
                        'step': step,
                        'direction': clean_direction,
                        'bleu': metrics.get('bleu'),
                        'chrf': metrics.get('chrf++')
                    })

df = pd.DataFrame(data_list).sort_values('step')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'font.family': 'serif'
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

model_colors = {
    'NLLB-200-distilled-600M': '#1f77b4',
    'Hy-MT2-1.8B': '#d62728'
}
direction_styles = {
    'kjh → ru': '-',
    'ru → kjh': '--'
}
direction_markers = {
    'kjh → ru': 'o',
    'ru → kjh': 's'
}

directions = df['direction'].unique()
models = df['model'].unique()

for model in models:
    color = model_colors.get(model, '#333333')

    for direction in directions:
        subset = df[(df['direction'] == direction) & (df['model'] == model)]
        if subset.empty:
            continue

        l_style = direction_styles.get(direction, '-')
        marker = direction_markers.get(direction, 'o')
        label_text = f'{model} ({direction})'

        ax1.plot(subset['step'], subset['bleu'], marker=marker, markersize=6,
                 linewidth=2, linestyle=l_style, label=label_text, color=color, alpha=0.8)

        # Отмечаем Baseline (step=0) для BLEU
        baseline_data = subset[subset['step'] == 0]
        if not baseline_data.empty:
            b_bleu = baseline_data.iloc[0]['bleu']
            # Горизонтальная линия бейзлайна
            ax1.axhline(y=b_bleu, color=color, linestyle=':', alpha=0.6, linewidth=1.5, zorder=1)
            # Выделяем стартовую точку
            ax1.plot(0, b_bleu, marker='d', color=color, markersize=9, markeredgecolor='black', zorder=4)

        max_bleu_idx = subset['bleu'].idxmax()
        max_bleu_step = subset.loc[max_bleu_idx, 'step']
        max_bleu_val = subset.loc[max_bleu_idx, 'bleu']
        ax1.plot(max_bleu_step, max_bleu_val, marker='*', color='gold',
                 markersize=15, markeredgecolor='black', zorder=5)

        ax2.plot(subset['step'], subset['chrf'], marker=marker, markersize=6,
                 linewidth=2, linestyle=l_style, label=label_text, color=color, alpha=0.8)

        # Отмечаем Baseline (step=0) для chrF++
        if not baseline_data.empty:
            b_chrf = baseline_data.iloc[0]['chrf']
            # Горизонтальная линия бейзлайна
            ax2.axhline(y=b_chrf, color=color, linestyle=':', alpha=0.6, linewidth=1.5, zorder=1)
            # Выделяем стартовую точку
            ax2.plot(0, b_chrf, marker='d', color=color, markersize=9, markeredgecolor='black', zorder=4)

        max_chrf_idx = subset['chrf'].idxmax()
        max_chrf_step = subset.loc[max_chrf_idx, 'step']
        max_chrf_val = subset.loc[max_chrf_idx, 'chrf']
        ax2.plot(max_chrf_step, max_chrf_val, marker='*', color='gold',
                 markersize=15, markeredgecolor='black', zorder=5)

for ax in [ax1, ax2]:
    ax.grid(True, linestyle=':', alpha=0.8, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax1.set_title('BLEU Score Progression', fontweight='bold', pad=5)
ax1.set_ylabel('BLEU')

# Создаем кастомные элементы для легенды
handles, labels = ax1.get_legend_handles_labels()
baseline_marker = plt.Line2D([0], [0], color='gray', marker='d', linestyle=':', markersize=9, markeredgecolor='black', label='Baseline (step=0)')
max_marker = plt.Line2D([0], [0], color='white', marker='*', markerfacecolor='gold', markersize=15, markeredgecolor='black', label='Best Score')
handles.extend([baseline_marker, max_marker])
labels.extend(['Baseline (step=0)', 'Best Score'])

ax1.legend(handles=handles, labels=labels, loc='lower center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)

ax2.set_title('chrF++ Score Progression', fontweight='bold', pad=15)
ax2.set_ylabel('chrF++')
ax2.set_xlabel('Training Steps')

formatter = ticker.FuncFormatter(lambda x, pos: f'{int(x / 1000)}k' if x >= 1000 else f'{int(x)}')
ax2.xaxis.set_major_formatter(formatter)

plt.tight_layout()

output_filename = '../assets/metrics_report.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"График успешно сохранен: {output_filename}")