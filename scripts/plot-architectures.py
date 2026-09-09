"""Run after compare-architectures.ts; requires matplotlib (static research figures)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out = Path('results/architecture-comparison')
summary = json.loads((out / 'summary.json').read_text())
ids = ['zbot', 'snake', 'yumi_style']
labels = ['ZBot orthogonal', 'Pitch/yaw snake', 'YuMi-inspired*']
colors = ['#0284c7', '#059669', '#a855f7']
fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
x = np.arange(3)
for ax, limit in zip(axes[0], [90, 180]):
    rows = {r['id']: r for r in summary['rows'] if r['limit'] == limit}
    bars1 = ax.bar(x-.18, [rows[k]['positionSolved']/128*100 for k in ids], .34, color=colors, alpha=.4, label='Position only')
    bars2 = ax.bar(x+.18, [rows[k]['poseSolved']/128*100 for k in ids], .34, color=colors, label='Position + orientation')
    ax.bar_label(bars1, fmt='%.1f', padding=3, fontsize=9)
    ax.bar_label(bars2, fmt='%.1f', padding=3, fontsize=9)
    ax.set(xticks=x, xticklabels=labels, ylim=(0,130), ylabel='Targets solved / %', title=f'All joints limited to +/-{limit} degrees')
    ax.legend(loc='upper left', fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', alpha=.15)
common = next(c for c in summary['common'] if c['limit']==180)
rows = {r['id']:r for r in common['rows']}
for ax, metric, title in zip(axes[1], ['dexterityP25','payloadTorqueP95'], ['Dexterity P25 (higher is better)', 'Payload torque P95 [Nm/kg] (lower is better)']):
    bars = ax.bar(labels, [rows[k][metric] for k in ids], color=colors, width=.6)
    ax.bar_label(bars, fmt='%.3f', padding=4)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0,max(rows[k][metric] for k in ids)*1.2)
    ax.set_xlabel(f'Same {len(common["indices"])} feasible pose targets; +/-180 deg')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15)
fig.suptitle('7 modules each | neutral envelope: diameter 100 mm x length 106 mm', fontsize=15)
fig.supxlabel('*YuMi-inspired axis abstraction, not ABB product geometry or performance. Static envelope checks only.', fontsize=10)
fig.savefig(out/'comparison.png', dpi=170)
fig.savefig(out/'comparison.svg')
plt.close(fig)
