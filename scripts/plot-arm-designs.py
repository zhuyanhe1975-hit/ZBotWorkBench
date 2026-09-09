"""Render static research figures after design:screen and design:validate (matplotlib)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out = Path('results/arm-design')
validation = json.loads((out / 'validation.json').read_text())
ids = ['design_6_22222', 'design_7_232222', 'design_7_221222', 'design_7_222222']
labels = ['6R orthogonal', '7R candidate A', '7R candidate B', '7R orthogonal']
colors = ['#64748b', '#0284c7', '#059669', '#a78bfa']
rows = {r['id']: r for r in validation['rows']}
shared = {r['id']: r for r in validation['commonTasks']}
fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout='constrained')
metrics = [
    ([rows[k]['success'] for k in ids], 'New position + orientation tasks', 'Solved / 64 (higher is better)'),
    ([shared[k]['dexterityP25'] for k in ids], 'Dexterity on 28 shared tasks', 'P25 normalized minimum singular value'),
    ([shared[k]['payloadTorqueP95NmPerKg'] for k in ids], 'Incremental payload torque', 'P95 maximum joint torque [Nm/kg]'),
]
for ax, (values, title, ylabel) in zip(axes, metrics):
    bars = ax.bar(labels, values, color=colors, width=.65)
    ax.bar_label(bars, labels=[f'{v:.3f}' if v < 5 else str(v) for v in values], padding=4)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis='x', labelrotation=25, labelsize=8)
    ax.set_ylim(0, max(values)*1.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.15)
fig.suptitle('ZBot: six / seven module design screening', fontsize=16)
fig.supxlabel('Provisional geometry and static poses. Torque excludes arm self-weight; not rated payload capacity.', fontsize=10)
fig.savefig(out / 'comparison.png', dpi=180)
fig.savefig(out / 'comparison.svg')
plt.close(fig)

poses = json.loads((out / 'plot-data.json').read_text())
fig = plt.figure(figsize=(12, 4.8), layout='constrained')
for i, pose in enumerate(poses):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    points = np.array([pose['root']] + [j['position'] for j in pose['joints']] + [pose['tip']])
    ax.plot(*points.T, '-o', color=colors[[1,2,0][i]], markersize=4, linewidth=2)
    for j in pose['joints']:
        ax.quiver(*j['position'], *np.array(j['axis'])*.05, color='#e11d48', linewidth=.7)
    ax.scatter(*pose['target']['position'], marker='*', s=100, color='#f59e0b')
    ax.set(xlim=(-.15,.55), ylim=(-.35,.35), zlim=(0,.7), xlabel='X / m', ylabel='Y / m', zlabel='Z / m')
    ax.set_box_aspect((1,1,1))
    ax.view_init(elev=22, azim=-55)
    ax.set_title(['7R candidate A', '7R candidate B', '6R orthogonal'][i])
fig.suptitle('Different chains reaching the same validated pose', fontsize=15)
fig.supxlabel('Joint-center schematic only (not CAD surfaces). Red: joint axes. Star: shared target.', fontsize=10)
fig.savefig(out / 'same-target.png', dpi=180, bbox_inches='tight', pad_inches=.25)
plt.close(fig)
