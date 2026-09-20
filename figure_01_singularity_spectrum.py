"""Plot one stored MFDFA spectrum without smoothing or recomputing features.

Default record: zero-based CSV row 0 (BAE, serial_no 0, label 1).
The stored D0 field is checked against the spectrum maximum.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--row', type=int, default=0)
    args = parser.parse_args()
    if args.row < 0:
        raise ValueError('Row index must be nonnegative.')
    with (ROOT / 'data' / 'mfdfa_features.csv').open(newline='') as handle:
        record = next(r for i, r in enumerate(csv.DictReader(handle)) if i == args.row)
    def array(key):
        return np.asarray(json.loads(record[key]), dtype=float)
    alpha, f, h, tau = [array(k) for k in ('alpha', 'f_alpha', 'h(q)', 'tau(q)')]
    assert len(alpha) == len(f) == len(h) == len(tau) == 11
    assert np.isfinite(alpha).all() and np.isfinite(f).all()
    peak = int(np.argmax(f))
    a0, f0 = alpha[peak], f[peak]
    assert peak == 5 and np.isclose(tau[peak], -1) and np.isclose(f0, 1)
    assert np.isclose(f0, float(record['D0']))
    order = np.argsort(alpha)
    x, y = alpha[order], f[order]
    amin, amax, fl, fr = x[0], x[-1], y[0], y[-1]
    width = amax - amin
    assert np.isclose(a0, float(record['alpha_0']))
    assert np.isclose(width, float(record['Delta_alpha']))
    assert np.isclose(fr - fl, float(record['Delta_f']))
    plt.rcParams.update({'font.size': 12, 'mathtext.fontset': 'stix', 'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(figsize=(9.3, 6.8))
    fig.subplots_adjust(left=.12, right=.96, bottom=.13, top=.94)
    ax.plot(x, y, '-o', color='#d95f02', lw=1.8, ms=4)
    ax.set(xlim=(amin-.16*width, amax+.23*width), ylim=(-.37, 1.19),
           xlabel=r'$\alpha$', ylabel=r'$f(\alpha)$')
    ax.axhline(0, color='.7', lw=.7)
    ax.plot([a0, a0], [0, f0], '--', color='.2', lw=1)
    ax.plot([amin-.16*width, a0], [f0, f0], '--', color='.2', lw=1)
    for a, val in ((amin, fl), (amax, fr)):
        ax.plot([a, a], [-.27, val], ':', color='.6', lw=.8)
    ax.text(amin-.12*width, 1.065, r'$D_0=f(\alpha_0)=1$')
    ax.text(a0, -.035, r'$\alpha_0$', ha='center', va='top')
    ax.text(amin, -.035, r'$\alpha_{\min}$', ha='center', va='top')
    ax.text(amax, -.035, r'$\alpha_{\max}$', ha='center', va='top')
    ax.annotate(r'$f(\alpha_{\min})$', (amin, fl), xytext=(7,-22), textcoords='offset points')
    ax.annotate(r'$f(\alpha_{\max})$', (amax, fr), xytext=(-12,15),
                textcoords='offset points', ha='right')
    def span(left, right, level, label):
        ax.annotate('', (left,level), (right,level),
                    arrowprops={'arrowstyle':'<->','color':'.2','lw':1})
        ax.text((left+right)/2, level-.025, label, ha='center', va='top')
    span(amin, a0, -.13, r'$\Delta\alpha_{\mathrm{left}}$')
    span(a0, amax, -.13, r'$\Delta\alpha_{\mathrm{right}}$')
    span(amin, amax, -.27, r'$\Delta\alpha=\alpha_{\max}-\alpha_{\min}$')
    # Delta f is a vertical endpoint difference, not the length of a diagonal.
    bx = amax+.08*width
    ax.plot([amin,bx], [fl,fl], ':', color='.65', lw=.8)
    ax.plot([amax,bx], [fr,fr], ':', color='.65', lw=.8)
    ax.annotate('', (bx,fl), (bx,fr), arrowprops={'arrowstyle':'<->','color':'.2'})
    ax.text(bx+.02*width, (fl+fr)/2, r'$\Delta f$', va='center')
    ax.spines[['top','right']].set_visible(False)
    output = ROOT / 'generated_figures' / 'figure_01_singularity_spectrum'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.png'), dpi=300)
    fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)
    provenance = {'csv_row_index_zero_based': args.row, 'record': record,
                  'plotted_peak_alpha': float(a0), 'plotted_peak_height': float(f0),
                  'H_at_q0': float(h[5]),
                  'note': 'Stored alpha and f_alpha plotted directly; straight segments connect samples. '
                          'The stored D0 equals f(alpha_0); no data were modified.'}
    output.with_suffix('.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(f'Row {args.row}: station={record["station"]}, serial={record["serial_no"]}')
    print(f'alpha_0={a0}, H(0)={h[5]}, f(alpha_0)={f0}, CSV D0={record["D0"]}')


if __name__ == '__main__':
    main()
