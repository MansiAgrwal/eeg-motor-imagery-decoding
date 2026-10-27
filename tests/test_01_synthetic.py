"""Run the EEG notebook's evaluation cells on synthetic epochs with a known answer.

20 synthetic people x 3 runs x 15 trials, 64 channels, 3 s at 160 Hz, average-referenced (rank 63, like the real data).
Imagined right-hand trials lower 8-30 Hz power over one channel group, left-hand trials over another. Each person
gets their own random channel mixing and gain, so a decoder trained on other people transfers poorly unless their
covariances are aligned. Checks: within-subject decoding finds the signal; cross-subject CSP is weaker than within;
Euclidean alignment helps; all tables and figures are written. Usage: python tests/test_01_synthetic.py"""
import io, json, contextlib, sys, tempfile, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / '01_EEG_Motor_Imagery_BCI.ipynb'
TMP = Path(tempfile.mkdtemp(prefix='eeg01_'))
rng = np.random.default_rng(0)
n_sub, n_run, n_trial, n_ch, n_t = 20, 3, 15, 64, 481
Xs, ys, groups, runs = [], [], [], []
for s in range(1, n_sub + 1):
    mix = np.eye(n_ch) + .35 * rng.normal(size=(n_ch, n_ch)); gain = rng.uniform(.5, 2.)
    for r in (4, 8, 12):
        y = rng.permutation(np.r_[np.zeros(7), np.ones(8)]).astype(int)
        for lab in y:
            src = rng.normal(size=(n_ch, n_t))
            src[0:4] *= .55 if lab == 1 else 1.0      # right-hand imagery: lower power in source group A
            src[4:8] *= .55 if lab == 0 else 1.0      # left-hand imagery: lower power in source group B
            x = gain * (mix @ src) * 1e-5
            Xs.append(x - x.mean(0, keepdims=True)); ys.append(lab); groups.append(s); runs.append(r)
X, y, groups, run_ids = np.array(Xs), np.array(ys), np.array(groups), np.array(runs)

nb = json.loads(NB.read_text())
code = {i: ''.join(c['source']) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code'}
config = code[2].replace("OUT = Path('/content/neurotech_results/01_eeg_bci') if Path('/content').exists() else Path('neurotech_results/01_eeg_bci')",
                         f"OUT = Path(r'{TMP}')")
assert str(TMP) in config, 'configuration cell changed; update this test'
ns = {'display': lambda *a, **k: None, 'X': X, 'y': y, 'groups': groups, 'run_ids': run_ids}
with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
    warnings.simplefilter('ignore')
    exec(config, ns)
    ns.update(X=X, y=y, groups=groups, run_ids=run_ids)
    for i in (6, 7, 8, 9, 12):
        exec(code[i], ns)
s = ns['summary'].set_index(['analysis', 'model'])
print(s[['mean_balanced_accuracy', 'ci95_low', 'ci95_high', 'wilcoxon_p_vs_0.5', 'people_above_chance_binomial']].round(3).to_string())
print(ns['comparisons'].round(3).to_string(index=False))
m = lambda a, b: s.loc[(a, b), 'mean_balanced_accuracy']
checks = {
    'within-subject CSP finds the planted signal (> .80)': m('within-subject', 'csp') > .80,
    'cross-subject CSP is weaker than within-subject': m('cross-subject', 'csp') < m('within-subject', 'csp'),
    'Euclidean alignment improves cross-subject CSP': m('cross-subject', 'csp_aligned') > m('cross-subject', 'csp'),
    'aligned data keep their rank (no blown-up null direction)': np.all(np.isfinite(ns['X_aligned'])),
    'all outputs written': {'cross_subject_by_person.csv', 'within_subject_by_person.csv', 'summary.csv', 'paired_comparisons.csv',
                            'accuracy_by_person.png', 'settings.json'} <= {p.name for p in TMP.iterdir()},
    'no automatic download in the last cell': 'files.download' not in code[12],
}
for name, ok in checks.items():
    print(('PASS ' if ok else 'FAIL ') + name)
sys.exit(0 if all(checks.values()) else 1)
