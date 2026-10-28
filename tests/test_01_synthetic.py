"""Run the EEG notebook's evaluation cells on synthetic epochs with a known answer.

20 synthetic people x 3 runs x 15 trials, 64 channels, 3 s at 160 Hz, average-referenced (rank 63, like the real data).
Imagined right-hand trials lower 8-30 Hz power over one channel group, left-hand trials over another. Each person
gets their own random channel mixing and gain, so a decoder trained on other people transfers poorly unless their
covariances are aligned. Checks: within-subject decoding finds the signal; cross-subject CSP is weaker than within;
Euclidean alignment helps, including when computed from a separate calibration run; the exploratory channel groups
have the intended electrodes and locate the planted signal (channels 0-7, FC5-C5, all sensorimotor); all tables and
figures are written. Usage: python tests/test_01_synthetic.py"""
import io, json, contextlib, sys, tempfile, warnings
from types import SimpleNamespace
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
CHANNELS = ('FC5 FC3 FC1 FCz FC2 FC4 FC6 C5 C3 C1 Cz C2 C4 C6 CP5 CP3 CP1 CPz CP2 CP4 CP6 Fp1 Fpz Fp2 AF7 AF3 AFz AF4 AF8 '
            'F7 F5 F3 F1 Fz F2 F4 F6 F8 FT7 FT8 T7 T8 T9 T10 TP7 TP8 P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2 Iz').split()
assert len(CHANNELS) == n_ch
ns = {'display': lambda *a, **k: None, 'X': X, 'y': y, 'groups': groups, 'run_ids': run_ids,
      'example_raw': SimpleNamespace(ch_names=CHANNELS)}
with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
    warnings.simplefilter('ignore')
    exec(config, ns)
    ns.update(X=X, y=y, groups=groups, run_ids=run_ids)
    for i in (6, 7, 8, 9, 11, 12, 15):
        exec(code[i], ns)
s = ns['summary'].set_index(['analysis', 'model'])
print(s[['mean_balanced_accuracy', 'ci95_low', 'ci95_high', 'wilcoxon_p_vs_0.5', 'people_above_chance_binomial']].round(3).to_string())
print(ns['comparisons'].round(3).to_string(index=False))
e = ns['explore_summary'].set_index(['analysis', 'model'])
print(e[['mean_balanced_accuracy', 'wilcoxon_p_vs_0.5', 'people_above_chance_binomial']].round(3).to_string())
print(ns['explore_pairs'].round(3).to_string(index=False))
CAL = ns['CAL']; regions = ns['regions']
m = lambda a, b: s.loc[(a, b), 'mean_balanced_accuracy']
checks = {
    'within-subject CSP finds the planted signal (> .80)': m('within-subject', 'csp') > .80,
    'cross-subject CSP is weaker than within-subject': m('cross-subject', 'csp') < m('within-subject', 'csp'),
    'Euclidean alignment improves cross-subject CSP': m('cross-subject', 'csp_aligned') > m('cross-subject', 'csp'),
    'aligned data keep their rank (no blown-up null direction)': np.all(np.isfinite(ns['X_aligned'])),
    'Euclidean alignment from a separate calibration run beats no alignment on the remaining runs':
        e.loc[(CAL, 'csp_aligned_calibration_run'), 'mean_balanced_accuracy'] > e.loc[(CAL, 'csp'), 'mean_balanced_accuracy'],
    'channel groups locate the planted signal: sensorimotor beats posterior and frontal by > .1':
        e.loc[('channels: sensorimotor', 'csp_aligned'), 'mean_balanced_accuracy']
        > .1 + max(e.loc[('channels: posterior', 'csp_aligned'), 'mean_balanced_accuracy'], e.loc[('channels: frontal', 'csp_aligned'), 'mean_balanced_accuracy']),
    'calibration analysis scores only runs 8 and 12 (30 trials per person)':
        set(ns['explore'][ns['explore'].analysis == CAL].n_trials) == {30},
    'channel groups: 21 sensorimotor, 18 posterior, 17 frontal, no overlap, no temporal channels':
        [len(v) for v in regions.values()] == [21, 18, 17] and len(set(sum(regions.values(), []))) == 56
        and not any(CHANNELS[i].upper().startswith(('T', 'FT')) for i in sum(regions.values(), [])),
    'all outputs written': {'cross_subject_by_person.csv', 'within_subject_by_person.csv', 'summary.csv', 'paired_comparisons.csv',
                            'accuracy_by_person.png', 'settings.json', 'exploratory_by_person.csv', 'exploratory_summary.csv',
                            'exploratory_paired_comparisons.csv', 'exploratory_by_person.png', 'exploratory_settings.json'}
                           <= {p.name for p in TMP.iterdir()},
    'no automatic download in the last cell': 'files.download' not in code[15],
}
for name, ok in checks.items():
    print(('PASS ' if ok else 'FAIL ') + name)
sys.exit(0 if all(checks.values()) else 1)
