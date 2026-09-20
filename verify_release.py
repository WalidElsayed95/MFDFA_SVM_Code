"""Read-only checks of saved final results; does not train models or overwrite outputs."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score

ROOT = Path(__file__).resolve().parent


def main():
    for path in ROOT.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    result = ROOT / 'results'
    provenance = json.loads((result / 'software_and_input_provenance.json').read_text())
    feature_path = ROOT / 'data' / 'mfdfa_features.csv'
    digest = hashlib.sha256(feature_path.read_bytes()).hexdigest()
    assert digest == provenance['input_sha256'], 'Input CSV hash mismatch'
    features = pd.read_csv(feature_path)
    selection = pd.read_csv(ROOT / 'data' / 'selected_records.csv')
    assert len(features) == len(selection) == 28022
    for column in ('serial_no', 'station', 'label'):
        assert features[column].astype(str).tolist() == selection[column].astype(str).tolist(), column
    predictions = pd.read_csv(result / 'nested_oof_predictions.csv')
    assert len(predictions) == 28022
    assert predictions.serial_no.nunique() == 28022
    assert predictions.station.nunique() == 9
    folds = pd.read_csv(result / 'nested_fold_results.csv')
    assert len(folds) == 90
    assert folds.groupby('station').outer_fold.nunique().eq(10).all()
    saved = json.loads((result / 'aggregate_metrics.json').read_text())
    metrics = []
    for _, rows in predictions.groupby('station'):
        y, p = rows.label_true, rows.label_predicted
        metrics.append([accuracy_score(y,p), balanced_accuracy_score(y,p),
                        precision_score(y,p,average='macro'), f1_score(y,p,average='macro')])
    keys = ['station_macro_accuracy','station_macro_balanced_accuracy',
            'station_macro_precision','station_macro_f1']
    for key, value in zip(keys, np.mean(metrics,axis=0)):
        assert np.isclose(value,saved[key],rtol=0,atol=1e-12), key
    import mfdfa
    sample = np.sin(np.linspace(0, 30, 3000)) + 0.05 * np.cos(np.linspace(0, 170, 3000))
    mfdfa_result = mfdfa.MFDFA(sample, np.arange(-5.0, 6.0))
    assert all(np.all(np.isfinite(value)) for value in mfdfa_result.values())
    assert np.isclose(mfdfa_result['D0'], 1.0, rtol=0, atol=1e-12)
    from fonts import font
    font(16); font(16,True)
    print('PASS: Python syntax, corrected feature hash, selection manifest,')
    print('28,022 unique OOF rows, nine stations, 90 outer folds, saved macro')
    print('metrics, MFDFA numerical smoke test, and portable font loading.')
    print('NOT TESTED: full raw-waveform re-extraction, retraining, map rendering,')
    print('or complete figure regeneration.')


if __name__ == '__main__':
    main()
