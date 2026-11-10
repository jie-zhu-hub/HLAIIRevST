"""Regression checks for training loaders, thresholding, and checkpoint safety."""

import contextlib
import gc
import io
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings
import zipfile

import numpy as np
import pandas as pd
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import main as training


class TrainingCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        source = pd.read_csv(CODE.parent / 'data/hlaii/test_data(+).csv')
        cls.data = pd.concat([source[source.label == 1].head(2), source[source.label == 0].head(3)], ignore_index=True)

    def test_threshold_rule_matches_python_comparisons(self):
        scores = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
        for threshold in [0.0, 0.5, np.float64(0.8), 1.0]:
            expected = [int(float(score) > float(threshold)) for score in scores]
            np.testing.assert_array_equal(training.transfer(scores, threshold), expected)
        self.assertEqual(training.transfer([]).shape, (0,))

    def test_removed_threshold_search_flag_is_rejected(self):
        result = subprocess.run([sys.executable, '-B', str(CODE / 'main.py'), '--mdrt'],
                                cwd=CODE, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn('unrecognized arguments: --mdrt', result.stderr)

    def test_pretrained_load_accepts_legacy_numpy_metadata(self):
        with tempfile.TemporaryDirectory(prefix='hlaiirevst-legacy-') as directory:
            original = Path(directory) / 'original.pkl'
            legacy = Path(directory) / 'legacy.pkl'
            source = torch.nn.Linear(2, 2)
            torch.save({'model_state_dict': source.state_dict(), 'threshold': np.float64(0.6)}, original)
            with zipfile.ZipFile(original) as reader, zipfile.ZipFile(legacy, 'w') as writer:
                for entry in reader.infolist():
                    data = reader.read(entry.filename)
                    if entry.filename.endswith('data.pkl'):
                        data = data.replace(b'numpy._core.multiarray', b'numpy.core.multiarray')
                    writer.writestr(entry, data)
            target = torch.nn.Linear(2, 2)
            with contextlib.redirect_stdout(io.StringIO()):
                training.load_pretrained_weights(target, str(legacy), torch.device('cpu'), strict=True)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(source.parameters(), target.parameters())))

    def test_labels_are_copied_without_changing_values(self):
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            tensors = training.make_data(self.data, 25, 15, 19, 30)
        self.assertFalse(any('not writable' in str(item.message) for item in captured))
        np.testing.assert_array_equal(tensors[-1].numpy(), self.data.label.to_numpy())
        tensors[-1][0] = 0
        self.assertEqual(self.data.label.iloc[0], 1)

    def test_windows_workers_and_singleton_batch(self):
        with tempfile.TemporaryDirectory(prefix='hlaiirevst-loader-') as directory:
            self.data.to_csv(Path(directory) / 'test_data(+).csv', index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                _, alpha, beta, peptide, tcr, labels, loader = training.data_with_loader(
                    type_='test', datapath=directory, batch_size=2, num_workers=2,
                )
            try:
                batches = list(loader)
                self.assertEqual([tuple(t.shape) for t in batches[-1]], [(1, 15), (1, 19), (1, 25), (1, 30), (1,)])
                for index, expected in enumerate([alpha, beta, peptide, tcr, labels]):
                    self.assertTrue(torch.equal(torch.cat([batch[index] for batch in batches]), expected))
            finally:
                del loader
                gc.collect()

    def checkpoint_fixture(self, directory):
        previous = Path(directory) / 'previous.pkl'
        previous.write_bytes(b'previous complete checkpoint')
        tracker = {'best': 0.5, 'epoch': 1, 'threshold': 0.5, 'model_path': str(previous),
                   'display_name': 'Validation AUPR Max', 'file_metric_name': 'AUPR', 'is_int': False}
        model = torch.nn.Linear(2, 2)
        args = SimpleNamespace(use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.05,
                               lora_targets='', lora_train_projection=False)
        return previous, tracker, model, args

    def test_failed_save_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix='hlaiirevst-save-') as directory:
            previous, tracker, model, args = self.checkpoint_fixture(directory)
            snapshot = tracker.copy()

            def fail_after_partial_write(payload, path):
                Path(path).write_bytes(b'partial checkpoint')
                raise OSError('simulated disk failure')

            with mock.patch.object(training.torch, 'save', side_effect=fail_after_partial_write):
                with self.assertRaises(OSError):
                    training.update_best_checkpoint({'val_aupr': tracker}, 'val_aupr', 0.75, 2,
                                                    model, 0.5, str(Path(directory) / 'model'), args)
            self.assertEqual(previous.read_bytes(), b'previous complete checkpoint')
            self.assertEqual(tracker, snapshot)
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    def test_failed_publish_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix='hlaiirevst-publish-') as directory:
            previous, tracker, model, args = self.checkpoint_fixture(directory)
            snapshot = tracker.copy()
            with mock.patch.object(training.os, 'replace', side_effect=OSError('simulated rename failure')):
                with self.assertRaises(OSError):
                    training.update_best_checkpoint({'val_aupr': tracker}, 'val_aupr', 0.75, 2,
                                                    model, 0.5, str(Path(directory) / 'model'), args)
            self.assertEqual(previous.read_bytes(), b'previous complete checkpoint')
            self.assertEqual(tracker, snapshot)
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    def test_successful_save_preserves_schema_and_removes_old(self):
        with tempfile.TemporaryDirectory(prefix='hlaiirevst-success-') as directory:
            previous, tracker, model, args = self.checkpoint_fixture(directory)
            args.threshold = np.float64(0.5)
            args.extra_integer = np.int64(2)
            with contextlib.redirect_stdout(io.StringIO()):
                changed = training.update_best_checkpoint({'val_aupr': tracker}, 'val_aupr', np.float64(0.75), 2,
                                                         model, np.float64(0.5), str(Path(directory) / 'model'), args)
            self.assertTrue(changed)
            self.assertFalse(previous.exists())
            self.assertTrue(tracker['model_path'].endswith('_best_val_aupr_epoch2_AUPR0p750000.pkl'))
            checkpoint = torch.load(tracker['model_path'], map_location='cpu')
            self.assertEqual(set(checkpoint), {'epoch', 'selection_key', 'selection_name', 'selection_score',
                                             'threshold', 'model_state_dict', 'use_lora',
                                             'lora_config', 'lora_state_dict', 'args'})
            self.assertEqual(checkpoint['threshold'], 0.5)
            self.assertIs(type(checkpoint['selection_score']), float)
            self.assertIs(type(checkpoint['args']['threshold']), float)
            self.assertIs(type(checkpoint['args']['extra_integer']), int)
            restored = torch.nn.Linear(2, 2)
            restored.load_state_dict(checkpoint['model_state_dict'], strict=True)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(model.parameters(), restored.parameters())))
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
