"""Run with: python -B -m unittest discover -s code/tests -v"""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
spec = importlib.util.spec_from_file_location('hlaiirevst_evaluation', CODE / 'test.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
from main import make_data
from predict_hlaiirevst import build_model as prediction_build_model


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        source = pd.read_csv(CODE.parent / 'data/hlaii/test_data(+).csv')
        cls.data = pd.concat([source[source.label == 1].head(2), source[source.label == 0].head(3)], ignore_index=True)

    def test_loader_matches_training_and_singleton_batch(self):
        dataset = evaluation.MyDataSet.from_dataframe(self.data, evaluation.VOCAB)
        expected = make_data(self.data, 25, 15, 19, 30)
        for actual, reference in zip(dataset.tensors, expected):
            self.assertTrue(torch.equal(actual, reference))
        batches = list(torch.utils.data.DataLoader(dataset, batch_size=2))
        self.assertEqual([tuple(t.shape) for t in batches[-1]], [(1, 15), (1, 19), (1, 25), (1, 30), (1,)])
        capital_label = self.data.rename(columns={'label': 'Label'})
        self.assertTrue(torch.equal(evaluation.MyDataSet.from_dataframe(capital_label, evaluation.VOCAB).tensors[-1], expected[-1]))

    def test_loader_rejects_invalid_input(self):
        cases = [self.data.drop(columns=['label']), self.data.iloc[:0], self.data.drop(columns=['HLA_beta'])]
        for column, value in [('MT_pep', 'ACZ'), ('MT_pep', None), ('MT_pep', 'A' * 26), ('label', 0.5), ('label', np.nan)]:
            data = self.data.copy()
            data[column] = value
            cases.append(data)
        for data in cases:
            with self.subTest(columns=list(data.columns)), self.assertRaises(ValueError):
                evaluation.MyDataSet.from_dataframe(data, evaluation.VOCAB)

    def test_metrics_and_undefined_values(self):
        metrics = evaluation.calculate_metrics([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], 0.7)
        self.assertAlmostEqual(metrics['roc_auc'], 0.75)
        self.assertAlmostEqual(metrics['aupr'], 19 / 24)
        self.assertEqual(metrics['ppvn'], 0.5)
        self.assertEqual([metrics[k] for k in ['tn', 'fp', 'fn', 'tp']], [1, 1, 1, 1])
        single_class = evaluation.calculate_metrics([0], [0.1], 0.5)
        self.assertIsNone(single_class['roc_auc'])
        self.assertIsNone(single_class['aupr'])
        self.assertIsNone(single_class['ppvn'])
        for scores, threshold in [([np.nan], 0.5), ([1.1], 0.5), ([0.1], float('nan'))]:
            with self.assertRaises(ValueError):
                evaluation.calculate_metrics([0], scores, threshold)

    def check_roundtrip(self, use_lora, raw_state=False):
        with tempfile.TemporaryDirectory(prefix='hlaiirevst-eval-') as directory:
            work = Path(directory)
            csv_path, checkpoint_path = work / 'input.csv', work / 'checkpoint.pkl'
            self.data.to_csv(csv_path, index=False)
            argv = ['--input', str(csv_path), '--checkpoint', str(checkpoint_path), '--output', str(work / 'results'), '--device', 'cpu']
            architecture = {'d_model': 8, 'e_layers': 1, 'dim': 4, 'n_heads': 2, 'd_ff': 16, 'dropout': 0.0, 'window_size': '3', 'batch_size': 2, 'hla_orientation': 'forward'}
            if raw_state:
                for name in ['d_model', 'e_layers', 'dim', 'n_heads', 'd_ff', 'dropout']:
                    argv.extend(['--' + name, str(architecture[name])])
                argv.extend(['--batch_size', '2'])
            args = evaluation.build_arg_parser().parse_args(argv)
            model_args = evaluation.checkpoint_args_to_namespace(architecture, args)
            torch.manual_seed(17)
            model = evaluation.build_model(model_args, torch.device('cpu'))
            self.assertEqual(model.binaryInteraction.layers[0].dual_att.window_T, 0.01)
            inference_model = prediction_build_model(model_args, torch.device('cpu'))
            self.assertEqual(inference_model.binaryInteraction.layers[0].dual_att.window_T, 0.01)
            checkpoint = {'args': architecture, 'threshold': np.float64(0.65), 'selection_score': np.float64(0.7), 'selection_key': 'val_aupr', 'use_lora': use_lora, 'lora_config': {'r': 2, 'alpha': 4, 'dropout': 0.0, 'targets': ['projection']}}
            with contextlib.redirect_stdout(io.StringIO()):
                evaluation.maybe_apply_lora(model, checkpoint, args)
            model.eval()
            dataset = evaluation.MyDataSet.from_dataframe(self.data, evaluation.VOCAB)
            loader = torch.utils.data.DataLoader(dataset, batch_size=2)
            labels, expected = evaluation.predict_step(model, loader, torch.device('cpu'))
            checkpoint['model_state_dict'] = model.state_dict()
            torch.save(model.state_dict() if raw_state else checkpoint, checkpoint_path)
            original_csv = csv_path.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                evaluation.main(argv)
            actual = pd.read_csv(work / 'results/hlaiirevst_test_predictions.csv')
            metrics = json.loads((work / 'results/hlaiirevst_test_metrics.json').read_text(encoding='utf-8'))
            np.testing.assert_allclose(actual.predicted_score, expected, rtol=1e-6, atol=1e-7)
            self.assertEqual(len(actual), len(labels))
            self.assertEqual(metrics['threshold'], 0.5 if raw_state else 0.65)
            self.assertEqual(metrics['threshold_source'], 'default_0.5' if raw_state else 'checkpoint')
            self.assertEqual(csv_path.read_bytes(), original_csv)
            with contextlib.redirect_stdout(io.StringIO()):
                evaluation.main(argv + ['--threshold', '1'])
            self.assertEqual(pd.read_csv(work / 'results/hlaiirevst_test_predictions.csv').predicted_label.sum(), 0)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluation.main(argv + ['--output', str(work), '--output_name', 'input.csv'])
            self.assertEqual(csv_path.read_bytes(), original_csv)

    def test_standard_checkpoint_roundtrip(self):
        self.check_roundtrip(False)

    def test_lora_checkpoint_roundtrip(self):
        self.check_roundtrip(True)

    def test_raw_state_dict_roundtrip(self):
        self.check_roundtrip(False, raw_state=True)


if __name__ == '__main__':
    unittest.main()
