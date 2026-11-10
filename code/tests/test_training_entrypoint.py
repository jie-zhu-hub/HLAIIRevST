"""Exercise real full-parameter and LoRA training without touching project outputs."""
import contextlib
import io
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import main as training


class TrainingEntrypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.data = pd.DataFrame({
            'HLA_alpha': ['ACDEFGHIKLMNPQR'] * 5,
            'HLA_beta': ['ACDEFGHIKLMNPQRSTVW'] * 5,
            'MT_pep': ['ACDE', 'FGHI', 'KLMN', 'PQRS', 'TVWY'],
            'CDR3': ['CASSF', 'CASSL', 'CASSY', 'CASSQ', 'CASSR'],
            'label': [1, 1, 0, 0, 0],
        })

    def check_training(self, use_lora):
        with tempfile.TemporaryDirectory(prefix='hlaiirevst-train-') as directory:
            work = Path(directory)
            dataset_dir = work / 'data/fixture'
            dataset_dir.mkdir(parents=True)
            for split in ('train', 'val', 'test'):
                self.data.to_csv(dataset_dir / f'{split}_data(-).csv', index=False)
            argv = ['--data_dir', str(work / 'data'), '--data_name', 'fixture',
                    '--output_dir', str(work / 'output'), '--device', 'cpu', '--num_workers', '0',
                    '--epochs', '1', '--batch_size', '2', '--d_model', '8', '--e_layers', '1',
                    '--dim', '4', '--n_heads', '2', '--d_ff', '16', '--hla_orientation', 'alpha_reverse']
            if use_lora:
                argv += ['--use_lora', '--lora_train_projection']
            build_model = training.build_model
            initial = {}

            def capture_initial(args, device):
                model = build_model(args, device)
                initial.update({key: value.detach().clone() for key, value in model.state_dict().items()})
                return model

            with mock.patch.object(training, 'build_model', side_effect=capture_initial), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                training.main(argv)
            output = work / 'output/seed_20220909'
            for key in ('test1_ppvn', 'val_aupr'):
                paths = list(output.glob(f'*_best_{key}_*.pkl'))
                self.assertEqual(len(paths), 1)
                checkpoint = torch.load(paths[0], map_location='cpu', weights_only=True)
                self.assertEqual(checkpoint['selection_key'], key)
                self.assertEqual(checkpoint['epoch'], 1)
                self.assertEqual(checkpoint['use_lora'], use_lora)
                self.assertTrue(np.isfinite(checkpoint['selection_score']))
                state = checkpoint['model_state_dict']
                self.assertTrue(all(torch.isfinite(value).all() for value in state.values()))
                if use_lora:
                    self.assertTrue(any('lora_B' in name and torch.count_nonzero(value) > 0
                                        for name, value in state.items()))
                    for name in initial:
                        if 'src_emb.weight' in name:
                            self.assertTrue(torch.equal(initial[name], state[name]))
                else:
                    self.assertTrue(any(not torch.equal(value, state[name]) for name, value in initial.items()))
            for filename, expected_count in (('train.pkl', 6), ('val.pkl', 5)):
                with (output / filename).open('rb') as handle:
                    labels, predictions, scores = pickle.load(handle)['1']
                self.assertEqual(len(labels), expected_count)
                np.testing.assert_array_equal(predictions, np.asarray(scores) > 0.5)
            self.assertFalse(list(output.glob('*.tmp')))
            pd.testing.assert_frame_equal(pd.read_csv(dataset_dir / 'train_data(-).csv'), self.data)

    def test_full_parameter_training_and_saved_outputs(self):
        self.check_training(False)

    def test_lora_training_updates_adapters_and_preserves_embeddings(self):
        self.check_training(True)


if __name__ == '__main__':
    unittest.main()
