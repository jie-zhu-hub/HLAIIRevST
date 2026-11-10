"""Synthetic checks for inactive parameters and checkpoint-compatible cleanup."""

import contextlib
import io
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import main as training
from Model.lora_utils import LoRALinear, get_trainable_parameter_groups
from Model.SubLayer import PoswiseFeedForwardNet


class ModelCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_feedforward_norm_preserves_original_affine_path_and_state_keys(self):
        layer = PoswiseFeedForwardNet(torch.device('cpu'), 8, 16)
        inputs = torch.randn(2, 5, 8)
        expected = F.layer_norm(layer.fc(inputs) + inputs, (8,),
                                torch.ones(8), torch.zeros(8))
        torch.testing.assert_close(layer(inputs), expected, rtol=0, atol=0)
        self.assertEqual(set(layer.state_dict()), {'fc.0.weight', 'fc.2.weight'})

    def test_optimizer_parameters_receive_gradients_single_and_multi_layer(self):
        for layers in (1, 2):
            for use_lora in (False, True):
                with self.subTest(layers=layers, use_lora=use_lora):
                    args = training.build_arg_parser().parse_args([
                        '--d_model', '8', '--e_layers', '1', '--dim', '4',
                        '--n_heads', '2', '--d_ff', '16', '--dropout', '0',
                        '--window_size', '3', '--interact_layers', str(layers)])
                    model = training.build_model(args, torch.device('cpu'))
                    if use_lora:
                        with contextlib.redirect_stdout(io.StringIO()):
                            training.apply_lora_to_model(model, training.parse_keyword_string(args.lora_targets),
                                                         r=2, alpha=4, dropout=0)
                        training.mark_only_lora_and_selected_trainable(model, ('projection',))
                        for module in model.modules():
                            if isinstance(module, torch.nn.MultiheadAttention):
                                self.assertIsInstance(module.out_proj, LoRALinear)
                                self.assertTrue(all(p.requires_grad for p in module.out_proj.lora_A.parameters()))
                                self.assertTrue(all(p.requires_grad for p in module.out_proj.lora_B.parameters()))
                    inputs = [torch.randint(1, 21, (2, length)) for length in (15, 19, 25, 30)]
                    F.cross_entropy(model(*inputs)[0], torch.tensor([0, 1])).backward()
                    optimizer_params = {id(p) for p in get_trainable_parameter_groups(model)}
                    missing = [name for name, p in model.named_parameters()
                               if id(p) in optimizer_params and p.grad is None]
                    self.assertEqual(missing, [])
                    inactive = list(model.hla_concat_proj.parameters())
                    inactive += list(model.ternaryInteraction.layers[-1].hla_ffn.parameters())
                    if use_lora:
                        for module in model.modules():
                            if isinstance(module, torch.nn.MultiheadAttention):
                                inactive += list(module.out_proj.lora_A.parameters())
                                inactive += list(module.out_proj.lora_B.parameters())
                    self.assertTrue(all(id(p) not in optimizer_params for p in inactive))
                    if layers > 1:
                        earlier = model.ternaryInteraction.layers[0].dual_att.W_Q_v
                        parameter = earlier.lora_B.weight if use_lora else earlier.weight
                        self.assertIsNotNone(parameter.grad)
                    binary = model.binaryInteraction.layers[-1].dual_att.W_Q_v
                    parameter = binary.lora_B.weight if use_lora else binary.weight
                    self.assertIsNotNone(parameter.grad)

    def test_removed_subset_arguments_are_rejected(self):
        parser = training.build_arg_parser()
        for option, value in [('--train_fraction', '1.0'), ('--train_subset_seed', '42')]:
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args([option, value])


if __name__ == '__main__':
    unittest.main()
