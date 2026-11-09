"""Tail padding preserves the original CUDA encoder shape and sample identities."""
import numpy as np
import torch

from scripts.diagnose_teacher_forced_readout import initialize_fixed_batch


def test_tail_padding_preserves_all_three_inputs_and_output_order():
    class RecordingInitializer:
        def initialize(self, images, actions, intervals):
            assert images.shape[0] == actions.shape[0] == intervals.shape[0] == 32
            # The last padded entries must duplicate all inputs from the same window.
            for values in (images, actions, intervals):
                torch.testing.assert_close(values[12:], values[11:12].expand_as(values[12:]))
            q = images[:, 0, 0, 0, :2].double()
            p = torch.cat((actions[:, 0], intervals[:, 0]), -1).double()
            return q, p, None, None

    images = [torch.tensor([[[[float(i), float(i + 1)]]]]) for i in range(12)]
    actions = [np.array([[i + .25]]) for i in range(12)]
    intervals = [np.array([[i + .5]]) for i in range(12)]
    features = initialize_fixed_batch(RecordingInitializer(), images, actions, intervals, 'cpu', 32)
    np.testing.assert_array_equal(features, np.array([[i, i + 1, i + .25, i + .5] for i in range(12)]))
    assert features.shape == (12, 4)
