import random
import unittest

import torch

import hamming_sim

PROPERTY_TEST_ITERATIONS = 10


class QuantizedTensorDistanceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(1)
        random.seed(1)

    def assertEqualTensor(self, a, b, msg=""):
        self.assertTrue(a.equal(b), "%s\n%s\nnot\n%s." % (msg, a, b))

    def assertAlmostEqualTensor(self, a, b, msg="", atol=1e-4):
        """
        Asserts that the two tensors are almost equal.

        Torch allclose broadcasts the tensors to the same shape, which can lead to false positives.
        This manifests in two ways:
            1. If the values in the tensors are the same.
                `torch.allclose(torch.zeros((1, 1)), torch.zeros((1,))` returns True.
            2. If either of the tenosors is empty, then the empty value matches all other values.
                `torch.allclose(torch.tensor([]), torch.rand((a, 1)))` returns True.
        To avoid this, we need to check that the shapes are the same.

        PS: If a test wants to raise an error on empty tensors, it should do so explicitly.
        """
        self.assertEqual(a.shape, b.shape, f"{msg}\nShapes differ!")
        self.assertTrue(a.allclose(b, atol=atol), "%s\n%s\nnot\n%s." % (msg, a, b))

    def test_1bit_binary_quantized_similarity(self):
        n_dim = 16
        for _ in range(PROPERTY_TEST_ITERATIONS):
            # tensor1 and tensor2 are packed tensors
            tensor1 = torch.randint(
                0, 256, (random.randint(1, 10), n_dim), dtype=torch.uint8
            )
            tensor2 = torch.randint(
                0, 256, (random.randint(1, 10), n_dim), dtype=torch.uint8
            )
            similarity = hamming_sim.quantized_1bit_tensor_similarity(tensor1, tensor2)
            similarity_pytorch = 1 - (
                torch.cdist(
                    hamming_sim.unpack_tensor(tensor1).float(),
                    hamming_sim.unpack_tensor(tensor2).float(),
                    p=0,
                )
                / (n_dim * 8)
            )
            self.assertAlmostEqualTensor(similarity, similarity_pytorch)

    def test_unpack_tensor_one_bit(self) -> None:
        tensor = torch.tensor([[181], [127]], dtype=torch.uint8)
        unpacked_tensor = hamming_sim.unpack_tensor(tensor)
        self.assertEqualTensor(
            unpacked_tensor,
            torch.tensor(
                [[1, 0, 1, 1, 0, 1, 0, 1], [0, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.float
            ),
        )
