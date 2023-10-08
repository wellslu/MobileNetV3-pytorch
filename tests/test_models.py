import unittest

import torch

from src.models import VGG


class TestModels(unittest.TestCase):

    def test_vgg(self):
        m = MobileNetV3()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            y = m(x)

        self.assertListEqual(list(y.size()), [1, 10])


if __name__ == '__main__':
    unittest.main()
