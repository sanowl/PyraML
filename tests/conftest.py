import pytest
import numpy as np
from pyraml.core import Tensor

@pytest.fixture
def random_tensor():
    return Tensor(np.random.randn(3, 3))

@pytest.fixture
def grad_tensor():
    return Tensor(np.random.randn(3, 3), requires_grad=True)
