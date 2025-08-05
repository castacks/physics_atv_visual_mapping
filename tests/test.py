import torch
import numpy as np

import pytest

@pytest.fixture(scope="module")
def trig_data():
    x = torch.rand(100, 100)
    y1 = x.cos()
    y2 = (x + 0.5*np.pi).sin()

    return {
        'x':x,
        'y1':y1,
        'y2':y2
    }

def test_1(trig_data):
    x = trig_data['x']
    print(x[0, 0])
    assert x.min() >= -1.
    assert x.max() <= 1.


def test_2(trig_data):
    x = trig_data['x']
    print(x[0, 0])

    assert torch.allclose(trig_data['y1'], trig_data['y2'])