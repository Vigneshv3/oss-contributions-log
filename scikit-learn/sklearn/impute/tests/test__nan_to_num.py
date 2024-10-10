import numpy as np
from sklearn.impute import _nan_to_num

def test_nan_to_num_dtype_preservation():
    X = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    result = _nan_to_num(X)
    assert result.dtype == X.dtype
