import numpy as np
import pandas as pd
import pytest

from src.forecast_intervals import add_prediction_bounds


def test_bounds_use_holdout_errors_and_clip_negative_demand():
    source = pd.DataFrame({"yhat": [2.0, 20.0]})
    result = add_prediction_bounds(source, [0, 10, 20], [0, 6, 10])
    assert result.yhat_lower.tolist() == [0, 10]
    assert result.yhat_upper.tolist() == [12, 30]
    assert "yhat_lower" not in source
    assert (result.yhat_lower <= result.yhat).all()
    assert (result.yhat <= result.yhat_upper).all()


@pytest.mark.parametrize("actual,predicted", [([], []), ([np.nan], [1])])
def test_invalid_calibration_is_rejected(actual, predicted):
    with pytest.raises(ValueError):
        add_prediction_bounds(pd.DataFrame({"yhat": [1]}), actual, predicted)
