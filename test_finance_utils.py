import numpy as np
import pandas as pd
import pytest
from finance_utils import *

def test_logret():
    sample1 = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    result1 = logret(sample1, to_cols=['logret1', 'logret2'], only_new=True)
    target1 = pd.DataFrame(
        data={
            'logret1': [np.log(2)-np.log(1)],
            'logret2': [np.log(4)-np.log(3)]},
        index=[1])
    assert result1.equals(target1)
test_logret()

