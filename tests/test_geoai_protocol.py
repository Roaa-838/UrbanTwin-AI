import numpy as np
import pytest

from urbantwin.common import require_real_provenance
from urbantwin.geoai.train import select_supported_classes


def test_class_eligibility_cannot_change_with_test_labels():
    labels = np.array([10, 10, 10, 50, 50, 50, 80, 80])
    train = np.array([0, 1, 2, 3, 4, 5])
    expected = select_supported_classes(labels, train, min_train_per_class=3)

    altered_test = labels.copy()
    altered_test[6:] = [20, 30]

    assert expected == [10, 50]
    assert select_supported_classes(altered_test, train, min_train_per_class=3) == expected


def test_a_bare_real_label_is_not_accepted_as_real_provenance():
    with pytest.raises(ValueError, match="schema_version"):
        require_real_provenance({"data_kind": "real"})
