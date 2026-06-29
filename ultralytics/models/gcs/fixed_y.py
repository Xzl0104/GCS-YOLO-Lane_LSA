"""Compatibility re-export for fixed-y utilities.

Keep dataset and standalone tools importing from ``ultralytics.utils.gcs_fixed_y``
so they do not depend on the model package.
"""

from ultralytics.utils.gcs_fixed_y import (
    expected_training_fixed_y_desc,
    expected_tusimple_h_samples,
    validate_fixed_y_anchors,
    validate_official_h_samples_asc,
    validate_training_fixed_y_desc,
)

__all__ = (
    "expected_training_fixed_y_desc",
    "expected_tusimple_h_samples",
    "validate_fixed_y_anchors",
    "validate_official_h_samples_asc",
    "validate_training_fixed_y_desc",
)
