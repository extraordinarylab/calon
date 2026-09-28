"""Error breakdown used by the OOD diagnostics."""
import numpy as np

from ierlcd.ood_eval import error_breakdown


def test_error_breakdown_classes():
    # candidates: (start, end, i, j)
    cands = [(0, 4, 0, 0), (0, 9, 0, 1), (10, 14, 2, 2), (20, 25, 4, 4), (30, 35, 6, 6)]
    e = {"cands": cands, "labels": ["person", "politician", "city"],
         "gold_spans": {(0, 4, "politician"), (10, 14, "city"), (30, 35, "city")}}
    z = np.full((3, len(cands)), -10.0)
    z[0, 0] = 5    # person on gold politician span, politician also predicted -> extra label
    z[1, 0] = 5    # correct
    z[2, 1] = 5    # city on (0,9): overlaps politician gold only -> boundary_and_type
    z[0, 3] = 5    # spurious
    z[2, 2] = 5    # correct city
    # (30,35) city missed
    out = error_breakdown([e], [{"z": z}], 1.0, 0.5)
    assert out["fp"] == {"wrong_type": 1, "wrong_type_extra_label": 1,
                         "boundary_and_type": 1, "spurious": 1}
    assert out["fn"] == {"missed": 1}
