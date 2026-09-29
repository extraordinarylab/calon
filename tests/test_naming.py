"""The method's old name still works where it names the head."""
from calon.model.model import canonical_head


def test_old_head_name_maps_to_calon():
    assert canonical_head("rlcd") == "calon"
    assert canonical_head("calon") == "calon"
    assert canonical_head("bce") == "bce"
