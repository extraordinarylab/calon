"""The method is RLCD (framework and package CalOn); "calon" still works as the head's name."""
from calon.eval.evaluate import calon_brier, calon_probs, rlcd_brier, rlcd_probs
from calon.model.losses import calon_log_probs, calon_loss, rlcd_log_probs, rlcd_loss
from calon.model.model import canonical_head


def test_head_name_is_rlcd_with_calon_alias():
    assert canonical_head("rlcd") == "rlcd"
    assert canonical_head("calon") == "rlcd"
    assert canonical_head("bce") == "bce"


def test_earlier_function_names_are_aliases():
    assert calon_loss is rlcd_loss and calon_log_probs is rlcd_log_probs
    assert calon_probs is rlcd_probs and calon_brier is rlcd_brier
