"""Where generation requests / outputs and generated sources live. A later round runs in its own
directories (IE_SYNTH2, IE_DATA_OUT) so that the data of a model still training is not touched."""
import os
from pathlib import Path

from calon.data.roots import ROOT

S2 = Path(os.environ.get("IE_SYNTH2", str(ROOT / "synth2")))           # requests and model outputs
DATA_OUT = Path(os.environ.get("IE_DATA_OUT", str(ROOT / "data")))     # <DATA_OUT>/re, <DATA_OUT>/ee, ...
