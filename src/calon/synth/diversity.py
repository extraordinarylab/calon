"""Attribute hints that diversify generations (after AttrPrompt, Yu et al. 2023, and Persona Hub,
Ge et al. 2024): every request gets its own writer persona, region and period, and the names
the round-5 analysis found overused are banned. Deterministic per request id. Used when the
environment sets IE_DIVERSE=1 (round 6 on); earlier rounds' prompts are unchanged."""
import os
import random

from .llm_annotate import seed_of

ENABLED = os.environ.get("IE_DIVERSE", "0") == "1"
REGIONS = ["Nigeria", "Kenya", "Ethiopia", "Ghana", "South Africa", "Egypt", "Morocco", "Brazil", "Argentina",
           "Chile", "Colombia", "Peru", "Mexico", "Cuba", "Canada", "the United States (the Midwest)",
           "the United States (the South)", "Ireland", "Scotland", "Wales", "Portugal", "Spain", "France",
           "the Netherlands", "Poland", "Czechia", "Hungary", "Romania", "Greece", "Turkey", "Ukraine", "Finland",
           "Norway", "Iceland", "Israel", "Iran", "Saudi Arabia", "Pakistan", "India (Tamil Nadu)",
           "India (Punjab)", "Bangladesh", "Sri Lanka", "Nepal", "Vietnam", "Thailand", "the Philippines",
           "Indonesia", "Malaysia", "South Korea", "Japan", "Taiwan", "Mongolia", "Kazakhstan", "Australia",
           "New Zealand", "Fiji", "Jamaica", "Georgia (the country)", "Lebanon", "Senegal"]
PERIODS = ["the 1890s", "the 1920s", "the 1950s", "the 1970s", "the 1980s", "the 1990s", "the early 2000s",
           "the 2010s", "the early 2020s", "the present day", "the present day", "the present day"]
PERSONAS = ["a retired schoolteacher", "a night-shift nurse", "a first-year university student",
            "a small-business owner", "a local reporter", "a museum archivist", "a software engineer",
            "a farmer", "a truck driver", "a hospital administrator", "a postdoctoral researcher",
            "a public-health officer", "a union organiser", "a church volunteer", "a sports blogger",
            "a court clerk", "a travel agent", "a pharmacist", "a high-school student", "a city planner",
            "a fisherman", "a chef", "a librarian", "a stay-at-home parent", "a junior lawyer",
            "a factory technician", "a wildlife ranger", "a stand-up comedian", "a bank teller",
            "a retired army officer", "a graduate teaching assistant", "a midwife", "a taxi driver",
            "a film critic", "a startup founder", "a translator", "a social worker", "an architect",
            "an amateur historian", "an IT helpdesk worker"]
TONES = ["plain and factual", "warm and chatty", "terse", "formal", "slightly humorous", "worried",
         "enthusiastic", "critical", "matter-of-fact", "nostalgic"]
OVERUSED = ["Elena", "Sarah", "Smith", "Aris", "Thorne", "Nadia", "Yuri", "Vance", "Rossi", "Jane Doe",
            "John Smith", "Alice", "Chen", "Maria", "Marcus", "Evelyn", "Julian"]


def hint(request_id, period=True, persona=True, names_only=False):
    """One paragraph appended to a generation prompt, or "" when IE_DIVERSE is off.
    names_only: the topic is fixed (scientific, biomedical, seeded data); only the names vary."""
    if not ENABLED:
        return ""
    rng = random.Random(seed_of(f"diverse:{request_id}"))
    if names_only:
        return (f"\n\nNames: where the text names people, places, organisations or institutions, use names "
                f"typical of {rng.choice(REGIONS)}, varied and specific; keep the topic and domain as asked. "
                "Do not use the names " + ", ".join(OVERUSED) + ".")
    parts = [f"Set it in {rng.choice(REGIONS)}" + (f", in {rng.choice(PERIODS)}" if period else "") + "."]
    if persona:
        parts.append(f"Write it as {rng.choice(PERSONAS)} would, in a {rng.choice(TONES)} tone.")
    parts.append("Give people, places and organisations names that fit this setting, varied and specific; "
                 "do not use the names " + ", ".join(OVERUSED) + ".")
    return "\n\nSetting: " + " ".join(parts)
