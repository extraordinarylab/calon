"""Round-5 targeted relation data, from the diagnosis of the joint model on the third batch of
evaluation sets (prepare_evalsets2): the relation head had no notion of relations whose ends
are pronouns, opinion words or events, and confused fine person / organisation attributes.
Label names are the evaluation sets' names (as round 4 did for SciER); the generator writes
every definition and every text, no evaluation example is shown, and the texts pass the
contamination filter against all evaluation sets.

  gcoref  passages with several people mentioned by name, pronoun and noun phrase; mentions
          grouped into entities. Two rows per passage: "pronoun" / "person name" with
          "refers to" (pronoun -> each name of its person), and "mention" with "coreference"
          (mention -> the nearest earlier mention of its entity).
  gopin   customer reviews of many products and services: "aspect term" -> "opinion term"
          with "positive / negative / neutral sentiment".
  gtemp   event-rich narratives: "event" and "time expression" mentions; temporal, causal
          and subevent relations between the first mentions of two events, "coreference"
          from a later mention of an event to its first mention.
  gslot   short news and web texts with person and organisation attributes, ten relation
          types per text drawn from the Re-TACRED names (direction-confusable pairs kept).
  gbio    biomedical abstracts: the six BioRED entity types, mentions grouped into concepts,
          the eight BioRED relation types between concepts (undirected: both directions
          between the first mentions of the two concepts).
The generator marks every mention inline as [[text|M<n>]], so repeated words ("he", "great")
get exact offsets. Filtering: every relation goes to a judge model (gpt-oss-120b), which sees
the passage with numbered mentions and answers yes / no; rejected relations are dropped, and
a relation type whose proposals were all rejected becomes "unknown".
Requests: synth2/<task>_requests.jsonl (llm_gen2 run, gen_node.sh); judge task r5jd.
Subcommands: prepare, assemble (candidates + judge requests), apply (-> data/re/synth_<x>).
"""
import argparse
import collections
import json
import random
import re

from calon.data.roots import ROOT

from .llm_annotate import clean_type, read_jsonl, write_jsonl
from .paths import DATA_OUT, S2  # noqa: E402

R5 = DATA_OUT / "r5"
TASKS = ("gcoref", "gopin", "gtemp", "gslot", "gbio", "gdaily", "gment", "gmisc", "gslot2", "gopin2", "gbio2", "gmeas")
SOURCE = {"gcoref": "synth_coref", "gopin": "synth_opin", "gtemp": "synth_temp", "gslot": "synth_slot",
          "gbio": "synth_bio", "gdaily": "synth_daily", "gment": "synth_ment", "gmisc": "synth_misc",
          "gslot2": "synth_slot2", "gopin2": "synth_opin2", "gbio2": "synth_bio2", "gmeas": "synth_meas"}
# round 8, from the case study against gpt-oss
#   gopin2  reviews with idioms, sarcasm, negation and comparisons ("out of this world", "to die for" were
#           read as negative with confidence 0.98+), longer multi-aspect reviews for aspect / opinion finding
#   gbio2   BioRED: positive / negative correlation told apart from association, relations across sentences
#   gmeas   MeasEval: several quantities in one passage, each bound to its measured entity and property
OPIN2_STYLES = ["a short review that praises with idioms (for example 'out of this world', 'to die for', 'hits the "
                "spot')", "a short sarcastic review", "a short review with negated opinions ('not bad', 'never "
                "disappoints', 'nothing special')", "a short review comparing two things ('better than', 'not as good as')",
                "a mixed review with four or more aspects", "a review with neutral, factual remarks about some aspects",
                "a one-sentence review with slang", "a longer review with several aspects and opinions each"]
BIO2_EXTRA = """
Relation types, in this abstract: "positive correlation" when more of one concept goes with more of the other (a drug increases, induces, raises, causes a disease or a gene's expression); "negative correlation" when more of one goes with less of the other (a drug treats, inhibits, reduces, protects against); "association" only when the abstract states a link without its direction (for example a variant associated with a disease). Write the abstract so that at least half of the relations hold between concepts mentioned in different sentences, and annotate those too."""
MEAS_ENTS = ["quantity", "measured entity", "measured property", "qualifier"]
MEAS_RELS = ["has quantity", "has property", "qualifies"]
GMEAS_PROMPT = """Write one realistic {genre} from a research article on {topic}, about {length} words long, in English, that reports at least four measurements (numbers with units, percentages, ranges, counts), several of them in the same sentence, about different things.

{mark_rule}
Mark every mention of these types: "quantity" (a number or range with its unit, for example "3.4%", "1 ml or 5 ml", "≈ 170°"), "measured entity" (the thing that is measured, for example "the syringe", "MEX orbits"), "measured property" (the property measured, for example "height", "wind speed", "concentration") and "qualifier" (a condition that qualifies the measurement, for example "at 25 °C", "in the control group").
Then list the relations, directed from head to tail, using only these types:
- "has quantity": from a measured entity or measured property to its quantity
- "has property": from a measured entity to its measured property
- "qualifies": from a qualifier to the quantity, entity or property it qualifies
Bind every quantity to the right entity: when a sentence has several quantities, each has its own measured entity or property.
Give a one-sentence definition, general enough for other texts, of every type and relation listed.

Return JSON only: {{"text": "...", "mentions": {{"M1": "type", ...}}, "relations": [{{"head": "M1", "relation": "...", "tail": "M2"}}], "definitions": {{"type or relation": "...", ...}}}}"""
MEAS_TOPICS = ["planetary physics", "atmospheric science", "materials engineering", "electrochemistry", "neuroscience",
               "ecology", "hydrology", "oncology", "pharmacology", "software engineering", "energy systems",
               "geology", "agronomy", "astronomy", "polymer chemistry", "marine biology", "epidemiology",
               "civil engineering", "food science", "particle physics"]
# round 7, from the ee3 error analysis
#   gment   every mention (names, common noun phrases, pronouns) of coarse types (GENTLE recall 0.04:
#           it annotates every noun phrase, our sources mostly named things)
#   gmisc   specific types with their general type and a residual "miscellaneous" (CrossNER
#           miscellaneous F1 0.2-0.36, writer -> person confusions)
#   gslot2  person / organization attributes with "person identity" and "organization alternate names"
#           always asked and defined (Re-TACRED person identity: 2,036 gold, recall 0.00)
MENT_SETS = [
    ["person", "place", "organization", "object", "event", "time", "abstract", "substance", "animal", "plant"],
    ["person", "organization", "geo-political entity", "location", "facility", "vehicle", "weapon"],
    ["person", "group", "place", "artifact", "event", "concept", "quantity", "time"],
]
MENT_GENRES = ["dictionary entry", "legal statute excerpt", "poem", "clinical case note", "university course syllabus",
               "threatening letter quoted in a court record", "esports match report", "mathematical proof sketch",
               "Reddit comment thread", "how-to guide", "biography", "fairy tale", "travel blog post", "interview transcript",
               "news report", "product manual", "recipe", "history textbook paragraph", "letter to a friend",
               "museum guide text", "philosophy essay paragraph", "sports commentary", "court testimony"]
IDENTITY_DEFS = {
    "person identity": "the head is a person and the tail is another mention of the same person in the text: another "
                       "name, a nickname, a title used as a name, or a pronoun or noun phrase referring to that person",
    "organization alternate names": "the head is an organization and the tail is another name of the same "
                                    "organization: an abbreviation, a former name or a short form",
}
# round 6, from the distribution analysis: everyday short texts (queries, requests, messages,
# one-line reviews) were scarce; general everyday topics, not only the evaluation domains
DAILY_KINDS = ["voice-assistant requests", "search-engine queries", "chat messages to a friend",
               "one-sentence reviews", "customer-support messages", "social media posts",
               "questions to a shop assistant", "notes to self", "text messages to family",
               "questions to a chatbot"]
DAILY_TOPICS = ["films", "restaurants and food", "music", "television series", "books", "podcasts", "video games",
                "sports results", "flights", "hotels", "train and bus travel", "weather", "online shopping",
                "groceries", "recipes and cooking", "food delivery", "taxis and ride-hailing", "home repairs",
                "concerts and events", "fitness and gyms", "banking and payments", "smart-home devices",
                "news headlines", "doctor and dentist appointments", "renting a flat", "cars and car repair",
                "pets and vets", "clothes and fashion", "phones and laptops", "online courses", "job search",
                "school and homework", "museums and exhibitions", "parks and hiking", "cafés and bars",
                "hair salons", "insurance", "parcels and post", "public services", "local sports clubs"]
SENTIMENTS = ["positive sentiment", "negative sentiment", "neutral sentiment"]
TEMP_RELATIONS = ["before", "overlap", "contains", "simultaneous", "ends on", "begins on", "cause", "precondition",
                  "subevent", "coreference"]
LINK_LABELS = {"refers to", "coreference"}       # judged as "the same entity"
MARK = re.compile(r"\[\[([^\[\]|]+?)\|\s*([A-Za-z]{0,3}\d+)\s*\]\]")

COREF_GENRES = ["novel excerpt", "short story passage", "nineteenth-century novel passage", "children's story",
                "biography paragraph", "encyclopedia biography", "news feature", "memoir passage", "sports report",
                "celebrity profile", "historical account", "obituary", "court case summary", "film plot summary",
                "interview write-up", "travel diary entry"]
OPIN_ITEMS = ["a laptop", "a smartphone", "a restaurant", "a café", "a hotel", "headphones", "a camera", "a car",
              "a mobile app", "a video game", "a book", "a film", "an airline flight", "a dental clinic",
              "an online course", "a bank", "a gym", "a sofa", "a washing machine", "a lipstick", "a winter coat",
              "a guided tour", "a concert", "a plumbing service", "a food delivery service", "a smartwatch",
              "a television", "a printer", "a bakery", "a bar", "a pizza place", "a sushi restaurant", "a tablet",
              "a keyboard", "a vacuum cleaner", "a hair salon", "a car rental", "a theme park", "a museum",
              "a coffee machine", "a pair of running shoes", "a mattress", "a hospital stay", "a language school",
              "an e-reader", "a streaming service", "a hotel restaurant", "a steakhouse", "a vegan restaurant"]
OPIN_STYLES = ["a one-sentence review", "a one-sentence review with two aspects", "a short mixed review",
               "a short mostly negative review", "a short mostly positive review",
               "a short factual review with neutral remarks", "a one-sentence complaint",
               "a one-sentence recommendation", "a short review in casual online style"]
TEMP_GENRES = ["Wikipedia article section", "encyclopedia entry", "news report", "historical narrative",
               "timeline paragraph", "chronicle of events", "investigation report summary"]
TEMP_TOPICS = ["a battle", "a war", "a national election", "an earthquake", "a flood", "a hurricane", "a wildfire",
               "a bombing", "a sports tournament", "a scientific expedition", "the history of a company", "a protest",
               "a revolution", "an epidemic", "a plane crash", "a shipwreck", "a music festival",
               "a construction project", "a political crisis", "a peace treaty", "a space mission", "a criminal trial",
               "a royal succession", "a strike by workers", "a stock market crash", "a volcanic eruption",
               "a famine", "a rescue operation", "a coup", "a siege", "a naval campaign", "an Olympic Games",
               "a championship season", "a scandal", "an assassination", "a colonial occupation", "a mass migration",
               "a train derailment", "a mining disaster", "a cyber attack"]
SLOT_GENRES = ["news report", "newswire paragraph", "web biography snippet", "press release", "obituary paragraph",
               "forum post", "local news brief", "business news brief", "sports news brief"]
BIO_GENRES = ["PubMed title and abstract", "PubMed abstract", "PubMed abstract of a case report",
              "PubMed abstract of a genetic association study", "PubMed abstract of a drug study"]
BIO_TOPICS = ["cancer genetics", "type 2 diabetes", "cardiovascular disease", "Alzheimer's disease",
              "Parkinson's disease", "rare inherited disorders", "infectious disease", "pharmacogenomics",
              "adverse drug reactions", "drug-drug interactions", "autoimmune disease", "epilepsy", "hypertension",
              "obesity and metabolism", "kidney disease", "liver disease", "psychiatric disorders",
              "hearing loss genetics", "cystic fibrosis", "immunotherapy", "antibiotic resistance",
              "cell line experiments", "animal models of disease", "gene expression regulation",
              "signalling pathways", "toxicology", "vaccines", "hormone therapy", "blood disorders", "eye disease"]
LENGTHS = {"gcoref": [120, 180, 250], "gopin": [15, 25, 40, 70], "gtemp": [80, 140, 200],
           "gslot": [25, 45, 70], "gbio": [150, 220, 300]}

MARK_RULE = """Mark every annotated mention inline in the text as [[mention text|M<number>]] (for example "[[Maria Lopez|M1]] said that [[she|M2]] ..."), numbering mentions M1, M2, ... in order of appearance. Every marker gets its own new number, also when the same word or the same thing is mentioned again. Markers must not be nested or overlap, and the text without the markers must read naturally."""

GCOREF_PROMPT = """Write one realistic {genre} from the domain of {domain}, about {length} words long, in English, in which at least three people (at least two of them of the same gender) and some places, organisations or objects are mentioned repeatedly: by name, by pronouns (he, she, him, his, her, they, it, ...) and by descriptive noun phrases (the old man, his sister, the company, ...).

{mark_rule}
Mark every mention of a person, place, organisation, facility, vehicle or group, including every pronoun and descriptive noun phrase that refers to one of them; at most 24 mentions. Give each mention one type: "person name" (a name of a person), "pronoun" (a personal or possessive pronoun) or "other mention" (any other name or noun phrase).
Then group the mentions into entities: list every set of two or more mentions that refer to the same person or thing.
Give a one-sentence definition, general enough for other texts, of: "pronoun", "person name", "mention" (any mention of a person, place, organisation, facility, vehicle or group), "refers to" (from a pronoun to a name of the person it refers to) and "coreference" (from a mention to an earlier mention of the same person or thing).

Return JSON only: {{"text": "...", "mentions": {{"M1": "type", ...}}, "entities": [["M1", "M3"], ...], "definitions": {{"pronoun": "...", "person name": "...", "mention": "...", "refers to": "...", "coreference": "..."}}}}"""

GOPIN_PROMPT = """Write {style} of {item}, about {length} words long, in English, as a customer would post it online.

{mark_rule}
Mark every aspect term (a word or phrase naming the product or service, or a part, attribute or feature of it, that the review evaluates, for example "battery life" or "the waiter") and every opinion term (the word or phrase that expresses the evaluation, for example "great", "too slow", "never came back"). Give each mention the type "aspect term" or "opinion term".
Then list every pair of an aspect term and an opinion term that evaluates it, from the aspect term (head) to the opinion term (tail), with its sentiment as the relation: "positive sentiment", "negative sentiment" or "neutral sentiment". An aspect term can have several opinion terms and an opinion term can evaluate several aspect terms.
Give a one-sentence definition, general enough for other reviews, of "aspect term", "opinion term" and the three relations.

Return JSON only: {{"text": "...", "mentions": {{"M1": "aspect term", ...}}, "relations": [{{"head": "M1", "relation": "...", "tail": "M2"}}], "definitions": {{"aspect term": "...", "opinion term": "...", "positive sentiment": "...", "negative sentiment": "...", "neutral sentiment": "..."}}}}"""

GTEMP_PROMPT = """Write one realistic {genre} about {topic} (invented or real, in any country and period), about {length} words long, in English, that describes a sequence of related events, with dates, times or durations where natural.

{mark_rule}
Mark every event mention (the word or short phrase, usually a verb or a noun, that expresses something that happens or a state that holds, for example "attacked", "the election", "flooding", "remained") and every time expression (dates, times, durations, periods); at most 22 mentions. Do not mark participants, people, organisations, places or objects (for example "the workers", "the market", "shares") as events: only the words that express what happens. If an event is mentioned again later, mark each mention. Give each mention the type "event" or "time expression".
Then list the relations between mentions, directed from head to tail, using only these types:
- "before": the head ends before the tail begins
- "overlap": the head and the tail overlap in time
- "contains": the time of the head contains the time of the tail
- "simultaneous": the head and the tail happen at the same time
- "ends on": the head ends when the tail happens or begins
- "begins on": the head begins when the tail happens or begins
- "cause": the head event causes the tail event
- "precondition": the head event is a necessary condition for the tail event
- "subevent": the tail event is a part of the head event
- "coreference": the head is a later mention of the same event as the earlier mention tail
Give the temporal relation of every pair of events (or event and time expression) whose order the text makes clear, and every causal, subevent and coreference relation the text states or clearly implies.
Give a one-sentence definition, general enough for other texts, of "event", "time expression" and every relation type.

Return JSON only: {{"text": "...", "mentions": {{"M1": "event", ...}}, "relations": [{{"head": "M1", "relation": "...", "tail": "M2"}}], "definitions": {{"event": "...", "time expression": "...", "before": "...", ...}}}}"""

GSLOT_PROMPT = """Write one realistic {genre} from the domain of {domain}, about {length} words long, in English, with concrete people and organisations and facts about them.

{mark_rule}
Mark every mention of these entity types: {etypes}.
Use these directed relation types (from head to tail): {rtypes}. Write the text so that several of them are stated, and annotate every relation of these types that the text states between two marked mentions. Mind the direction: the head of a "person ..." relation is a person and the head of an "organization ..." relation is an organization.
Give a one-sentence definition, general enough for other texts, of every entity type and every relation type listed.

Return JSON only: {{"text": "...", "mentions": {{"M1": "entity type", ...}}, "relations": [{{"head": "M1", "relation": "...", "tail": "M2"}}], "definitions": {{"type or relation": "...", ...}}}}"""

GBIO_PROMPT = """Write one realistic {genre} on {topic}, about {length} words long, in English, with specific genes, proteins, variants, diseases, chemicals or drugs, organisms and cell lines (real or plausible), several of them mentioned more than once.

{mark_rule}
Mark every mention of these entity types: {etypes}. Give each mention its type and a concept id (C1, C2, ...): mentions of the same gene, disease, chemical, variant, organism or cell line (including abbreviations and synonyms) share one concept id.
Then list the relations between concepts that the abstract states, using only these relation types (undirected: the order of the two concepts does not matter): {rtypes}.
Give a one-sentence definition, general enough for other abstracts, of every entity type and every relation type listed.

Return JSON only: {{"text": "...", "mentions": {{"M1": {{"type": "...", "concept": "C1"}}, ...}}, "relations": [{{"head": "C1", "relation": "...", "tail": "C2"}}], "definitions": {{"type or relation": "...", ...}}}}"""

JUDGE_PROMPT = """Text (mentions numbered in brackets):
{text}

Relation types:
{rtypes}

Candidate relations:
{cands}

For each candidate answer "yes" if the text states or clearly implies that relation from the head to the tail, otherwise "no".{extra}
Return JSON only: {{"R1": "yes" or "no", ...}}"""
GDAILY_PROMPT = """Write 8 different {kind} in English about {topic}, each 3 to 30 words, the way real people type or say them: varied wording and intent, everyday names of titles, people, places, products, times, prices and the like; casual style, lowercase and small typos are fine.

First design an entity schema for {topic}: 4 to 10 entity types that a developer of an app for {topic} would want to extract from such texts, each with a one-sentence definition that would apply to other texts too.
Mark every mention of a schema type inline in every text as [[mention text|M<number>]], numbering mentions M1, M2, ... across all 8 texts. Every marker gets its own new number. Markers must not be nested or overlap, and each text without the markers must read naturally. Texts may contain no mention at all.

Return JSON only: {{"entity_types": {{"type": "definition"}}, "texts": ["...", "..."], "mentions": {{"M1": "type", ...}}}}"""

GMENT_PROMPT = """Write one realistic {genre} from the domain of {domain}, about {length} words long, in English.

{mark_rule}
Mark EVERY mention of these entity types: {etypes}. A mention is any expression that refers to such a thing: proper names ("Lake Ohrid"), common noun phrases ("the old bridge", "a contract", "his anger", "the next meeting") and pronouns ("it", "she", "they"). Mark the full noun phrase, including its determiner and modifiers. Abstract things (ideas, feelings, rules, processes) count where the type list has a type for them. Do not mark expressions that fit none of the listed types. At most 30 mentions; if there are more, mark the first 30.
Give a one-sentence definition, general enough for other texts, of every entity type listed.

Return JSON only: {{"text": "...", "mentions": {{"M1": "type", ...}}, "definitions": {{"type": "...", ...}}}}"""

GMISC_PROMPT = """Write one realistic {genre} from the domain of {domain}, about {length} words long, in English, with many specific named things (people, organisations, places, works, products, events, nationalities, titles and the like).

Design an entity schema for this text: 5 to 9 specific entity types that a user working in {domain} would want to extract, including at least one general type together with a more specific subtype of it (for example "person" and "writer", or "organization" and "record label"), plus the type "miscellaneous" for named things that fit none of the other types (for example nationalities, named events, languages, slogans, titles of works the schema has no type for).
{mark_rule}
Mark every named mention of the schema types with its most specific type: a writer is "writer", not "person"; use the general type only when no more specific type applies; use "miscellaneous" only for named things outside every other type.
Give a one-sentence definition, general enough for other texts, of every type of the schema.

Return JSON only: {{"text": "...", "mentions": {{"M1": "type", ...}}, "definitions": {{"type": "...", ...}}}}"""

MENTION_JUDGE_PROMPT = """Text (mentions numbered in brackets):
{text}

Mention types:
{etypes}

Candidate mentions:
{cands}

For each candidate answer "yes" if the bracketed span is a complete mention of that type in this text (right boundaries, right type), otherwise "no".
Return JSON only: {{"E1": "yes" or "no", ...}}"""
MENTION_JUDGED = ("gtemp", "gopin", "gdaily", "gment", "gmisc", "gopin2", "gmeas")
LINK_EXTRA =' For "refers to" and "coreference" answer "yes" only if the two mentions refer to the same person or thing.'


def eval_schema():
    """Re-TACRED and BioRED label names (names only) from the converted evaluation sets."""
    m = json.loads((ROOT / "data/ood_re_x2/manifest.json").read_text())["sets"]
    first = {}
    for name in ("retacred", "biored_re"):
        with open(ROOT / f"data/ood_re_x2/{name}.jsonl") as f:
            first[name] = json.loads(f.readline())
    return {"slot_entities": m["retacred"]["labels"],
            "slot_relations": [q["label"] for q in first["retacred"]["rel_queries"]],
            "bio_entities": m["biored_re"]["labels"],
            "bio_relations": [q["label"] for q in first["biored_re"]["rel_queries"]]}


def quoted(xs):
    return ", ".join(f'"{x}"' for x in xs)


def prompt_for(task, r):
    if task == "r5jd" and r.get("kind") == "mention":
        return MENTION_JUDGE_PROMPT.format(text=r["text"], cands=r["cands"], etypes="\n".join(
            f'- "{t}": {d}' if d else f'- "{t}"' for t, d in r["etypes"]))
    if task == "r5jd":
        return JUDGE_PROMPT.format(text=r["text"], rtypes="\n".join(f'- "{t}": {d}' if d else f'- "{t}"'
                                                                  for t, d in r["rtypes"]),
                                   cands=r["cands"], extra=LINK_EXTRA if r.get("links") else "")
    kw = dict(r, mark_rule=MARK_RULE)
    if task in ("gslot", "gbio", "gslot2", "gment", "gbio2", "gmeas"):
        kw["etypes"] = quoted(r["etypes"])
        kw["rtypes"] = quoted(r.get("rtypes", []))
    from .diversity import hint
    period, persona = {"gcoref": (True, True), "gopin": (False, True), "gtemp": (True, True),
                       "gslot": (True, True), "gbio": (False, False), "gdaily": (False, True),
                       "gment": (True, True), "gmisc": (True, False), "gslot2": (True, True),
                       "gopin2": (False, True), "gbio2": (False, False), "gmeas": (False, False)}[task]
    prompt = {"gcoref": GCOREF_PROMPT, "gopin": GOPIN_PROMPT, "gtemp": GTEMP_PROMPT, "gslot": GSLOT_PROMPT,
              "gbio": GBIO_PROMPT, "gdaily": GDAILY_PROMPT, "gment": GMENT_PROMPT, "gmisc": GMISC_PROMPT,
              "gslot2": GSLOT_PROMPT, "gopin2": GOPIN_PROMPT, "gbio2": GBIO_PROMPT, "gmeas": GMEAS_PROMPT}[task].format(**kw)
    if task == "gslot2":
        prompt += "\n\nUse these definitions:\n" + "\n".join(f'- "{k}": {v}' for k, v in IDENTITY_DEFS.items()) + \
            "\nState at least one person identity and one organization alternate names relation in the text."
    if task == "gbio2":
        prompt += BIO2_EXTRA
    if task == "gopin2":
        prompt += ("\n\nMark the full opinion expression, including idioms, negation and comparison words "
                   "(\"out of this world\", \"not as good as\"), and give the sentiment the reviewer means, also when "
                   "the words are figurative or sarcastic.")
    return prompt + hint(r["id"], period=period, persona=persona, names_only=task in ("gbio", "gbio2", "gmeas"))


# ---------------------------------------------------------------- prepare

SLOT_PAIRS = [("person employee of", "organization top members/employees"),
              ("organization founded by", "person employee of"),
              ("person parents", "person children"), ("person city of birth", "person cities of residence"),
              ("organization parents", "organization subsidiaries"), ("organization members", "organization member of"),
              ("person origin", "person countries of residence"), ("person date of birth", "person age")]


def prepare(args):
    from .llm_gen2 import DOMAINS3
    sch = eval_schema()
    rng = random.Random(args.seed + 17)
    reqs = {t: [] for t in TASKS}
    n = {"gcoref": args.n_coref, "gopin": args.n_opin, "gtemp": args.n_temp, "gslot": args.n_slot,
         "gbio": args.n_bio, "gdaily": args.n_daily}
    for k in range(n["gdaily"]):
        reqs["gdaily"].append({"id": f"gdaily:{k}", "topic": DAILY_TOPICS[k % len(DAILY_TOPICS)],
                               "kind": DAILY_KINDS[(k // len(DAILY_TOPICS)) % len(DAILY_KINDS)]})
    for k in range(n["gcoref"]):
        reqs["gcoref"].append({"id": f"gcoref:{k}", "genre": COREF_GENRES[k % len(COREF_GENRES)],
                               "domain": rng.choice(DOMAINS3), "length": rng.choice(LENGTHS["gcoref"])})
    for k in range(n["gopin"]):
        reqs["gopin"].append({"id": f"gopin:{k}", "item": OPIN_ITEMS[k % len(OPIN_ITEMS)],
                              "style": rng.choice(OPIN_STYLES), "length": rng.choice(LENGTHS["gopin"])})
    for k in range(n["gtemp"]):
        reqs["gtemp"].append({"id": f"gtemp:{k}", "genre": rng.choice(TEMP_GENRES),
                              "topic": TEMP_TOPICS[k % len(TEMP_TOPICS)], "length": rng.choice(LENGTHS["gtemp"])})
    rels = [x for x in sch["slot_relations"]]
    for k in range(n["gslot"]):
        pair = [x for x in rng.choice(SLOT_PAIRS) if x in rels]
        rest = [x for x in rels if x not in pair]
        rng.shuffle(rest)
        rtypes = pair + rest[:10 - len(pair)]
        rng.shuffle(rtypes)
        reqs["gslot"].append({"id": f"gslot:{k}", "genre": rng.choice(SLOT_GENRES), "domain": rng.choice(DOMAINS3),
                              "length": rng.choice(LENGTHS["gslot"]), "etypes": sch["slot_entities"],
                              "rtypes": rtypes})
    for k in range(n["gbio"]):
        reqs["gbio"].append({"id": f"gbio:{k}", "genre": rng.choice(BIO_GENRES),
                             "topic": BIO_TOPICS[k % len(BIO_TOPICS)], "length": rng.choice(LENGTHS["gbio"]),
                             "etypes": sch["bio_entities"], "rtypes": sch["bio_relations"]})
    from .llm_gen2 import GENRES3
    for k in range(args.n_ment):
        reqs["gment"].append({"id": f"gment:{k}", "genre": MENT_GENRES[k % len(MENT_GENRES)],
                              "domain": rng.choice(DOMAINS3), "length": rng.choice([60, 120, 200]),
                              "etypes": MENT_SETS[0] if k % 2 == 0 else MENT_SETS[1 + (k // 2) % 2]})
    for k in range(args.n_misc):
        reqs["gmisc"].append({"id": f"gmisc:{k}", "genre": rng.choice(GENRES3), "domain": DOMAINS3[k % len(DOMAINS3)],
                              "length": rng.choice([60, 120, 200])})
    for k in range(args.n_slot2):
        rest = [x for x in rels if x not in IDENTITY_DEFS]
        rng.shuffle(rest)
        rtypes = list(IDENTITY_DEFS) + rest[:8]
        rng.shuffle(rtypes)
        reqs["gslot2"].append({"id": f"gslot2:{k}", "genre": rng.choice(SLOT_GENRES), "domain": rng.choice(DOMAINS3),
                               "length": rng.choice(LENGTHS["gslot"]), "etypes": sch["slot_entities"], "rtypes": rtypes})
    for k in range(args.n_opin2):
        reqs["gopin2"].append({"id": f"gopin2:{k}", "item": OPIN_ITEMS[k % len(OPIN_ITEMS)],
                               "style": OPIN2_STYLES[(k // len(OPIN_ITEMS)) % len(OPIN2_STYLES)],
                               "length": rng.choice([25, 40, 70, 120])})
    for k in range(args.n_bio2):
        reqs["gbio2"].append({"id": f"gbio2:{k}", "genre": rng.choice(BIO_GENRES), "topic": BIO_TOPICS[k % len(BIO_TOPICS)],
                              "length": rng.choice([180, 250, 320]), "etypes": sch["bio_entities"],
                              "rtypes": sch["bio_relations"]})
    for k in range(args.n_meas):
        reqs["gmeas"].append({"id": f"gmeas:{k}", "genre": rng.choice(["results paragraph", "methods paragraph",
                                                                        "abstract", "figure description paragraph"]),
                              "topic": MEAS_TOPICS[k % len(MEAS_TOPICS)], "length": rng.choice([80, 140, 200]),
                              "etypes": MEAS_ENTS, "rtypes": MEAS_RELS})
    for task, rs in reqs.items():
        if not rs:
            continue
        rng.shuffle(rs)
        write_jsonl(S2 / f"{task}_requests.jsonl", rs)
    print({t: len(r) for t, r in reqs.items()}, json.dumps(sch)[:800])


# ---------------------------------------------------------------- assemble

def unmark(marked):
    """Plain text, {mention id: (start, end)} and {id: [ids of its repeats]} from [[text|ID]]
    markers. Generators sometimes number by kind (G1, D1, T1) or reuse one id for every
    mention of the same thing: a repeated id becomes ID~2, ID~3, ... and is listed under ID."""
    out, spans, dup, pos = [], {}, collections.defaultdict(list), 0
    for m in MARK.finditer(marked):
        out.append(marked[pos:m.start()])
        start = sum(len(x) for x in out)
        word = m.group(1).strip()
        out.append(word)
        mid = m.group(2)
        if mid in spans:
            dup[mid].append(f"{mid}~{len(dup[mid]) + 2}")
            mid = dup[mid][-1]
        spans[mid] = (start, start + len(word))
        pos = m.end()
    out.append(marked[pos:])
    text = "".join(out)
    # generators sometimes repeat the marked words after the marker ("[[CCM|M2]] CCM"): unusable
    echo = any(marked[m.end():].lstrip().startswith(m.group(1).strip() + " ") and len(m.group(1).strip()) > 2
               for m in MARK.finditer(marked))
    return text, spans, dict(dup), ("[[" in text or "]]" in text or "**" in text or echo)


def by_order(keys):
    """Mention-list keys in numeric order ("M2" before "M10")."""
    def num(k):
        m = re.search(r"(\d+)$", str(k))
        return int(m.group(1)) if m else 10 ** 9
    return sorted(keys, key=num)


def remap_ids(marker_ids, raw):
    """The text numbers its markers its own way (P1, L1, 1, 2, ...) while the mention list is
    M1..Mn in order of appearance: map the k-th marker to the k-th list entry. None unless the
    counts agree and the list keys do not already match."""
    if sum(m in raw for m in marker_ids) >= len(marker_ids) / 2 or len(raw) != len(marker_ids):
        return None
    return dict(zip(marker_ids, by_order(raw)))


def defs_of(js, names):
    d = {clean_type(k): str(v) for k, v in (js.get("definitions") or {}).items() if isinstance(v, str)}
    return {x: d.get(x, "") for x in names}


def union(pairs, ids):
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    groups = collections.defaultdict(list)
    for i in ids:
        groups[find(i)].append(i)
    return list(groups.values())


def daily_rows(rid, js, stats):
    """gdaily: up to 8 short texts sharing one entity schema; one NER row per text (types of the
    schema absent from a text are explicit negatives: the generator marked every mention)."""
    if not isinstance(js, dict) or not isinstance(js.get("texts"), list):
        stats["gdaily_parse_fail"] += 1
        return []
    etypes = {clean_type(k): str(v) for k, v in (js.get("entity_types") or {}).items() if clean_type(k)}
    raw = js.get("mentions") or {}
    if isinstance(raw, list):
        raw = {str(x.get("id", "")): x for x in raw if isinstance(x, dict)}
    mt = {str(k): clean_type(v.get("type", "") if isinstance(v, dict) else v) for k, v in raw.items()}
    parsed = []
    for k, marked in enumerate(js["texts"]):
        if not isinstance(marked, str) or not marked.strip():
            continue
        text, spans, dup, leftover = unmark(marked.strip())
        if leftover:
            stats["gdaily_bad_markup"] += 1
            continue
        parsed.append((k, text, spans))
    # markers numbered per text or by kind (P1, L1, 1, 2): map to M1..Mn by order over all texts
    order = [(k, m) for k, _, sp in parsed for m in sorted(sp, key=lambda m: sp[m])]
    mp = None if sum(m in mt for _, m in order) >= len(order) / 2 else remap_ids([f"{k}/{m}" for k, m in order], mt)
    if mp:
        stats["gdaily_ids_remapped_by_order"] += 1
        parsed = [(k, text, {mp[f"{k}/{m}"]: s for m, s in sp.items()}) for k, text, sp in parsed]
    out = []
    for k, text, spans in parsed:
        ents = {}
        for m in spans:
            t = mt.get(m.split("~")[0])
            if t in etypes:
                ents[m] = t
            else:
                stats["gdaily_mention_type_unknown"] += 1
        if len(ents) < len(spans) / 2:
            stats["gdaily_id_mismatch"] += 1
            continue
        out.append({"id": f"{rid}:{k}", "text": text, "spans": {m: list(spans[m]) for m in spans}, "ents": ents,
                    "rels": [], "etypes": etypes, "rtypes": {}})
        stats["gdaily_texts"] += 1
    return out


def cand_rows(task, rid, js, stats):
    """Candidate rows of one generation: {id, text, ents {mid: (s, e, label)}, rels [(h, t, label)],
    etypes {label: def}, rtypes {label: def}, sym}. Mention ids index `spans` of the text."""
    if task == "gdaily":
        return daily_rows(rid, js, stats)
    if not isinstance(js, dict) or not isinstance(js.get("text"), str):
        stats[f"{task}_parse_fail"] += 1
        return []
    text, spans, dup, leftover = unmark(js["text"].strip())
    if leftover or not spans:
        stats[f"{task}_bad_markup"] += 1
        return []
    stats[f"{task}_repeated_ids"] += bool(dup)
    raw = js.get("mentions") or {}
    if isinstance(raw, list):              # [{"id": "M1", "type": ...}] instead of {"M1": ...}
        raw = {str(x.get("id") or x.get("mention") or ""): x for x in raw if isinstance(x, dict)}
    if not isinstance(raw, dict):
        raw = {}
    mp = remap_ids(sorted(spans, key=lambda m: spans[m]), raw)
    if mp:
        spans, dup = {mp[m]: s for m, s in spans.items()}, {}
        stats[f"{task}_ids_remapped_by_order"] += 1
    mtype = {}
    for mid, v in raw.items():
        t = clean_type(v.get("type", "") if isinstance(v, dict) else v)
        if mid in spans and t:
            mtype[mid] = t
            for x in dup.get(mid, []):
                mtype[x] = t
    same = [[m] + xs for m, xs in dup.items()]          # repeats of one id: the same thing
    if len(mtype) < len(spans) / 2:                     # the mention list uses other ids than the text
        stats[f"{task}_id_mismatch"] += 1
        return []
    order = sorted(spans, key=lambda m: spans[m])
    rels = []
    for x in js.get("relations") or []:
        if isinstance(x, dict):
            rels.append((str(x.get("head", "")).strip(), clean_type(x.get("relation", "")),
                         str(x.get("tail", "")).strip()))
    base = {"text": text, "spans": {m: list(spans[m]) for m in spans}}
    out = []
    if task == "gcoref":
        names = ["pronoun", "person name", "mention", "refers to", "coreference"]
        d = defs_of(js, names)
        groups = [[str(m) for m in c if str(m) in spans] for c in js.get("entities") or [] if isinstance(c, list)]
        pairs = [(c[0], m) for c in groups + same for m in c[1:]]
        clusters = [sorted(c, key=lambda m: spans[m]) for c in union(pairs, order) if len(c) >= 2]
        refers, coref = [], []
        for c in clusters:
            for k, m in enumerate(c):
                if k:
                    coref.append((m, c[k - 1], "coreference"))
                if mtype.get(m) == "pronoun":
                    refers += [(m, x, "refers to") for x in c if mtype.get(x) == "person name"]
        gap_ents = {m: "pronoun" if mtype[m] == "pronoun" else "person name" for m in order
                    if mtype.get(m) in ("pronoun", "person name")}
        if refers:
            out.append(dict(base, id=f"{rid}:refers", ents=gap_ents, rels=refers,
                            etypes={"pronoun": d["pronoun"], "person name": d["person name"]},
                            rtypes={"refers to": d["refers to"]}))
        if coref:
            out.append(dict(base, id=f"{rid}:coref", ents={m: "mention" for m in order}, rels=coref,
                            etypes={"mention": d["mention"]}, rtypes={"coreference": d["coreference"]}))
    elif task in ("gopin", "gopin2"):
        d = defs_of(js, ["aspect term", "opinion term"] + SENTIMENTS)
        ents = {m: t for m, t in mtype.items() if t in ("aspect term", "opinion term")}
        rr = [(h, t, lab) for h, lab, t in rels if lab in SENTIMENTS and ents.get(h) == "aspect term"
              and ents.get(t) == "opinion term"]
        stats["gopin_rel_dropped"] += len(rels) - len(rr)
        if ents:
            out.append(dict(base, id=rid, ents=ents, rels=rr,
                            etypes={k: d[k] for k in ("aspect term", "opinion term")},
                            rtypes={k: d[k] for k in SENTIMENTS}))
    elif task == "gtemp":
        d = defs_of(js, ["event", "time expression"] + TEMP_RELATIONS)
        ents = {m: t for m, t in mtype.items() if t in ("event", "time expression")}
        good = [(h, t, lab) for h, lab, t in rels if lab in TEMP_RELATIONS and h in ents and t in ents and h != t]
        stats["gtemp_rel_dropped"] += len(rels) - len(good)
        events = [m for m in ents if ents[m] == "event"]
        groups = union([(h, t) for h, t, lab in good if lab == "coreference"]
                       + [(c[0], m) for c in same for m in c[1:]], list(ents))
        first = {}
        for g in groups:
            f = min(g, key=lambda m: spans[m])
            for m in g:
                first[m] = f
        rr = {(m, first[m], "coreference") for m in events if first[m] != m}
        rr |= {(first[h], first[t], lab) for h, t, lab in good if lab != "coreference" and first[h] != first[t]}
        out.append(dict(base, id=rid, ents=ents, rels=sorted(rr),
                        etypes={k: d[k] for k in ("event", "time expression")},
                        rtypes={k: d[k] for k in TEMP_RELATIONS}))
    elif task in ("gment", "gmisc"):
        if task == "gment":
            names = list(js["_req"]["etypes"])
        else:
            names = [clean_type(k) for k in (js.get("definitions") or {}) if clean_type(k)]
            if "miscellaneous" not in names:
                names.append("miscellaneous")
        d = defs_of(js, names)
        ents = {m: t for m, t in mtype.items() if t in names}
        stats[f"{task}_mentions_type_unknown"] += len(mtype) - len(ents)
        if ents:
            out.append(dict(base, id=rid, ents=ents, rels=[], etypes={k: d[k] for k in names}, rtypes={}))
    elif task in ("gslot", "gslot2", "gmeas"):
        req_e, req_r = js["_req"]["etypes"], js["_req"]["rtypes"]
        d = defs_of(js, req_e + req_r)
        ents = {m: t for m, t in mtype.items() if t in req_e}
        rr = [(h, t, lab) for h, lab, t in rels if lab in req_r and h in ents and t in ents and h != t]
        stats["gslot_rel_dropped"] += len(rels) - len(rr)
        present = sorted(set(ents.values()))
        rng = random.Random(rid)
        absent = [t for t in req_e if t not in present]
        rng.shuffle(absent)
        if ents:
            out.append(dict(base, id=rid, ents=ents, rels=rr, etypes={k: d[k] for k in present + absent[:3]},
                            rtypes={k: d[k] for k in req_r}))
    else:   # gbio
        req_e, req_r = js["_req"]["etypes"], js["_req"]["rtypes"]
        d = defs_of(js, req_e + req_r)
        concept = {}
        for mid, v in raw.items():
            if mid in spans:
                c = str(v.get("concept", "")).strip() if isinstance(v, dict) else ""
                for x in [mid] + dup.get(mid, []):
                    concept[x] = c or mid                  # no concept id: a repeated marker id is one
        ents = {m: t for m, t in mtype.items() if t in req_e}
        firsts = {}
        for m in sorted(ents, key=lambda m: spans[m]):
            firsts.setdefault(concept.get(m, m), m)
        rr = set()
        for h, lab, t in rels:
            h, t = concept.get(h, h), concept.get(t, t)      # relations may name mentions, not concepts
            if lab in req_r and h in firsts and t in firsts and h != t:
                a, b = sorted((firsts[h], firsts[t]), key=lambda m: spans[m])
                rr.add((a, b, lab))
            else:
                stats[f"gbio_rel_dropped_{'label' if lab not in req_r else 'ends'}"] += 1
        if ents:
            out.append(dict(base, id=rid, ents=ents, rels=sorted(rr), etypes={k: d[k] for k in req_e},
                            rtypes={k: d[k] for k in req_r}, sym=True))
    return out


def numbered(text, spans):
    """Text with every mention shown as [mention](M<n>) for the judge."""
    out, pos = [], 0
    for m, (s, e) in sorted(spans.items(), key=lambda kv: kv[1]):
        if s < pos:
            continue
        out.append(text[pos:s] + f"[{text[s:e]}]({m})")
        pos = e
    return "".join(out) + text[pos:]


def assemble(args):
    from .ee_gen import load_outputs, parse
    stats, cands, jreqs = collections.Counter(), [], []
    reqs = {t: {r["id"]: r for r in read_jsonl(S2 / f"{t}_requests.jsonl")} for t in TASKS
            if (S2 / f"{t}_requests.jsonl").exists()}
    for task in reqs:
        for rid, o in load_outputs(task).items():
            js = parse(o["output"])
            if isinstance(js, dict):
                js["_req"] = reqs[task].get(rid, {})
            try:
                rows = cand_rows(task, rid, js, stats)
            except (AttributeError, TypeError, KeyError, ValueError, IndexError):   # malformed answer
                stats[f"{task}_malformed"] += 1
                rows = []
            for r in rows:
                r["task"] = task
                cands.append(r)
                stats[f"{task}_rows"] += 1
                stats[f"{task}_relations"] += len(r["rels"])
    for r in cands:
        shown = numbered(r["text"], {m: r["spans"][m] for m in r["ents"]})
        if r["task"] in MENTION_JUDGED:
            ms = sorted(r["ents"], key=lambda m: r["spans"][m])
            for k in range(0, len(ms), 40):
                lines = "\n".join(f'E{j + 1}: {m} "{r["text"][slice(*r["spans"][m])]}" is "{r["ents"][m]}"'
                                  for j, m in enumerate(ms[k:k + 40]))
                jreqs.append({"id": f"{r['id']}#m{k}", "kind": "mention", "text": shown,
                              "etypes": sorted(r["etypes"].items()), "cands": lines})
        for k in range(0, len(r["rels"]), 40):
            part = r["rels"][k:k + 40]
            lines = "\n".join(
                f'R{j + 1}: head {h} "{r["text"][slice(*r["spans"][h])]}" -> "{lab}" -> tail {t} '
                f'"{r["text"][slice(*r["spans"][t])]}"' for j, (h, t, lab) in enumerate(part))
            jreqs.append({"id": f"{r['id']}#{k}", "text": shown, "rtypes": sorted(r["rtypes"].items()),
                          "cands": lines, "links": any(lab in LINK_LABELS for _, _, lab in part)})
    R5.mkdir(parents=True, exist_ok=True)
    write_jsonl(R5 / "candidates.jsonl", cands)
    write_jsonl(S2 / "r5jd_requests.jsonl", jreqs)
    stats["judge_requests"] = len(jreqs)
    print(json.dumps(dict(sorted(stats.items()))))
    for task in TASKS:
        for x in [c for c in cands if c["task"] == task][:args.show]:
            t = x["text"]
            print(task, "|", t[:250].replace("\n", " "))
            print("   E:", [(t[slice(*x["spans"][m])], lab) for m, lab in list(x["ents"].items())[:10]])
            print("   R:", [(t[slice(*x["spans"][h])], lab, t[slice(*x["spans"][tt])]) for h, tt, lab in x["rels"][:8]])


# ---------------------------------------------------------------- apply

def to_row(c, keep, stats):
    task, sp = c["task"], c["spans"]
    proposed = collections.Counter(lab for _, _, lab in c["rels"])
    bad = set(c.get("bad", []))
    kept = [x for k, x in enumerate(c["rels"]) if keep.get((c["id"], k), False) and not {x[0], x[1]} & bad]
    stats[f"{task}_relations_kept"] += len(kept)
    stats[f"{task}_relations_rejected"] += len(c["rels"]) - len(kept)
    has = collections.Counter(lab for _, _, lab in kept)
    ents = sorted({(sp[m][0], sp[m][1], lab) for m, lab in c["ents"].items()})
    present = {lab for _, _, lab in ents}
    queries = [{"label": t, "description": d, "status": "positive" if t in present else "explicit_absent",
                "description_origin": "llm_authored", "supervision_origin": "llm_generated"}
               for t, d in c["etypes"].items()]
    rq = [{"label": t, "description": d,
           "status": "positive" if has[t] else ("unknown" if proposed[t] else "explicit_absent")}
          for t, d in c["rtypes"].items()]
    rels = set()
    for h, t, lab in kept:
        rels.add((tuple(sp[h]), tuple(sp[t]), lab))
        if c.get("sym"):
            rels.add((tuple(sp[t]), tuple(sp[h]), lab))
    src = SOURCE[task]
    return {"id": f"{src}:{c['id']}", "source": src, "text": c["text"], "queries": queries,
            "entities": [{"start": a, "end": b, "label": lab} for a, b, lab in ents], "rel_queries": rq,
            "relations": [{"head": list(h), "tail": list(t), "label": lab} for h, t, lab in sorted(rels)]}


def apply(args):
    from ..data.prepare_mix import contamination_filter
    from ..data.re_prep import eval_texts
    from .ee_gen import load_outputs, parse
    stats = collections.Counter()
    cands = read_jsonl(R5 / "candidates.jsonl")
    keep = {}
    answers = load_outputs("r5jd")
    for c in cands:
        if c["task"] in MENTION_JUDGED:     # rejected mentions leave the row with their relations
            ms = sorted(c["ents"], key=lambda m: c["spans"][m])
            bad = set()
            for k in range(0, len(ms), 40):
                o = answers.get(f"{c['id']}#m{k}")
                js = parse(o["output"]) if o else None
                for j, m in enumerate(ms[k:k + 40]):
                    v = str(js.get(f"E{j + 1}", "")).strip().lower() if isinstance(js, dict) else "yes"
                    if not v.startswith("y"):
                        bad.add(m)
            stats[f"{c['task']}_mentions_rejected"] += len(bad)
            stats[f"{c['task']}_mentions"] += len(ms)
            c["ents"] = {m: t for m, t in c["ents"].items() if m not in bad}
            c["bad"] = sorted(bad)
        for k in range(0, len(c["rels"]), 40):
            o = answers.get(f"{c['id']}#{k}")
            js = parse(o["output"]) if o else None
            if js is None:
                stats["judge_missing"] += 1
            for j in range(min(40, len(c["rels"]) - k)):
                v = str((js or {}).get(f"R{j + 1}", "")).strip().lower() if isinstance(js, dict) else ""
                keep[(c["id"], k + j)] = v.startswith("y")
    by = collections.defaultdict(list)
    for c in cands:
        r = to_row(c, keep, stats)
        if r["relations"] or c["task"] in ("gopin", "gslot", "gslot2", "gdaily", "gment", "gmisc", "gopin2",
                                           "gmeas"):   # rows without a relation teach "none"
            by[SOURCE[c["task"]]].append(r)
    texts = eval_texts()
    for src, rows in by.items():
        random.Random(args.seed).shuffle(rows)
        rows = contamination_filter(rows, texts, stats, src)
        n = args.n_dev
        parts = {"dev_select": rows[:n], "dev_calib": rows[n:2 * n], "test": rows[2 * n:3 * n], "train": rows[3 * n:]}
        for sp, rs in parts.items():
            write_jsonl(DATA_OUT / "re" / src / f"{sp}.jsonl", rs)
            stats[f"{src}_{sp}"] = len(rs)
    (R5 / "apply_summary.json").write_text(json.dumps(dict(sorted(stats.items())), indent=1))
    print(json.dumps(dict(sorted(stats.items()))))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--n-coref", type=int, default=40000)
    p.add_argument("--n-opin", type=int, default=40000)
    p.add_argument("--n-temp", type=int, default=40000)
    p.add_argument("--n-slot", type=int, default=40000)
    p.add_argument("--n-bio", type=int, default=30000)
    p.add_argument("--n-daily", type=int, default=0)
    p.add_argument("--n-ment", type=int, default=0)
    p.add_argument("--n-misc", type=int, default=0)
    p.add_argument("--n-slot2", type=int, default=0)
    p.add_argument("--n-opin2", type=int, default=0)
    p.add_argument("--n-bio2", type=int, default=0)
    p.add_argument("--n-meas", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("assemble")
    p.add_argument("--show", type=int, default=2)
    p = sub.add_parser("apply")
    p.add_argument("--n-dev", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    {"prepare": prepare, "assemble": assemble, "apply": apply}[args.cmd](args)


if __name__ == "__main__":
    main()
