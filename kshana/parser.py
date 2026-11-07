"""Prompt parser (spaCy): objects, counts, colours, spatial relations, complexity."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import spacy

# Order follows the 80-class COCO index used by the detector.
COCO_INDEX = (
    "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,"
    "parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,"
    "handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,"
    "surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,"
    "broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,"
    "mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,"
    "teddy bear,hair drier,toothbrush"
).split(",")

SYNONYMS = {
    "man": "person", "woman": "person", "boy": "person", "girl": "person", "child": "person", "kid": "person",
    "people": "person", "lady": "person", "guy": "person", "baby": "person", "astronaut": "person",
    "bike": "bicycle", "motorbike": "motorcycle", "scooter": "motorcycle", "plane": "airplane",
    "aeroplane": "airplane", "jet": "airplane", "taxi": "car", "automobile": "car", "sofa": "couch",
    "television": "tv", "phone": "cell phone", "smartphone": "cell phone", "mobile phone": "cell phone",
    "table": "dining table", "desk": "dining table", "plant": "potted plant", "houseplant": "potted plant",
    "ball": "sports ball", "football": "sports ball", "basketball": "sports ball", "soccer ball": "sports ball",
    "mug": "cup", "teacup": "cup", "ship": "boat", "sailboat": "boat", "yacht": "boat", "puppy": "dog",
    "retriever": "dog", "labrador": "dog", "poodle": "dog", "kitten": "cat", "tiger cat": "cat",
    "parrot": "bird", "sparrow": "bird", "eagle": "bird", "pigeon": "bird", "owl": "bird", "duck": "bird",
    "pony": "horse", "lamb": "sheep", "calf": "cow", "bull": "cow", "doughnut": "donut", "purse": "handbag",
    "rucksack": "backpack", "fridge": "refrigerator", "hairdryer": "hair drier", "teddy": "teddy bear",
    "couch": "couch", "lorry": "truck", "van": "truck", "racket": "tennis racket", "sandwiches": "sandwich",
}

COLOURS = {
    "red": "red", "crimson": "red", "scarlet": "red", "orange": "orange", "yellow": "yellow", "golden": "yellow",
    "gold": "yellow", "green": "green", "emerald": "green", "blue": "blue", "navy": "blue", "cyan": "cyan",
    "teal": "cyan", "turquoise": "cyan", "purple": "purple", "violet": "purple", "pink": "pink",
    "magenta": "pink", "brown": "brown", "black": "black", "white": "white", "gray": "gray", "grey": "gray",
    "silver": "gray",
}

NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "couple": 2, "pair": 2,
}

# Phrase -> relation. Only left/right/above/below can be checked by GenEval's geometry.
SPATIAL = {
    "to the left of": "left of", "left of": "left of", "to the right of": "right of", "right of": "right of",
    "on top of": "above", "above": "above", "over": "above", "under": "below", "underneath": "below",
    "beneath": "below", "below": "below", "next to": "next to", "beside": "next to", "behind": "behind",
    "in front of": "in front of",
}
VERIFIABLE_RELATIONS = {"left of", "right of", "above", "below"}

# Words that parse as nouns but do not name something depicted.
NOT_OBJECTS = set(
    "picture photo photograph image painting drawing illustration render style close closeup close-up "
    "shot view scene background foreground left right top bottom side front back middle center centre "
    "day night morning evening afternoon detail quality art artwork portrait lot bunch group".split()
)


@dataclass
class ObjectSpec:
    noun: str
    coco: str | None
    count: int | None = None
    colour: str | None = None
    start: int = 0  # token index, used to bind relations


@dataclass
class Relation:
    subject: int  # index into objects
    relation: str
    target: int


@dataclass
class ParsedPrompt:
    text: str
    objects: list[ObjectSpec] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    colours: list[str] = field(default_factory=list)
    numbers: list[int] = field(default_factory=list)
    spatial_words: list[str] = field(default_factory=list)
    complexity: float = 0.0

    @property
    def names_colours(self) -> bool:
        return bool(self.colours)


_nlp = None


def _load():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["ner"])
    return _nlp


def _to_coco(name: str) -> str | None:
    name = name.lower()
    if name in COCO_INDEX:
        return name
    return SYNONYMS.get(name)


def _number(tok) -> int | None:
    if tok.like_num:
        low = tok.lower_
        if low.isdigit():
            return int(low)
        return NUMBERS.get(low)
    return None


def complexity_score(n_obj: int, n_counts: int, n_colours: int, n_rel: int) -> float:
    """0 (one plain subject) to 1 (crowded multi-constraint scene)."""
    raw = 0.2 * max(n_obj - 1, 0) + 0.2 * n_counts + 0.15 * n_colours + 0.25 * n_rel
    return round(min(raw, 1.0), 3)


def parse(text: str) -> ParsedPrompt:
    doc = _load()(text)
    objects: list[ObjectSpec] = []
    consumed: set[int] = set()

    # Multi-word names first ("traffic light", "mobile phone"), longest match wins.
    lowered = [t.lemma_.lower() for t in doc]
    multi = sorted(
        {n for n in [*COCO_INDEX, *SYNONYMS] if " " in n}, key=lambda s: -len(s.split())
    )
    for name in multi:
        parts = name.split()
        for i in range(len(doc) - len(parts) + 1):
            if any(j in consumed for j in range(i, i + len(parts))):
                continue
            window = lowered[i : i + len(parts)]
            exact = [t.lower_ for t in doc[i : i + len(parts)]]
            if window == parts or exact == parts:
                objects.append(ObjectSpec(noun=name, coco=_to_coco(name), start=i + len(parts) - 1))
                consumed.update(range(i, i + len(parts)))

    for tok in doc:
        if tok.i in consumed or tok.pos_ not in {"NOUN", "PROPN"}:
            continue
        # "golden retriever puppy": the head noun is the object.
        if tok.dep_ == "compound" and tok.head.pos_ in {"NOUN", "PROPN"} and _to_coco(tok.lemma_) is None:
            continue
        if tok.dep_ == "compound" and tok.head.pos_ in {"NOUN", "PROPN"} and _to_coco(tok.head.lemma_):
            continue
        low = tok.lemma_.lower()
        if low in NOT_OBJECTS or not tok.is_alpha:
            continue
        if low in COLOURS and _to_coco(low) is None:  # "orange" alone is the fruit
            continue
        objects.append(ObjectSpec(noun=low, coco=_to_coco(low), start=tok.i))

    objects.sort(key=lambda o: o.start)

    numbers: list[int] = []
    colours: list[str] = []
    object_tokens = {o.start for o in objects}
    for tok in doc:
        low = tok.lower_
        if low in COLOURS and tok.pos_ in {"ADJ", "NOUN", "PROPN"} and tok.i not in object_tokens:
            colours.append(COLOURS[low])
    for obj in objects:
        head = doc[obj.start]
        for child in head.children:
            n = _number(child) if child.dep_ == "nummod" else None
            if n is not None:
                obj.count = n
                numbers.append(n)
            low = child.lower_
            # "golden retriever", "golden hour": counts as naming a colour, but is not a colour check.
            if low in COLOURS and low != "golden" and child.dep_ in {"amod", "compound", "nmod"} and obj.colour is None:
                obj.colour = COLOURS[low]

    relations: list[Relation] = []
    spatial_words: list[str] = []
    lower_text = " ".join(t.lower_ for t in doc)
    taken: list[tuple[int, int]] = []
    for phrase in sorted(SPATIAL, key=len, reverse=True):
        for m in re.finditer(rf"\b{re.escape(phrase)}\b", lower_text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue  # "left of" inside "to the left of"
            taken.append((m.start(), m.end()))
            # Char offset in the joined text back to a token index.
            start_tok = lower_text[: m.start()].count(" ")
            end_tok = start_tok + phrase.count(" ")
            before = [i for i, o in enumerate(objects) if o.start < start_tok]
            after = [i for i, o in enumerate(objects) if o.start > end_tok]
            spatial_words.append(phrase)
            if before and after:
                relations.append(Relation(before[-1], SPATIAL[phrase], after[0]))

    score = complexity_score(len(objects), len(numbers), len(colours), len(relations))
    return ParsedPrompt(
        text=text,
        objects=objects,
        relations=relations,
        colours=colours,
        numbers=numbers,
        spatial_words=spatial_words,
        complexity=score,
    )


def rewordings(parsed: ParsedPrompt) -> list[str]:
    """Rewordings tried alongside the original. A one-step model binds colours and layout far
    better when the contested object comes first and the layout is stated separately.
    Candidates are always scored against the original prompt."""
    objects = parsed.objects
    if len(objects) < 2:
        return []
    relation = next((r for r in parsed.relations if r.relation in VERIFIABLE_RELATIONS), None)
    if relation is None and sum(1 for o in objects if o.colour) < 2:
        return []

    def name(o: ObjectSpec) -> str:
        if o.count and o.count > 1:
            return f"{o.count} {o.noun}"
        words = " ".join(w for w in (o.colour, o.noun) if w)
        return f"{'an' if words[:1] in 'aeiou' else 'a'} {words}"

    def keep_scene(line: str, noun: str) -> str:
        lower = parsed.text.lower()
        at = lower.rfind(noun)
        tail = lower[at + len(noun):].strip() if at >= 0 else ""
        return f"{line}, {tail}" if tail else line

    if relation is not None:
        subject, target = objects[relation.subject], objects[relation.target]
        a, b = name(subject), name(target)
        lines = {
            "left of": [f"{b}, {a} beside it on the left", f"{b} and {a}, the {subject.noun} is on the left", f"{b} on the right, {a} on the left"],
            "right of": [f"{b}, {a} beside it on the right", f"{b} and {a}, the {subject.noun} is on the right", f"{b} on the left, {a} on the right"],
            "above": [f"{b}, {a} above it", f"{b} below, {a} above"],
            "below": [f"{b}, {a} below it", f"{b} above, {a} below"],
        }[relation.relation]
        return [keep_scene(line, target.noun) for line in lines]
    a, b = name(objects[0]), name(objects[1])
    return [keep_scene(f"{b} and {a}", objects[1].noun), keep_scene(f"{a}, {b}, side by side", objects[1].noun)]
