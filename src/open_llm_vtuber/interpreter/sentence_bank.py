# src/open_llm_vtuber/interpreter/sentence_bank.py
"""The doctor's sentence bank and the matcher that finds a sentence in it.

Nothing here translates. A doctor's Thai (typed, picked or spoken) is compared
with the Thai of the standard sentences; the Pattani text that is shown and
played is the one a person wrote for that sentence
(pattani_dataset/standard_sentences.jsonl, written by build_phrasebook.py).
"""

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from pythainlp.tokenize import word_tokenize
from pythainlp.util import thaiword_to_num
from rapidfuzz import fuzz

from ..translate.pattani_script import thai_key

THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
NUMBER_WORD = re.compile(
    r"(ศูนย์|หนึ่ง|เอ็ด|สอง|ยี่|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า|สิบ|ร้อย|พัน|หมื่น|แสน|ล้าน)+"
)
# Words that say how much or how often. A sentence with one of these, or with
# a number, is never offered for a quick pick: 2 times a day and 3 times a day
# look alike and must not be confused.
DOSE_WORDS = ("ครั้ง", "วันละ", "เม็ด", "ช้อน", "มื้อ", "ชั่วโมง", "ทุก", "ซีซี", "มิลลิ")
NOT_FOUND = "ไม่พบประโยค กรุณาใช้ล่าม"
# Spellings of the same question word; speech recognition writes any of them.
QUESTION_WORD = re.compile("มั้ย|มั๊ย|ไม๊|ไม้ย")


def normalize(text: str) -> str:
    """Thai text without spaces, punctuation and polite ending, digits as 0-9."""
    return thai_key(QUESTION_WORD.sub("ไหม", text.translate(THAI_DIGITS)))


# Words that make a request polite without changing what is asked:
# ช่วยอ้าปากหน่อย is the standard sentence อ้าปาก.
SOFTENERS = {"ช่วย", "กรุณา", "โปรด", "หน่อย", "นะ", "น่ะ", "เลย", "ด้วย", "ที", "สิ", "ซิ"}


def tokenize(text: str) -> list[str]:
    """The words of a sentence that carry its meaning."""
    words = [t for t in word_tokenize(normalize(text), engine="newmm") if t.strip()]
    return [t for t in words if t not in SOFTENERS] or words


def numbers_in(tokens: list[str]) -> list[int]:
    """Every number said in a sentence, digits and number words alike."""
    found = []
    for token in tokens:
        if token.isdigit():
            found.append(int(token))
        elif NUMBER_WORD.fullmatch(token):
            try:
                found.append(thaiword_to_num(token))
            except (ValueError, KeyError):
                pass
    return sorted(found)


@dataclass
class Sentence:
    id: str
    category: str
    thai: str
    pattani: str
    variants: list[str] = field(default_factory=list)
    source: str = ""
    # numbers and dose words make a sentence one the doctor must read closely
    numbers: list[int] = field(default_factory=list)
    check_numbers: bool = False


@dataclass
class Candidate:
    id: str
    score: float
    thai: str
    # (c) of the safety rules: must be picked deliberately, never by default
    check_numbers: bool = False
    # the number that was said is not the number in this sentence
    number_mismatch: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "score": round(self.score, 3),
            "thai": self.thai,
            "check_numbers": self.check_numbers,
            "number_mismatch": self.number_mismatch,
        }


class SentenceBank:
    """Standard sentences, and matching of free Thai text against them."""

    def __init__(
        self,
        sentences_path: str = "pattani_dataset/standard_sentences.jsonl",
        threshold_path: str = "pattani_dataset/matcher_threshold.json",
    ):
        self.sentences: dict[str, Sentence] = {}
        # one entry per wording: (sentence id, normalized text, tf-idf vector)
        self._wordings: list[tuple[str, str, dict[str, float]]] = []
        self._idf: dict[str, float] = {}
        self._load(Path(sentences_path))

        # (d) of the safety rules: the threshold comes from test data
        # (calibrate_matcher.py), it is not a number picked by hand
        self.threshold: float | None = None
        path = Path(threshold_path)
        if path.exists():
            with path.open(encoding="utf-8") as f:
                self.threshold = float(json.load(f)["threshold"])
        else:
            logger.warning(
                f"No matcher threshold at '{threshold_path}'. Run calibrate_matcher.py; "
                "until then no sentence is offered for free text."
            )

    def _load(self, path: Path) -> None:
        if not path.exists():
            logger.warning(
                f"Sentence bank not found at '{path}'. Run build_phrasebook.py."
            )
            return
        tokens_of: list[tuple[str, str, list[str]]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                sentence = Sentence(
                    id=entry["id"],
                    category=entry["category"],
                    thai=entry["thai"],
                    pattani=entry["pattani"],
                    variants=entry.get("variants", []),
                    source=entry.get("source", ""),
                )
                for wording in [sentence.thai, *sentence.variants]:
                    tokens = tokenize(wording)
                    tokens_of.append((sentence.id, "".join(tokens), tokens))
                    if wording == sentence.thai:
                        sentence.numbers = numbers_in(tokens)
                        sentence.check_numbers = bool(sentence.numbers) or any(
                            word in wording for word in DOSE_WORDS
                        )
                self.sentences[sentence.id] = sentence

        documents = len(tokens_of)
        seen_in = Counter(token for _, _, tokens in tokens_of for token in set(tokens))
        self._idf = {
            token: math.log((documents + 1) / (count + 1)) + 1
            for token, count in seen_in.items()
        }
        self._wordings = [
            (ident, text, self._vector(tokens)) for ident, text, tokens in tokens_of
        ]
        logger.info(f"Sentence bank loaded: {len(self.sentences)} standard sentences.")

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        """Unit-length tf-idf vector over words; unknown words weigh the most."""
        default = math.log(len(self._wordings) + 2) + 1 if self._wordings else 1.0
        weights = {
            token: count * self._idf.get(token, default)
            for token, count in Counter(tokens).items()
        }
        length = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        return {token: w / length for token, w in weights.items()}

    def score(self, text: str, tokens: list[str], wording: tuple) -> float:
        """How alike two sentences are, 0..1: letters and words count equally."""
        _, other, other_vector = wording
        vector = self._vector(tokens)
        words = sum(w * other_vector.get(token, 0.0) for token, w in vector.items())
        letters = fuzz.ratio(text, other) / 100
        return (letters + words) / 2

    def categories(self) -> list[dict]:
        counts = Counter(s.category for s in self.sentences.values())
        return [{"name": name, "count": count} for name, count in counts.items()]

    def in_category(self, category: str) -> list[dict]:
        return [
            {"id": s.id, "thai": s.thai, "check_numbers": s.check_numbers}
            for s in self.sentences.values()
            if s.category == category
        ]

    def ranked(self, text: str) -> list[Candidate]:
        """Every sentence with its score for this text, best first."""
        tokens = tokenize(text)
        norm = "".join(tokens)
        if not norm:
            return []
        best: dict[str, float] = {}
        for wording in self._wordings:
            value = self.score(norm, tokens, wording)
            if value > best.get(wording[0], 0.0):
                best[wording[0]] = value
        said = numbers_in(tokens)
        candidates = []
        for ident, value in sorted(
            best.items(), key=lambda item: item[1], reverse=True
        ):
            sentence = self.sentences[ident]
            candidates.append(
                Candidate(
                    id=ident,
                    score=value,
                    thai=sentence.thai,
                    check_numbers=sentence.check_numbers or bool(said),
                    number_mismatch=said != sentence.numbers,
                )
            )
        return candidates

    def match(self, text: str, limit: int = 3) -> list[Candidate]:
        """The 1-3 standard sentences this text may mean; [] when none is
        close enough, in which case nothing may be played."""
        if self.threshold is None:
            return []
        return [c for c in self.ranked(text)[:limit] if c.score >= self.threshold]
