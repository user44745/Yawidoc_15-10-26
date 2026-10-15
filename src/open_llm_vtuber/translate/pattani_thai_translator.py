# src/open_llm_vtuber/translate/pattani_thai_translator.py
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger
from openai import OpenAI

from .nllb_translator import NLLBTranslator
from .pattani_script import NOT_TEXT, collapse_repeats
from .thai_romanize import thai_to_latin
from .translate_interface import TranslateInterface

THAI_CHAR = re.compile(r"[฀-๿]")
LATIN_CHAR = re.compile(r"[A-Za-z]")
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
WORD = re.compile(r"[a-z0-9]+")

# Whisper writes this for unclear audio; nobody says it in a hospital.
RECOGNIZER_ARTEFACTS = {"terimakasihkeranamenonton"}
# Spoken to the Thai listener when what was heard cannot be understood.
REPEAT_REQUEST = "ฟังไม่ชัด กรุณาพูดอีกครั้ง"
UNCLEAR = "???"

# A correction or a phrase-table match must never flip one of these.
NEGATIONS = {"tak", "tidak", "tiada", "takde", "bukan", "jangan", "belum", "dok"}
CRITICAL_WORDS = {
    "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "lapan", "sembilan",
    "sepuluh", "sekali", "kanan", "kiri", "sebelum", "selepas", "pagi", "malam",
}  # fmt: skip

DIALECT_NOTE = (
    "Pattani dialect words: dok / tak = tidak, ado = ada, gapo = apa, mano = mana, "
    "pala / palo = kepala, peghut = perut, demey = demam, ubak = ubat, make = makan, "
    "loni / lor ni = sekarang, doh / dah = sudah, mari = datang, hok = yang, nok = nak."
)

FIX_SYSTEM_PROMPT = """\
You fix speech recognition errors in Malay sentences spoken in a hospital in southern
Thailand (Standard Malay or Pattani dialect, sometimes with a Thai accent).
The recognizer often writes a wrong word that sounds similar. Rewrite the sentence with
the words that were most likely said in a hospital conversation.
- Change only words that are clearly misheard; keep everything else, keep the word
  order, never add new information.
- Never add or remove a negation (tak, tidak, tiada, bukan, jangan, belum).
- If the sentence already makes sense, reply with it unchanged.
Phrases often said here: {phrases}
{dialect}
Reply with the corrected Malay sentence only."""

JUDGE_SYSTEM_PROMPT = """\
You check machine-translated Thai sentences from a conversation between a patient and a
nurse in a hospital.
Reply YES if the sentence is normal Thai that makes sense in such a conversation. Short
answers (yes, no, thanks, a number, a time, a body part) are fine.
Reply NO if the sentence is nonsense, garbled, has leftover foreign words, or talks about
something absurd for a hospital (for example courts, gods, stones, snacks, boxes).
Reply with YES or NO only."""

TRANSLATE_SYSTEM_PROMPT = """\
You are a medical interpreter in a hospital in southern Thailand. Translate the Malay
sentence (Standard Malay or Pattani dialect) into one short, natural, polite spoken Thai
sentence, as a nurse or patient would say it.
Translate only what is said; never add symptoms, numbers or details.
The hospital phrases you are given only show the wording used here. Do not answer with
one of them unless the sentence says the same thing.
If it cannot be understood at all, reply with exactly: ???
{dialect}
Reply with the Thai sentence only."""


def _key(text: str) -> str:
    return "".join(WORD.findall(text.lower()))


def _edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def _sound_similarity(a: str, b: str) -> float:
    """1.0 for the same letters, lower the more letters differ."""
    if not a or not b:
        return 0.0
    return 1 - _edit_distance(a, b) / max(len(a), len(b))


def _bigrams(text: str) -> set[str]:
    text = NOT_TEXT.sub("", text.lower())
    return {text[i : i + 2] for i in range(len(text) - 1)} or {text}


def _overlap(a: str, b: str) -> float:
    """Share of letter pairs two texts have in common, 0.0 to 1.0."""
    grams_a, grams_b = _bigrams(a), _bigrams(b)
    return 2 * len(grams_a & grams_b) / (len(grams_a) + len(grams_b))


def _marked(text: str, vocabulary: set[str]) -> list[str]:
    return sorted(
        w for w in WORD.findall(text.lower()) if w in vocabulary or w.isdigit()
    )


def _same_critical_words(a: str, b: str) -> bool:
    """Do both sentences carry the same negations, numbers, sides and times?"""
    return _marked(a, NEGATIONS) == _marked(b, NEGATIONS) and _marked(
        a, CRITICAL_WORDS
    ) == _marked(b, CRITICAL_WORDS)


def _first_line(answer: str | None) -> str:
    answer = THINK_BLOCK.sub("", answer or "")
    line = next((ln for ln in answer.splitlines() if ln.strip()), "")
    return line.strip().strip("\"'“”").strip()


class PattaniThaiTranslator(TranslateInterface):
    """Malay / Pattani Malay (as written by the speech recognizer) -> Thai.

    Built for hospital conversations, where a translation about the wrong
    topic is worse than none:

    1. Phrase table: what was heard, or something that sounds almost the same,
       is a known hospital phrase -> its vetted Thai.
    2. Optional LLM: correct misheard Malay words using the hospital context,
       then try the phrase table again.
    3. NLLB translates the Malay into Thai (zsm_Latn -> tha_Thai).
    4. Optional LLM: check that NLLB's Thai is something said in a hospital.
       If it is, it is used as is. If it is not (wrong topic, garbled,
       leftover foreign words), the LLM translates the sentence again as a
       hospital interpreter with the closest phrase-table entries as
       reference; if it cannot make sense of the sentence either, the
       listener is asked to have it repeated.

    Without the LLM (or when it fails) the NLLB translation is returned.
    """

    # how alike the letters of two sentences must be for a phrase-table match;
    # a long phrase is unlikely to resemble another by chance, so it may differ more
    CLOSE_MATCH = 0.85
    CLOSE_MATCH_LONG = 0.78
    LONG_PHRASE = 12  # letters
    # ... and how alike each single word must be
    CLOSE_WORD = 0.6
    # an LLM correction must still sound like what was heard: word by word
    # when the word count is unchanged, otherwise as a whole sentence
    # (a short word has few letters to go by, so it must be nearly the same)
    CORRECTED_WORD = 0.6
    CORRECTED_SHORT_WORD = 0.75
    SHORT_WORD = 4  # letters
    CORRECTED_SENTENCE = 0.75
    # two Thai translations sharing this much say the same thing
    AGREEMENT = 0.6
    # Thai letters per Malay letter beyond which a translation adds content
    MAX_LENGTH_RATIO = 2.5

    def __init__(
        self,
        phrases_path: str = "pattani_dataset/medical_phrases.jsonl",
        nllb_model: str = "facebook/nllb-200-distilled-600M",
        device: str = "cpu",
        llm_review: bool = True,
        llm_base_url: str = "http://localhost:11434/v1",
        llm_model: str = "qwen3:8b",
        llm_api_key: str = "ollama",
        llm_temperature: float = 0.0,
        num_examples: int = 6,
        **kwargs,
    ):
        """
        :param phrases_path: JSONL of hospital phrases, {"malay", "thai"}
        :param nllb_model: Hugging Face model name or local path of the NLLB model
        :param device: "cpu" or "cuda" for NLLB
        :param llm_review: let the LLM correct mishearings and re-translate
            sentences whose translation does not fit a hospital conversation
        :param num_examples: phrase-table entries shown to the LLM per request
        """
        self.phrases: list[tuple[str, str]] = []
        self.exact: dict[str, str] = {}
        self._load_phrases(phrases_path)

        self.nllb = NLLBTranslator(
            model_name=nllb_model,
            src_lang="zsm_Latn",
            tgt_lang="tha_Thai",
            device=device,
        )

        self.llm_review = llm_review
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.num_examples = num_examples
        self.client = (
            OpenAI(base_url=llm_base_url, api_key=llm_api_key, timeout=60)
            if llm_review
            else None
        )
        self.translate_prompt = TRANSLATE_SYSTEM_PROMPT.format(dialect=DIALECT_NOTE)

    def _load_phrases(self, phrases_path: str) -> None:
        path = Path(phrases_path)
        if not path.exists():
            logger.warning(f"Hospital phrase table not found at '{phrases_path}'.")
            return
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    self.phrases.append((entry["malay"], entry["thai"]))
                    self.exact.setdefault(_key(entry["malay"]), entry["thai"])
        logger.info(f"Hospital phrase table loaded: {len(self.phrases)} phrases.")

    def _lookup(self, heard: str) -> str | None:
        """Thai for a known hospital phrase that matches what was heard."""
        key = _key(heard)
        if key in self.exact:
            return self.exact[key]

        best, score = None, 0.0
        for malay, thai in self.phrases:
            similarity = _sound_similarity(key, _key(malay))
            if similarity > score:
                best, score = (malay, thai), similarity
        needed = (
            self.CLOSE_MATCH_LONG if len(key) >= self.LONG_PHRASE else self.CLOSE_MATCH
        )
        if not best or score < needed:
            return None
        malay, thai = best
        if not _same_critical_words(heard, malay):
            return None
        # every word of the phrase must be recognisable in what was heard, from
        # its first letter on: "tangan kiri" must not pass for "tangan kanan",
        # nor "selamat datang" for "selamat petang"
        heard_words = WORD.findall(heard.lower())
        heard_words += [a + b for a, b in zip(heard_words, heard_words[1:])]
        for word in WORD.findall(malay.lower()):
            # (an initial h is often not pronounced: hulurkan / ulurkan)
            start = word.lstrip("h")[:1]
            alike = (
                _sound_similarity(word, h)
                for h in heard_words
                if h.lstrip("h")[:1] == start
            )
            if max(alike, default=0.0) < self.CLOSE_WORD:
                return None
        logger.info(f"Heard '{heard}' taken as the hospital phrase '{malay}'.")
        return thai

    def _ask_llm(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.llm_temperature,
            seed=0,  # same input, same answer
            # qwen3 "thinks" by default, which costs ~10x the latency here
            extra_body={"reasoning_effort": "none"},
        )
        return _first_line(response.choices[0].message.content)

    def _fix_mishearings(self, heard: str) -> str:
        """Let the LLM replace misheard words; keep what was heard if in doubt."""
        # the known phrases that sound most like what was heard
        candidates = sorted(
            self.phrases,
            key=lambda pair: _sound_similarity(_key(heard), _key(pair[0])),
            reverse=True,
        )[: 2 * self.num_examples]
        prompt = FIX_SYSTEM_PROMPT.format(
            phrases="; ".join(malay for malay, _ in candidates), dialect=DIALECT_NOTE
        )
        fixed = self._ask_llm(prompt, heard)
        before, after = WORD.findall(heard.lower()), WORD.findall(fixed.lower())
        if len(before) == len(after):
            sounds_alike = all(
                _sound_similarity(a, b)
                >= (
                    self.CORRECTED_SHORT_WORD
                    if min(len(a), len(b)) <= self.SHORT_WORD
                    else self.CORRECTED_WORD
                )
                for a, b in zip(before, after)
            )
        else:
            sounds_alike = (
                _sound_similarity(_key(heard), _key(fixed)) >= self.CORRECTED_SENTENCE
            )
        if not LATIN_CHAR.search(fixed) or THAI_CHAR.search(fixed):
            reason = "not Malay"
        elif not _same_critical_words(heard, fixed):
            reason = "changes a negation, number, side or time"
        elif not sounds_alike:
            reason = "does not sound like what was heard"
        else:
            if _key(fixed) != _key(heard):
                logger.info(f"LLM corrected what was heard: {fixed}")
            return fixed
        logger.warning(f"LLM correction rejected ({reason}): {fixed!r}")
        return heard

    def _retranslate(self, malay: str) -> str | None:
        """Thai from the LLM, UNCLEAR, or None when its answer is unusable."""
        phrases = sorted(
            self.phrases, key=lambda pair: _overlap(malay, pair[0]), reverse=True
        )[: self.num_examples]
        lines = ["Hospital phrases (Malay => Thai):"]
        lines += [f"- {ms} => {th}" for ms, th in phrases]
        lines += ["", f"Malay sentence: {malay}", "Thai:"]

        answer = self._ask_llm(self.translate_prompt, "\n".join(lines))
        if answer.startswith(UNCLEAR):
            return UNCLEAR
        if (
            not THAI_CHAR.search(answer)
            or LATIN_CHAR.search(answer)
            or len(answer) > 4 * len(malay) + 30
            # a number nobody said ("Anambalas" -> "อายุ 18")
            or any(n not in malay for n in re.findall(r"\d+", answer))
        ):
            logger.warning(f"LLM translation rejected: {answer!r}")
            return None
        spelled = _key(thai_to_latin(answer))
        if (
            len(spelled) > 6
            and _sound_similarity(spelled, _key(malay)) >= self.AGREEMENT
        ):
            # it only spelled the sounds in Thai letters: it did not understand
            logger.warning(f"LLM could not make sense of {malay!r}: {answer!r}")
            return UNCLEAR
        return answer

    def _fits_hospital(self, malay: str, thai: str) -> bool:
        """Is this Thai something a patient or a nurse would say in a hospital?"""
        if LATIN_CHAR.search(thai):  # words NLLB could not translate
            return False
        if any(n not in malay for n in re.findall(r"\d+", thai)):
            return False  # a number nobody said ("Anambalas" -> "อายุ 18")
        return self._ask_llm(JUDGE_SYSTEM_PROMPT, thai).upper().startswith("YES")

    def translate(self, text: str) -> str:
        """
        Translate what the recognizer heard into Thai.

        Returns "" for recognizer artefacts (nothing is shown or spoken) and
        REPEAT_REQUEST when the sentence cannot be understood.
        """
        if not text or not text.strip():
            return text
        heard = collapse_repeats(text.strip())
        if _key(heard) in RECOGNIZER_ARTEFACTS:
            logger.info(f"Ignoring a recognizer artefact: {heard!r}")
            return ""
        if re.fullmatch(r"\d+[.,]?", heard):
            # a spoken number is written in digits; the Thai voice reads them
            return heard.strip(".,")

        known = self._lookup(heard)
        if known:
            return known
        if not self.llm_review:
            return self._nllb(heard)

        # NLLB (local model) starts on what was heard while the LLM (network
        # call) checks it for mishearings; usually nothing needs correcting
        with ThreadPoolExecutor(max_workers=1) as pool:
            early = pool.submit(self._nllb, heard)
            malay = heard
            try:
                malay = self._fix_mishearings(heard)
            except Exception as e:
                logger.error(f"LLM correction failed, using what was heard: {e}")
            if _key(malay) != _key(heard):
                known = self._lookup(malay)
                if known:
                    return known
            draft = early.result() if _key(malay) == _key(heard) else self._nllb(malay)
        logger.info(f"NLLB Thai draft: {draft}")

        try:
            if self._fits_hospital(malay, draft):
                return draft
            logger.info(
                "The draft does not fit a hospital conversation; translating again."
            )
            again = self._retranslate(malay)
        except Exception as e:
            logger.error(f"LLM review failed, using the NLLB draft: {e}")
            return draft

        if again == UNCLEAR or not again:
            logger.info("Not understandable; asking for it to be repeated.")
            return REPEAT_REQUEST
        if _overlap(draft, again) >= self.AGREEMENT:
            return draft
        if len(again) > self.MAX_LENGTH_RATIO * len(_key(malay)) + 6:
            # far more Thai than was said ("Encik" -> a whole question):
            # the LLM filled in words nobody spoke
            logger.warning(f"LLM re-translation says more than was heard: {again!r}")
            return draft
        logger.info(f"Re-translated for the hospital context: {again}")
        return again

    def _nllb(self, malay: str) -> str:
        return self.nllb.translate(malay).strip().lstrip("-–—• ").strip()
