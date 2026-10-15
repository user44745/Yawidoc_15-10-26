# src/open_llm_vtuber/translate/pattani_translator.py
import json
import re
from pathlib import Path

from loguru import logger
from openai import OpenAI

from .nllb_translator import NLLBTranslator
from .pattani_script import collapse_repeats, malay_to_pattani
from .pattani_script import thai_key as _normalize
from .thai_romanize import thai_to_latin
from .translate_interface import TranslateInterface

THAI_CHAR = re.compile(r"[฀-๿]")
LATIN_CHAR = re.compile(r"[A-Za-z]")
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
TONE_MARKS = re.compile("[\u0e48-\u0e4b]")
# A near match must not flip one of these.
THAI_NEGATIONS = ("ไม่", "อย่า", "ห้าม")
# letters from which a phrase is long enough to be matched with one wrong letter
LONG_PHRASE = 10
# An LLM answer this similar to the Thai source is the source repeated back,
# not a translation.
ECHO_SIMILARITY = 0.8

CHECK_SYSTEM_PROMPT = """\
You check a machine translation from Thai into Malay for a hospital interpreter.
You get the Thai source and a Malay draft.
- If the draft already says what the Thai says, reply with the draft exactly as given.
- If the draft is wrong or leaves something out, reply with a corrected translation
  in short, simple, everyday Standard Malay.
Reply with one line of Malay only (Latin letters). No Thai, no explanation, no quotes."""

WRITE_SYSTEM_PROMPT = """\
You write Pattani Malay (Kelantan-Pattani dialect, ภาษามลายูปัตตานี) for a hospital
interpreter in southern Thailand. Pattani Malay is spelled phonetically in Thai script.

You get a Central Thai sentence, its Standard Malay translation, and a first version
in Pattani Malay made by sound rules. Write the final Pattani sentence.

Rules:
- Start from the first version. Change it only where the reference pairs show that
  Pattani speakers use a different word, spelling or word order.
- Keep the meaning of the Thai sentence.
- Never answer with the Thai sentence itself. Thai words are allowed only for medical
  terms and names, the way the reference pairs use them.
- Reply with the Pattani sentence only: one line, Thai script, no Latin letters,
  no quotes, no explanation."""


def _bigrams(text: str) -> set[str]:
    text = _normalize(text)
    return {text[i : i + 2] for i in range(len(text) - 1)} or {text}


def _similarity(a: str, b: str) -> float:
    """Character-bigram overlap of two texts, 0.0 (nothing shared) to 1.0."""
    grams_a, grams_b = _bigrams(a), _bigrams(b)
    return 2 * len(grams_a & grams_b) / (len(grams_a) + len(grams_b))


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


def _first_line(answer: str | None) -> str:
    answer = THINK_BLOCK.sub("", answer or "")
    line = next((ln for ln in answer.splitlines() if ln.strip()), "")
    return line.strip().strip("\"'“”").strip()


class PattaniTranslator(TranslateInterface):
    """Central Thai -> Pattani Malay written in Thai script.

    NLLB cannot produce Pattani Malay, so translation runs in stages:

    1. Phrasebook: an input that matches a human-translated phrasebook entry
       returns that entry as is.
    2. NLLB translates Thai into Standard Malay (tha_Thai -> zsm_Latn).
    3. Sound rules (pattani_script.py) turn the Malay into Pattani Malay in
       Thai script. This version always exists and is the fallback.
    4. Optional LLM consult: an OpenAI-compatible LLM first corrects NLLB's
       Malay if it mistranslates the Thai, then writes the final Pattani
       sentence, guided by the most similar phrasebook entries. An answer that
       is not a usable translation (Thai source repeated back, Latin script,
       empty ...) is discarded in favour of the rule-based version.
    """

    def __init__(
        self,
        phrasebook_path: str = "pattani_dataset/phrasebook.jsonl",
        nllb_model: str = "facebook/nllb-200-distilled-600M",
        device: str = "cpu",
        llm_consult: bool = True,
        llm_base_url: str = "http://localhost:11434/v1",
        llm_model: str = "qwen3:8b",
        llm_api_key: str = "ollama",
        llm_temperature: float = 0.0,
        num_examples: int = 10,
        show_latin: bool = True,
        **kwargs,
    ):
        """
        :param phrasebook_path: JSONL built by build_phrasebook.py
        :param nllb_model: Hugging Face model name or local path of the NLLB model
        :param device: "cpu" or "cuda" for NLLB
        :param llm_consult: let the LLM write the final Pattani sentence
        :param llm_base_url: OpenAI-compatible endpoint (Ollama by default)
        :param llm_model: model name at that endpoint
        :param num_examples: phrasebook sentences shown to the LLM per request
        :param show_latin: also show the Pattani sentence in Latin letters
        """
        self.sentences: list[tuple[str, str]] = []
        self.terms: list[tuple[str, str]] = []
        self.exact: dict[str, str] = {}
        self.toneless: dict[str, str | None] = {}
        self._load_phrasebook(phrasebook_path)

        self.nllb = NLLBTranslator(
            model_name=nllb_model,
            src_lang="tha_Thai",
            tgt_lang="zsm_Latn",
            device=device,
        )

        self.llm_consult = llm_consult
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.num_examples = num_examples
        self.show_latin = show_latin
        self.client = (
            OpenAI(base_url=llm_base_url, api_key=llm_api_key, timeout=60)
            if llm_consult
            else None
        )

    def _load_phrasebook(self, phrasebook_path: str) -> None:
        path = Path(phrasebook_path)
        if not path.exists():
            logger.warning(
                f"Pattani phrasebook not found at '{phrasebook_path}'. "
                "Run build_phrasebook.py; translating without reference pairs."
            )
            return
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                thai, pattani = entry["thai"], entry["pattani"]
                # first translation wins when the same Thai appears twice
                key = _normalize(thai)
                self.exact.setdefault(key, pattani)
                # None once two different entries differ only in their tone marks
                toneless = TONE_MARKS.sub("", key)
                if self.toneless.setdefault(toneless, pattani) != self.exact[key]:
                    self.toneless[toneless] = None
                if entry.get("kind") == "term":
                    self.terms.append((thai, pattani))
                else:
                    self.sentences.append((thai, pattani))
        logger.info(
            f"Pattani phrasebook loaded: {len(self.sentences)} sentences, "
            f"{len(self.terms)} terms."
        )

    def _lookup(self, text: str) -> str | None:
        """The phrasebook entry for a sentence, allowing for what speech
        recognition typically gets wrong."""
        key = _normalize(text)
        if key in self.exact:
            return self.exact[key]
        # a wrong or missing tone mark: ไอ่ for ไอ
        known = self.toneless.get(TONE_MARKS.sub("", key))
        if known:
            return known
        # one wrong letter in a long phrase: สวัสดิภาค for สวัสดิภาพ
        if len(key) < LONG_PHRASE:
            return None
        for entry, pattani in self.exact.items():
            if (
                abs(len(entry) - len(key)) <= 1
                and _edit_distance(entry, key) <= 1
                and [entry.count(n) for n in THAI_NEGATIONS]
                == [key.count(n) for n in THAI_NEGATIONS]
                and re.findall(r"\d+", entry) == re.findall(r"\d+", key)
            ):
                return pattani
        return None

    def _similar_sentences(self, text: str) -> list[tuple[str, str]]:
        """Phrasebook sentences ranked by character-bigram overlap with text."""
        ranked = sorted(
            self.sentences, key=lambda pair: _similarity(text, pair[0]), reverse=True
        )
        return ranked[: self.num_examples]

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

    def _check_draft(self, text: str, draft: str) -> str:
        """Let the LLM correct NLLB's Malay draft; keep the draft if it can't."""
        try:
            checked = self._ask_llm(
                CHECK_SYSTEM_PROMPT, f"Thai source: {text}\nMalay draft: {draft}"
            )
        except Exception as e:
            logger.error(f"LLM draft check failed, keeping the NLLB draft: {e}")
            return draft
        if (
            not LATIN_CHAR.search(checked)
            or THAI_CHAR.search(checked)
            or len(checked) > 3 * len(draft) + 40
        ):
            logger.warning(f"LLM draft check rejected, keeping NLLB: {checked!r}")
            return draft
        if checked != draft:
            logger.info(f"LLM corrected the Malay draft: {checked}")
        return checked

    def _write_pattani(self, text: str, draft: str, rule_based: str) -> str:
        """Let the LLM write the final Pattani; fall back to the rule-based one."""
        lines = ["Reference pairs (Thai => Pattani):"]
        lines += [f"- {th} => {pt}" for th, pt in self._similar_sentences(text)]
        # the longest glossary terms found in the sentence; one- and two-letter
        # terms (ตา, ยา) turn up inside unrelated words
        terms = sorted(
            ((th, pt) for th, pt in self.terms if len(th) > 2 and th in text),
            key=lambda pair: len(pair[0]),
            reverse=True,
        )[: self.num_examples]
        if terms:
            lines.append("Glossary (Thai => Pattani):")
            lines += [f"- {th} => {pt}" for th, pt in terms]
        lines += [
            "",
            f"Thai sentence: {text}",
            f"Standard Malay: {draft}",
            f"First version (sound rules): {rule_based}",
            "Final Pattani (Thai script):",
        ]
        try:
            answer = self._ask_llm(WRITE_SYSTEM_PROMPT, "\n".join(lines))
        except Exception as e:
            logger.error(f"LLM consult failed, using the rule-based version: {e}")
            return rule_based

        if not THAI_CHAR.search(answer) or LATIN_CHAR.search(answer):
            reason = "not in Thai script"
        elif _similarity(answer, text) >= ECHO_SIMILARITY:
            reason = "repeats the Thai source"
        elif len(answer) > 3 * max(len(rule_based), len(text)) + 20:
            reason = "far too long"
        else:
            return answer
        logger.warning(
            f"LLM answer rejected ({reason}), using the rule-based version: {answer!r}"
        )
        return rule_based

    def display_text(self, translation: str) -> str:
        """Pattani in Thai script, with the same sentence in Latin letters below."""
        latin = thai_to_latin(translation) if self.show_latin else ""
        if not latin or latin == translation:
            return translation
        return f"{translation} \n{latin}"

    def translate(self, text: str) -> str:
        """
        Translate Central Thai to Pattani Malay in Thai script.

        Raises RuntimeError when NLLB produces no Malay, so the caller reports
        an error instead of passing the Thai input off as a translation.
        """
        if not text or not text.strip():
            return text
        text = collapse_repeats(text)

        known = self._lookup(text)
        if known:
            logger.info("Pattani translation: phrasebook match.")
            return known

        draft = self.nllb.translate(text).strip().lstrip("-–—• ").strip()
        # NLLBTranslator hands the input back unchanged when it fails
        if not LATIN_CHAR.search(draft) or THAI_CHAR.search(draft):
            raise RuntimeError(f"NLLB did not return a Malay translation: {draft!r}")
        logger.info(f"NLLB Standard Malay draft: {draft}")

        if self.llm_consult:
            draft = self._check_draft(text, draft)
        rule_based = malay_to_pattani(draft)
        logger.info(f"Rule-based Pattani: {rule_based}")
        if not self.llm_consult:
            return rule_based
        return self._write_pattani(text, draft, rule_based)
