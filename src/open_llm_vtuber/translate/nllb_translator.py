# src/open_llm_vtuber/translate/nllb_translator.py
from loguru import logger
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .translate_interface import TranslateInterface


class NLLBTranslator(TranslateInterface):
    """Local NLLB (No Language Left Behind) translator.

    Language codes follow the FLORES-200 convention, e.g.
    src_lang="zsm_Latn" (Standard Malay, Latin script — the script
    Whisper transcribes Malay in) and tgt_lang="tha_Thai" (Thai).
    """

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        src_lang: str = "zsm_Latn",
        tgt_lang: str = "tha_Thai",
        device: str = "cpu",
        **kwargs,
    ):
        """
        Load the NLLB model and tokenizer.

        :param model_name: Hugging Face model name or local path
            (e.g. "facebook/nllb-200-distilled-600M")
        :param src_lang: FLORES-200 source language code (e.g. "zsm_Latn")
        :param tgt_lang: FLORES-200 target language code (e.g. "tha_Thai")
        :param device: "cpu" or "cuda"
        """
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        logger.info(
            f"Loading NLLB translator '{model_name}' "
            f"({src_lang} -> {tgt_lang}) on {device}..."
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device

        self.tgt_lang_id = self._resolve_lang_id(tgt_lang)
        logger.info("NLLB translator loaded.")

    def _resolve_lang_id(self, lang_code: str) -> int:
        """Map a FLORES-200 language code to its token id."""
        # newer transformers versions may drop `lang_code_to_id` on fast tokenizers
        lang_code_to_id = getattr(self.tokenizer, "lang_code_to_id", None)
        if isinstance(lang_code_to_id, dict) and lang_code in lang_code_to_id:
            token_id = lang_code_to_id[lang_code]
        else:
            token_id = self.tokenizer.convert_tokens_to_ids(lang_code)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            raise ValueError(
                f"Invalid NLLB FLORES-200 language code: '{lang_code}'. "
                "Expected codes like 'zsm_Latn' or 'tha_Thai'."
            )
        return token_id

    def translate(self, text: str) -> str:
        """
        Translate text from src_lang to tgt_lang.

        Returns the original text unchanged when the input is empty
        or when translation fails, so a model error never breaks the
        conversation pipeline.
        """
        if not text or not text.strip():
            return text
        try:
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tgt_lang_id,
                    max_length=512,
                    # unfamiliar input (dialect words) can make the model
                    # repeat one word until max_length: "แม่แม่แม่แม่..."
                    no_repeat_ngram_size=4,
                )
            translated = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )[0]
            return translated.strip()
        except Exception as e:
            logger.error(f"NLLB translation failed, returning original text: {e}")
            return text
