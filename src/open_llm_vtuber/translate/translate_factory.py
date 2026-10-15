from .deeplx import DeepLXTranslate
from .tencent import TencentTranslate
from .nllb_translator import NLLBTranslator
from .pattani_thai_translator import PattaniThaiTranslator
from .pattani_translator import PattaniTranslator
from .translate_interface import TranslateInterface


class TranslateFactory:
    @staticmethod
    def get_translator(
        translate_provider: str, translate_provider_config: dict
    ) -> TranslateInterface:
        translate_provider = translate_provider.lower()
        if translate_provider == "deeplx":
            return DeepLXTranslate(
                api_endpoint=translate_provider_config.get("deeplx_api_endpoint"),
                target_lang=translate_provider_config.get("deeplx_target_lang"),
            )
        elif translate_provider == "tencent":
            return TencentTranslate(
                secret_id=translate_provider_config.get("secret_id"),
                secret_key=translate_provider_config.get("secret_key"),
                region=translate_provider_config.get("region"),
                source_lang=translate_provider_config.get("source_lang"),
                target_lang=translate_provider_config.get("target_lang"),
            )
        elif translate_provider == "nllb":
            # service_context passes the provider section's model_dump() directly,
            # so the keys are flat: model / src_lang / tgt_lang / device
            return NLLBTranslator(
                model_name=translate_provider_config.get(
                    "model", "facebook/nllb-200-distilled-600M"
                ),
                src_lang=translate_provider_config.get("src_lang", "zsm_Latn"),
                tgt_lang=translate_provider_config.get("tgt_lang", "tha_Thai"),
                device=translate_provider_config.get("device", "cpu"),
            )
        elif translate_provider == "pattani":
            # flat keys from PattaniConfig.model_dump()
            return PattaniTranslator(**translate_provider_config)
        elif translate_provider == "pattani_thai":
            # flat keys from PattaniThaiConfig.model_dump()
            return PattaniThaiTranslator(**translate_provider_config)
        else:
            raise ValueError(f"Unsupported translate provider: {translate_provider}")
