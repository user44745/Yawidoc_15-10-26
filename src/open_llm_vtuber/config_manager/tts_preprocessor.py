# config_manager/translate.py
from typing import Literal, Optional, Dict, ClassVar
from pydantic import ValidationInfo, Field, model_validator
from .i18n import I18nMixin, Description

# --- Sub-models for specific Translator providers ---


class DeepLXConfig(I18nMixin):
    """Configuration for DeepLX translation service."""

    deeplx_target_lang: str = Field(..., alias="deeplx_target_lang")
    deeplx_api_endpoint: str = Field(..., alias="deeplx_api_endpoint")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "deeplx_target_lang": Description(
            en="Target language code for DeepLX translation",
            zh="DeepLX 翻译的目标语言代码",
        ),
        "deeplx_api_endpoint": Description(
            en="API endpoint URL for DeepLX service", zh="DeepLX 服务的 API 端点 URL"
        ),
    }


class TencentConfig(I18nMixin):
    """Configuration for tencent translation service."""

    secret_id: str = Field(..., description="Tencent Secret ID")
    secret_key: str = Field(..., description="Tencent Secret Key")
    region: str = Field(..., description="Region for Tencent Service")
    source_lang: str = Field(
        ..., description="Source language code for tencent translation"
    )
    target_lang: str = Field(
        ..., description="Target language code for tencent translation"
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "secret_id": Description(en="Tencent Secret ID", zh="腾讯服务的Secret ID"),
        "secret_key": Description(en="Tencent Secret Key", zh="腾讯服务的Secret Key"),
        "region": Description(en="Region for Tencent Service", zh="腾讯服务使用的区域"),
        "source_lang": Description(
            en="Source language code for tencent translation", zh="腾讯翻译的源语言代码"
        ),
        "target_lang": Description(
            en="Target language code for tencent translation",
            zh="腾讯翻译的目标语言代码",
        ),
    }


class NLLBConfig(I18nMixin):
    """Configuration for NLLB (No Language Left Behind) local translation."""

    model: str = Field("facebook/nllb-200-distilled-600M", alias="model")
    src_lang: str = Field("zsm_Latn", alias="src_lang")
    tgt_lang: str = Field("tha_Thai", alias="tgt_lang")
    device: Literal["cpu", "cuda"] = Field("cpu", alias="device")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "model": Description(
            en="NLLB model name or path (Hugging Face)",
            zh="NLLB 模型名称或路径（Hugging Face）",
        ),
        "src_lang": Description(
            en="FLORES-200 source language code (e.g. zsm_Latn for Malay in Latin script)",
            zh="FLORES-200 源语言代码（如 zsm_Latn，拉丁字母马来语）",
        ),
        "tgt_lang": Description(
            en="FLORES-200 target language code (e.g. tha_Thai for Thai)",
            zh="FLORES-200 目标语言代码（如 tha_Thai，泰语）",
        ),
        "device": Description(
            en="Device to run the NLLB model on (cpu or cuda)",
            zh="运行 NLLB 模型的设备（cpu 或 cuda）",
        ),
    }


class PattaniConfig(I18nMixin):
    """Configuration for Thai -> Pattani Malay translation (NLLB + rules + LLM)."""

    phrasebook_path: str = Field(
        "pattani_dataset/phrasebook.jsonl", alias="phrasebook_path"
    )
    nllb_model: str = Field("facebook/nllb-200-distilled-600M", alias="nllb_model")
    device: Literal["cpu", "cuda"] = Field("cpu", alias="device")
    llm_consult: bool = Field(True, alias="llm_consult")
    llm_base_url: str = Field("http://localhost:11434/v1", alias="llm_base_url")
    llm_model: str = Field("qwen3:8b", alias="llm_model")
    llm_api_key: str = Field("ollama", alias="llm_api_key")
    llm_temperature: float = Field(0.0, alias="llm_temperature")
    num_examples: int = Field(10, alias="num_examples")
    show_latin: bool = Field(True, alias="show_latin")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "phrasebook_path": Description(
            en="Path to the Thai -> Pattani phrasebook JSONL (build_phrasebook.py)",
            zh="泰语 -> 北大年马来语短语库 JSONL 的路径（build_phrasebook.py）",
        ),
        "nllb_model": Description(
            en="NLLB model name or path used for the Thai -> Standard Malay draft",
            zh="用于生成泰语 -> 标准马来语初稿的 NLLB 模型名称或路径",
        ),
        "device": Description(
            en="Device to run the NLLB model on (cpu or cuda)",
            zh="运行 NLLB 模型的设备（cpu 或 cuda）",
        ),
        "llm_consult": Description(
            en="Let an LLM check the NLLB draft and write the final Pattani sentence",
            zh="让 LLM 检查 NLLB 初稿并写出最终的北大年语句子",
        ),
        "llm_base_url": Description(
            en="OpenAI-compatible endpoint of the consult LLM (e.g. Ollama)",
            zh="用于改写的 LLM 的 OpenAI 兼容端点（如 Ollama）",
        ),
        "llm_model": Description(
            en="Model name of the consult LLM", zh="用于改写的 LLM 模型名称"
        ),
        "llm_api_key": Description(
            en="API key for the consult LLM endpoint", zh="LLM 端点的 API 密钥"
        ),
        "llm_temperature": Description(
            en="Sampling temperature of the consult LLM", zh="LLM 的采样温度"
        ),
        "num_examples": Description(
            en="Number of similar phrasebook sentences shown to the LLM",
            zh="提供给 LLM 的相似短语库句子数量",
        ),
        "show_latin": Description(
            en="Also show the Pattani sentence in Latin letters under the Thai script",
            zh="在泰文字母下方同时显示拉丁字母拼写的北大年语句子",
        ),
    }


class PattaniThaiConfig(I18nMixin):
    """Configuration for Pattani Malay -> Thai translation in a hospital setting."""

    phrases_path: str = Field(
        "pattani_dataset/medical_phrases.jsonl", alias="phrases_path"
    )
    nllb_model: str = Field("facebook/nllb-200-distilled-600M", alias="nllb_model")
    device: Literal["cpu", "cuda"] = Field("cpu", alias="device")
    llm_review: bool = Field(True, alias="llm_review")
    llm_base_url: str = Field("http://localhost:11434/v1", alias="llm_base_url")
    llm_model: str = Field("qwen3:8b", alias="llm_model")
    llm_api_key: str = Field("ollama", alias="llm_api_key")
    llm_temperature: float = Field(0.0, alias="llm_temperature")
    num_examples: int = Field(6, alias="num_examples")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "phrases_path": Description(
            en="Path to the hospital phrase table JSONL ({malay, thai} per line)",
            zh="医院常用语表 JSONL 的路径（每行 {malay, thai}）",
        ),
        "nllb_model": Description(
            en="NLLB model name or path used for the Malay -> Thai translation",
            zh="用于马来语 -> 泰语翻译的 NLLB 模型名称或路径",
        ),
        "device": Description(
            en="Device to run the NLLB model on (cpu or cuda)",
            zh="运行 NLLB 模型的设备（cpu 或 cuda）",
        ),
        "llm_review": Description(
            en="Let an LLM correct mishearings and re-translate sentences whose translation does not fit a hospital conversation",
            zh="让 LLM 纠正误听，并重新翻译不符合医院对话语境的句子",
        ),
        "llm_base_url": Description(
            en="OpenAI-compatible endpoint of the review LLM (e.g. Ollama)",
            zh="用于审校的 LLM 的 OpenAI 兼容端点（如 Ollama）",
        ),
        "llm_model": Description(
            en="Model name of the review LLM", zh="用于审校的 LLM 模型名称"
        ),
        "llm_api_key": Description(
            en="API key for the review LLM endpoint", zh="LLM 端点的 API 密钥"
        ),
        "llm_temperature": Description(
            en="Sampling temperature of the review LLM", zh="LLM 的采样温度"
        ),
        "num_examples": Description(
            en="Number of similar phrase-table entries shown to the LLM",
            zh="提供给 LLM 的相似常用语条目数量",
        ),
    }


# --- Main TranslatorConfig model ---


class TranslatorConfig(I18nMixin):
    """Configuration for translation services."""

    translate_audio: bool = Field(..., alias="translate_audio")
    translate_provider: Literal[
        "deeplx", "tencent", "nllb", "pattani", "pattani_thai"
    ] = Field(..., alias="translate_provider")
    deeplx: Optional[DeepLXConfig] = Field(None, alias="deeplx")
    tencent: Optional[TencentConfig] = Field(None, alias="tencent")
    nllb: Optional[NLLBConfig] = Field(None, alias="nllb")
    pattani: Optional[PattaniConfig] = Field(None, alias="pattani")
    pattani_thai: Optional[PattaniThaiConfig] = Field(None, alias="pattani_thai")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "translate_audio": Description(
            en="Enable audio translation (requires DeepLX deployment)",
            zh="启用音频翻译（需要部署 DeepLX）",
        ),
        "translate_provider": Description(
            en="Translation service provider to use", zh="要使用的翻译服务提供者"
        ),
        "deeplx": Description(
            en="Configuration for DeepLX translation service", zh="DeepLX 翻译服务配置"
        ),
        "tencent": Description(
            en="Configuration for TenCent translation service", zh="腾讯 翻译服务配置"
        ),
        "nllb": Description(
            en="Configuration for NLLB local translation", zh="NLLB 本地翻译配置"
        ),
        "pattani": Description(
            en="Configuration for Thai -> Pattani Malay translation",
            zh="泰语 -> 北大年马来语翻译配置",
        ),
        "pattani_thai": Description(
            en="Configuration for Pattani Malay -> Thai translation",
            zh="北大年马来语 -> 泰语翻译配置",
        ),
    }

    @model_validator(mode="after")
    def check_translator_config(cls, values: "TranslatorConfig", info: ValidationInfo):
        translate_audio = values.translate_audio
        translate_provider = values.translate_provider

        if translate_audio:
            if translate_provider == "deeplx" and values.deeplx is None:
                raise ValueError(
                    "DeepLX configuration must be provided when translate_audio is True and translate_provider is 'deeplx'"
                )
            elif translate_provider == "tencent" and values.tencent is None:
                raise ValueError(
                    "Tencent configuration must be provided when translate_audio is True and translate_provider is 'tencent'"
                )
            elif translate_provider == "nllb" and values.nllb is None:
                raise ValueError(
                    "NLLB configuration must be provided when translate_audio is True and translate_provider is 'nllb'"
                )
            elif translate_provider == "pattani" and values.pattani is None:
                raise ValueError(
                    "Pattani configuration must be provided when translate_audio is True and translate_provider is 'pattani'"
                )
            elif translate_provider == "pattani_thai" and values.pattani_thai is None:
                raise ValueError(
                    "Pattani-Thai configuration must be provided when translate_audio is True and translate_provider is 'pattani_thai'"
                )

        return values


class TTSPreprocessorConfig(I18nMixin):
    """Configuration for TTS preprocessor."""

    remove_special_char: bool = Field(..., alias="remove_special_char")
    ignore_brackets: bool = Field(default=True, alias="ignore_brackets")
    ignore_parentheses: bool = Field(default=True, alias="ignore_parentheses")
    ignore_asterisks: bool = Field(default=True, alias="ignore_asterisks")
    ignore_angle_brackets: bool = Field(default=True, alias="ignore_angle_brackets")
    translator_config: TranslatorConfig = Field(..., alias="translator_config")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "remove_special_char": Description(
            en="Remove special characters from the input text",
            zh="从输入文本中删除特殊字符",
        ),
        "translator_config": Description(
            en="Configuration for translation services", zh="翻译服务的配置"
        ),
    }
