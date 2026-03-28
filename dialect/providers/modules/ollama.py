# Copyright 2026 Sunur Efe Vural
# SPDX-License-Identifier: GPL-3.0-or-later

import json

from dialect.define import LANGUAGES as ALL_LANGUAGES
from dialect.providers.base import (
    ProviderCapability,
    ProviderFeature,
    Translation,
    TranslationRequest,
)
from dialect.providers.errors import RequestError, UnexpectedError
from dialect.providers.soup import SoupProvider

AUTO_PROMPT = (
    "You are a professional {dest_lang} ({dest_code}) translator. Your goal is to"
    " accurately convey the meaning and nuances of the original text while adhering"
    " to {dest_lang} grammar, vocabulary, and cultural sensitivities.\n"
    "Produce only the {dest_lang} translation, without any additional explanations or"
    " commentary. Please translate the following text into {dest_lang}:\n\n\n"
)

TRANSLATE_PROMPT = (
    "You are a professional {src_lang} ({src_code}) to {dest_lang} ({dest_code})"
    " translator. Your goal is to accurately convey the meaning and nuances of the"
    " original {src_lang} text while adhering to {dest_lang} grammar, vocabulary,"
    " and cultural sensitivities.\n"
    "Produce only the {dest_lang} translation, without any additional explanations or"
    " commentary. Please translate the following {src_lang} text into {dest_lang}:\n\n\n"
)

SUPPORTED_LANGS = [
    "aa", "ab", "af", "ak", "am", "an", "ar", "as", "az", "ba", "be", "bg", "bm", "bn", "bo",
    "br", "bs", "ca", "ce", "co", "cs", "cv", "cy", "da", "de", "dv", "dz", "ee", "el", "en",
    "es", "et", "eu", "fa", "ff", "fi", "fo", "fr", "fy", "ga", "gd", "gl", "gn", "gu", "gv",
    "ha", "he", "hi", "hr", "ht", "hu", "hy", "ia", "id", "ie", "ig", "ii", "ik", "io", "is",
    "it", "iu", "ja", "jv", "ka", "ki", "kk", "kl", "km", "kn", "ko", "ks", "ku", "kw", "ky",
    "la", "lb", "lg", "ln", "lo", "lt", "lu", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms",
    "mt", "my", "nb", "nd", "ne", "nl", "nn", "no", "nr", "nv", "ny", "oc", "om", "or", "os",
    "pa", "pl", "ps", "pt", "qu", "rm", "rn", "ro", "ru", "rw", "sa", "sc", "sd", "se", "sg",
    "si", "sk", "sl", "sn", "so", "sq", "sr", "ss", "st", "su", "sv", "sw", "ta", "te", "tg",
    "th", "ti", "tk", "tl", "tn", "to", "tr", "ts", "tt", "ug", "uk", "ur", "uz", "ve", "vi",
    "vo", "wa", "wo", "xh", "yi", "yo", "za", "zh", "zu",
]

class Provider(SoupProvider):
    name = "ollama"
    prettyname = "Ollama"

    capabilities = ProviderCapability.TRANSLATION
    features = ProviderFeature.INSTANCES | ProviderFeature.ENGINES | ProviderFeature.DETECTION | ProviderFeature.API_KEY

    defaults = {
        "instance_url": "localhost:11434/api",
        "engine_name": "translategemma",
        "api_key": "",
        "src_langs": [],
        "dest_langs": ["en", "zh", "hi", "es", "ar"],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def generate_url(self):
        return self.format_url(self.instance_url, "/generate")

    async def _fetch_model_names(self, url) -> list[str]:
        response = await self.get(self.format_url(url, "/tags"), check_common=False)
        return [m["name"] for m in response.get("models", [])]

    async def validate_instance(self, url):
        try:
            return bool(await self._fetch_model_names(url))
        except Exception:
            return False

    async def validate_engine(self, name):
        try:
            available = await self._fetch_model_names(self.instance_url)
            engine = name if ":" in name else name + ":latest"
            return engine in available
        except Exception:
            return False

    async def init_trans(self):
        available = await self._fetch_model_names(self.instance_url)
        if not available:
            raise RequestError("Ollama instance not reachable or has no models")
        engine = self.engine if ":" in self.engine else self.engine + ":latest"
        if engine not in available:
            raise RequestError(f'Model "{self.engine}" is not available on this Ollama instance')
        for code in SUPPORTED_LANGS:
            self.add_lang(code, ALL_LANGUAGES.get(code))

    def _build_prompt(self, request: TranslationRequest) -> str:
        dest_lang = self._languages_names.get(request.dest, request.dest)

        if request.src == "auto":
            return AUTO_PROMPT.format(dest_lang=dest_lang, dest_code=request.dest) + request.text
        else:
            src_lang = self._languages_names.get(request.src, request.src)
            prompt = TRANSLATE_PROMPT.format(
                src_lang=src_lang,
                src_code=request.src,
                dest_lang=dest_lang,
                dest_code=request.dest,
            )
            return prompt + request.text

    async def translate(self, request: TranslationRequest) -> Translation:
        prompt = self._build_prompt(request)

        data = {
            "model": self.engine,
            "prompt": prompt,
        }

        # Ollama returns newline-delimited JSON (streaming)
        raw = await self.post(self.generate_url, data, return_json=False)

        try:
            parts = []
            for line in raw.splitlines():
                if not line:
                    continue
                chunk = json.loads(line)
                parts.append(chunk.get("response", ""))
            translated = "".join(parts)
            return Translation(translated, request)
        except Exception as exc:
            raise UnexpectedError from exc

    def check_known_errors(self, status, data):
        pass
