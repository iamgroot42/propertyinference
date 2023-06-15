from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizerFast
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration

from distribution_inference.models.core import BaseModel


class WhisperASR(BaseModel):
    def __init__(self, name: str):
        super().__init__(is_asr_model=True, is_hf_model=True)
        self.name = name
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(name)
        self.tokenizer_fast = WhisperTokenizerFast.from_pretrained(name)
        self.tokenizer = WhisperTokenizer.from_pretrained(name)
        self.processor = WhisperProcessor.from_pretrained(name)
        self.model = WhisperForConditionalGeneration.from_pretrained(name)

        # make sure model uses 50257 as BOS
        # bos = self.tokenizer("<|startoftranscript|>").input_ids[0]
        # self.model.config.decoder_start_token_id = bos

        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        self.model.config.use_cache = False

    def save(self, path, **kwargs):
        self.model.save_pretrained(path, **kwargs)
    
    def load(self, path, **kwargs):
        self.model = self.model.from_pretrained(path, **kwargs)

    def forward(self, **kwargs):
        """
            Simply forward to self.model
        """
        return self.model(**kwargs)


class WhisperTiny(WhisperASR):
    def __init__(self):
        super().__init__("openai/whisper-tiny.en")


class WhisperBase(WhisperASR):
    def __init__(self):
        super().__init__("openai/whisper-base.en")


class WhisperSmall(WhisperASR):
    def __init__(self):
        super().__init__("openai/whisper-small.en")
