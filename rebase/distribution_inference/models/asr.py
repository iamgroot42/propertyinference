from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration

from distribution_inference.models.core import BaseModel

import torch as ch
from datasets import Audio


class WhisperASR(BaseModel):
    def __init__(self, name: str):
        super().__init__(is_asr_model=True, is_hf_model=True)
        self.name = name
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(name)
        self.tokenizer = WhisperTokenizer.from_pretrained(name, language="English", task="transcribe")
        self.processor = WhisperProcessor.from_pretrained(name, language="English", task="transcribe")
        self.model = WhisperForConditionalGeneration.from_pretrained(name)

        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="en", task="transcribe")
        self.model.config.suppress_tokens = []
    
    def save(self, path, **kwargs):
        self.model.save_pretrained(path, **kwargs)
    
    def load(self, path, **kwargs):
        self.model = self.model.from_pretrained(path, **kwargs)

    def forward(self, **kwargs):
        """
            Simply forward to self.model
        """
        return self.model(**kwargs)

    # def process_data(self, dataset, n_proc: int):
    #     # Set to 16kHz
    #     sampling_rate = 16000
    #     return whisper_asr_process_data(dataset, self.feature_extractor, self.tokenizer, sampling_rate, n_proc)


class WhisperTiny(WhisperASR):
    def __init__(self):
        super().__init__("openai/whisper-tiny")


class WhisperSmall(WhisperASR):
    def __init__(self):
        super().__init__("openai/whisper-small")


def whisper_asr_process_data(dataset, feature_extractor, tokenizer,
                             sampling_rate: int = 16000,
                             n_proc: int = 4):
    dataset_ = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    def prepare_dataset(batch):
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=sampling_rate).input_features[0]

        # encode target text to label ids 
        batch["labels"] = tokenizer(batch["text"]).input_ids
        return batch

    dataset_ = dataset_.map(prepare_dataset, num_proc=n_proc)
    return dataset_
