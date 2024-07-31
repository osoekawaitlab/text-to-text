import os
from typing import Any, List, Optional, Union

from torch import Tensor

class PreTrainedTokenizerBase:
    def encode(self, text: str, return_tensors: str) -> Tensor: ...
    def decode(self, ids: List[int], skip_special_tokens: bool) -> str: ...

class PreTrainedModel:
    def generate(
        self, input_ids: Tensor, max_new_tokens: int, do_sample: bool, top_p: float, top_k: int
    ) -> List[List[int]]: ...
    @property
    def device(self) -> str: ...

class AutoTokenizer:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike[str]],
        *init_inputs: Any,
        cache_dir: Optional[Union[str, os.PathLike[str]]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs: Any,
    ) -> PreTrainedTokenizerBase: ...

class AutoModelForCausalLM:
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike[str]], *model_args: Any, **kwargs: Any
    ) -> PreTrainedModel: ...

class Pipeline:
    def __call__(
        self,
        text: str | list[dict[str, str]],
        max_new_tokens: Optional[int],
        return_full_text: Optional[bool],
        do_sample: Optional[bool],
    ) -> List[dict[str, str]]: ...

def pipeline(
    task: str, model: Optional[PreTrainedModel] = None, tokenizer: Optional[PreTrainedTokenizerBase] = None
) -> Pipeline: ...
