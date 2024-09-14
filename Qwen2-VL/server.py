import logging
import os
import sys
from dataclasses import dataclass, field
from threading import Thread
from typing import (
    Optional,
    Any,
    Tuple,
    Dict,
)

import litserve as ls
import torch
from litserve.specs.openai import ChatCompletionRequest
from qwen_vl_utils import process_vision_info
from transformers import (
    HfArgumentParser,
    Qwen2VLForConditionalGeneration,
    TextIteratorStreamer,
    AutoProcessor,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune or infer.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )


@dataclass
class InferArguments:
    host: str = field(
        default="0.0.0.0",
        metadata={
            "help": "API server host."
        },
    )
    port: int = field(
        default=8000,
        metadata={
            "help": "API server port."
        },
    )
    num_api_servers: Optional[int] = field(
        default=None,
        metadata={
            "help": "Launch servers on multiple process or thread."
        },
    )
    accelerator: str = field(
        default="auto",
        metadata={
            "help": "The type of hardware to use (cpu, GPUs, mps)."
        },
    )
    workers_per_device: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of workers (processes) per device."
        },
    )
    timeout: Optional[float] = field(
        default=30,
        metadata={
            "help": "The timeout (in seconds) for a request."
        },
    )


_INFER_ARGS = [ModelArguments, InferArguments]
_INFER_CLS = Tuple[ModelArguments, InferArguments]


def _parse_args(parser: "HfArgumentParser", args: Optional[Dict[str, Any]] = None) -> Tuple[Any]:
    if args is not None:
        return parser.parse_dict(args)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if unknown_args:
        print(parser.format_help())
        print("Got unknown args, potentially deprecated arguments: {}".format(unknown_args))
        raise ValueError("Some specified arguments are not used by the HfArgumentParser: {}".format(unknown_args))

    return *parsed_args,


def _parse_infer_args(args: Optional[Dict[str, Any]] = None) -> _INFER_CLS:
    parser = HfArgumentParser(_INFER_ARGS)
    return _parse_args(parser, args)


def get_infer_args(args: Optional[Dict[str, Any]] = None) -> _INFER_CLS:
    model_args, infer_args = _parse_infer_args(args)
    logger.info(f"Model arguments: {model_args}")
    logger.info(f"API server arguments: {infer_args}")
    return model_args, infer_args


model_args, infer_args = get_infer_args()


class Qwen2VLAPI(ls.LitAPI):
    def setup(self, device):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            # _attn_implementation="flash_attention_2",
            device_map="auto",
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
        self.streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def decode_request(self, request: ChatCompletionRequest, context: dict):
        context["generation_args"] = {
            "max_new_tokens": request.max_tokens if request.max_tokens else 2048,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }

        messages = [message.model_dump(exclude_none=True) for message in request.messages]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        return inputs

    def predict(self, model_inputs, context: dict):
        generation_kwargs = dict(
            model_inputs,
            streamer=self.streamer,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **context["generation_args"],
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for text in self.streamer:
            yield text


if __name__ == "__main__":
    api = Qwen2VLAPI()
    server = ls.LitServer(
        api,
        accelerator=infer_args.accelerator,
        workers_per_device=infer_args.workers_per_device,
        timeout=infer_args.timeout,
        spec=ls.OpenAISpec(),
    )
    server.run(
        port=infer_args.port,
        num_api_servers=infer_args.num_api_servers,
        generate_client_file=False,
    )
