import logging
import os
import secrets
import sys
import time
from dataclasses import dataclass, field
from typing import (
    Optional,
    Any,
    Tuple,
    Dict,
    List,
    Literal,
)

import litserve as ls
import torch
from pydantic import BaseModel, Field
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    HfArgumentParser,
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
    devices: str = field(
        default="auto",
        metadata={
            "help": "The number of (CPUs, GPUs) to use for the server."
        },
    )
    workers_per_device: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of workers (processes) per device."
        },
    )
    max_batch_size: Optional[int] = field(
        default=1,
        metadata={
            "help": "The max number of requests to batch together."
        },
    )
    timeout: Optional[float] = field(
        default=30,
        metadata={
            "help": "The timeout (in seconds) for a request."
        },
    )
    batch_timeout: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "The timeout (in ms) until the server stops waiting to batch inputs."
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
config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
if "UIE" in config.architectures[0]:
    infer_args.max_batch_size = 1


class IECreateParams(BaseModel):
    text: str
    ie_schema: Optional[Any] = None


class CLSResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cls-{secrets.token_hex(12)}")
    object: Literal["text-classification"] = "text-classification"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    labels: List[Any]


class NERResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"ner-{secrets.token_hex(12)}")
    object: Literal["named-entity-recognition"] = "named-entity-recognition"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    labels: List[Dict[str, Any]]


class RELResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"rel-{secrets.token_hex(12)}")
    object: Literal["relation-extraction"] = "relation-extraction"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    labels: List[Dict[str, Any]]


class EVENTResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"event-{secrets.token_hex(12)}")
    object: Literal["event-extraction"] = "event-extraction"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    labels: List[Dict[str, Any]]


class UIEResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"uie-{secrets.token_hex(12)}")
    object: Literal["uie"] = "uie"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    labels: List[Dict[str, Any]]


RESPONSE_MAP = {
    "classification": CLSResponse,
    "ner": NERResponse,
    "relextraction": RELResponse,
    "eventextraction": EVENTResponse,
    "uiemodel": UIEResponse,
}


def get_response_model(architecture: str):
    for k, v in RESPONSE_MAP.items():
        if architecture.endswith(k):
            return v


class FastIEAPI(ls.LitAPI):
    def setup(self, device):
        """Load the tokenizer and model, and move the model to the specified device."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()

        self.architecture = config.architectures[0]
        print(f"Loading Model Architecture: {self.architecture}")
        self.response_model = get_response_model(self.architecture.lower())

    def decode_request(self, request: IECreateParams, **kwargs):
        """Convert the request payload to your model input."""
        return {"text": request.text, "schema": request.ie_schema}

    @torch.inference_mode()
    def predict(self, x, **kwargs):
        """Run the model on the input and return or yield the output."""
        if isinstance(x, list):
            assert infer_args.max_batch_size > 1, "Unexpected batch request recieved!"
            texts = [d["text"] for d in x]
            schema = None
        else:
            texts, schema = x["text"], x["schema"]

        kwargs = {"texts": texts}
        if "UIE" in self.architecture:
            kwargs["schema"] = schema

        return self.model.predict(self.tokenizer, **kwargs)

    def encode_response(self, output, **kwargs):
        """Convert the model output to a response payload."""
        return self.response_model(model=self.architecture, labels=output)


if __name__ == "__main__":
    api = FastIEAPI()
    server = ls.LitServer(
        api,
        accelerator=infer_args.accelerator,
        devices=infer_args.devices,
        max_batch_size=infer_args.max_batch_size,
        workers_per_device=infer_args.workers_per_device,
        api_path="/v1/ie",
        timeout=infer_args.timeout,
        batch_timeout=infer_args.batch_timeout,
    )
    server.run(
        port=infer_args.port,
        num_api_servers=infer_args.num_api_servers,
        generate_client_file=False,
    )
