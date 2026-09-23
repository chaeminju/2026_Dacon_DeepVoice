"""AntiDeepfake XLS-R-2B (nii-yamagishilab) 로더.

원본 체크포인트는 fairseq 키 구조(m_ssl.model.*)라서 fairseq 없이 HuggingFace
Wav2Vec2Model에 키를 매핑해 로드한다. 백엔드는 시간축 평균 풀링 + FC(1920->2)이고
출력 순서는 [fake, real]이다.
"""

import json
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import Wav2Vec2Config, Wav2Vec2Model

FAKE_LABEL_INDEX = 0


def _map_key(key):
    """fairseq 키 -> (HF Wav2Vec2Model 키 | None). None이면 버린다(사전학습 전용 파라미터)."""
    if not key.startswith("m_ssl.model."):
        return None
    k = key[len("m_ssl.model."):]

    if k.startswith(("quantizer.", "project_q.", "final_proj.")) or k == "mask_emb":
        return None

    m = re.fullmatch(r"feature_extractor\.conv_layers\.(\d+)\.0\.(weight|bias)", k)
    if m:
        return f"feature_extractor.conv_layers.{m[1]}.conv.{m[2]}"
    m = re.fullmatch(r"feature_extractor\.conv_layers\.(\d+)\.2\.1\.(weight|bias)", k)
    if m:
        return f"feature_extractor.conv_layers.{m[1]}.layer_norm.{m[2]}"

    m = re.fullmatch(r"layer_norm\.(weight|bias)", k)
    if m:
        return f"feature_projection.layer_norm.{m[1]}"
    m = re.fullmatch(r"post_extract_proj\.(weight|bias)", k)
    if m:
        return f"feature_projection.projection.{m[1]}"

    m = re.fullmatch(r"encoder\.pos_conv\.0\.(bias|weight_g|weight_v)", k)
    if m:
        name = {
            "bias": "bias",
            "weight_g": "parametrizations.weight.original0",
            "weight_v": "parametrizations.weight.original1",
        }[m[1]]
        return f"encoder.pos_conv_embed.conv.{name}"

    if k.startswith("encoder.layer_norm."):
        return k

    m = re.fullmatch(r"encoder\.layers\.(\d+)\.(.+)", k)
    if m:
        i, rest = m[1], m[2]
        rest = rest.replace("self_attn_layer_norm", "layer_norm")
        rest = rest.replace("self_attn.", "attention.")
        rest = rest.replace("fc1.", "feed_forward.intermediate_dense.")
        rest = rest.replace("fc2.", "feed_forward.output_dense.")
        return f"encoder.layers.{i}.{rest}"

    raise KeyError(f"매핑되지 않은 키: {key}")


class AntiDeepfakeDetector(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        model_dir = Path(model_dir)
        with open(model_dir / "config.json") as f:
            config = Wav2Vec2Config.from_dict(json.load(f))
        config.layerdrop = 0.0
        config.apply_spec_augment = False

        state = load_file(str(model_dir / "model.safetensors"))
        ssl_state = {}
        for key, value in state.items():
            new_key = _map_key(key)
            if new_key is not None:
                ssl_state[new_key] = value

        with torch.device("meta"):
            self.ssl = Wav2Vec2Model(config)
        missing, unexpected = self.ssl.load_state_dict(
            ssl_state, strict=False, assign=True
        )
        # masked_spec_embed는 추론에서 쓰이지 않는다.
        missing = [m for m in missing if m != "masked_spec_embed"]
        if missing or unexpected:
            raise RuntimeError(f"가중치 로드 불일치 missing={missing} unexpected={unexpected}")
        if self.ssl.masked_spec_embed.is_meta:
            self.ssl.masked_spec_embed = nn.Parameter(
                torch.zeros(config.hidden_size), requires_grad=False
            )

        self.proj_fc = nn.Linear(config.hidden_size, 2)
        self.proj_fc.weight.data.copy_(state["proj_fc.weight"])
        self.proj_fc.bias.data.copy_(state["proj_fc.bias"])
        del state, ssl_state

    def forward(self, input_values):
        wav = input_values.reshape(1, -1)
        wav = F.layer_norm(wav, wav.shape)
        hidden = self.ssl(wav).last_hidden_state  # [1, T, D]
        return {"logits": self.proj_fc(hidden.mean(dim=1))}  # [fake, real]
