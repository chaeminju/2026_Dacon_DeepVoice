"""[벤더링 추가] timm.layers에서 가져오던 Mlp/DropPath/use_fused_attn 세 개만
timm 패키지 자체를 설치하지 않고도 쓸 수 있도록 로컬로 복사해온 것이다.

이유: timm은 setup.py상 torchvision을 필수 의존성으로 선언하고(실제로
timm.layers.norm_act가 torchvision.ops.misc를 최상단에서 import한다), pip로
timm을 설치하면 torchvision이 함께 설치된다. 이때 torchvision이 요구하는
특정 torch 버전에 맞추려고 pip 의존성 해석기가 채점 서버에 이미 CUDA용으로
빌드되어 있던 torch를 다른 버전으로 교체해버릴 수 있고, 그 경우 기존
torchaudio 네이티브 라이브러리(libtorchaudio.so)와 torch ABI가 어긋나
`OSError: Could not load this library: .../libtorchaudio.so`로 채점이
아예 실패한다(실제 재현됨). 이 프로젝트는 SpecTTTra 모델(sonics/models/
spectttra.py)만 쓰고 ViT/timm 인코더 분기는 쓰지 않으므로, transformer.py가
필요로 하는 세 개 함수/클래스만 아래처럼 그대로 복사해 timm 의존성 자체를
없앴다. 출처: timm 1.0.9, timm/layers/mlp.py, timm/layers/drop.py,
timm/layers/config.py (Apache-2.0 라이선스) — 동작은 원본과 동일하고,
state_dict 파라미터 이름(fc1/act/drop1/norm/fc2/drop2)도 그대로 유지해
체크포인트 로딩 호환성을 지킨다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_2tuple(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks
    (timm.layers.Mlp과 동일 구조)."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (timm.layers.drop.drop_path과 동일)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (timm.layers.DropPath과 동일).
    파라미터가 없는 모듈이라 state_dict 호환성에는 영향이 없다."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


def use_fused_attn() -> bool:
    """torch 2.x의 F.scaled_dot_product_attention(수학적으로 수동 softmax
    어텐션과 동일한 결과) 사용 가능 여부만 확인한다. timm.layers.use_fused_attn의
    ONNX-export/experimental 플래그 처리는 이 오프라인 추론 경로에서 쓰지
    않으므로 단순화했다."""
    return hasattr(F, "scaled_dot_product_attention")
