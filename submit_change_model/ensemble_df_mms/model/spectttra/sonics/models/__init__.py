from sonics.models.model import AudioClassifier
from sonics.models.spectttra import SpecTTTra
# [벤더링 수정] ViT는 이 오프라인 패키지에서 쓰지 않고, timm.layers.PatchEmbed
# 의존성(→ torchvision) 때문에 패키지 최상단에서 미리 import하면 timm이 없을 때
# 전체 로딩이 실패한다. model.py의 get_encoder()에서 실제로 ViT 분기를 탈 때만
# 지연 import하도록 옮겼다 (model.py, _timm_compat.py 주석 참고).
