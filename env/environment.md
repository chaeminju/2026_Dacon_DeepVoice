# 환경 세팅 기록

conda env 이름: `deepvoice` (python 3.10, /opt/conda 하위)

## 재현
```
conda create -n deepvoice python=3.10 -y
conda activate deepvoice
pip install -r requirements-freeze.txt
```

## 설치 중 발견한 이슈 및 조치
- `pyworld`가 내부적으로 `pkg_resources`를 import하는데, 최신 setuptools(>=81)는
  pkg_resources를 제거했음 → `pip install "setuptools<81"`로 고정.
- 최신 `demucs`(PyPI 4.1.0, adefossez 유지보수판)는 `demucs.separate.load_track`을
  제거하고 `Separator` 클래스로 API를 바꿈. baseline_submit/script.py는 구 API를
  기대하므로 `demucs==4.0.1`로 고정해야 함.
- ffmpeg에는 AMR-NB 인코더가 없음(디코더만 있음, 특허 이슈로 기본 빌드 제외).
  코덱 왕복 증강은 `gsm`(libgsm) 인코더를 사용하기로 함(인코드/디코드 모두 지원 확인됨).

## 검증
`baseline_submit/script.py`를 로컬 샘플 3개(`data/test/`)로 GPU에서 실행해 정상적으로
5개 확률 컬럼이 나오는 것을 확인함 (3개 파일이 MD5 동일한 동일 파일이라 출력도 동일 —
정상).
