"""SpoofCeleb의 분할된 tar.gz(29파트, 총 134GB)를 처음부터 순차 스트리밍하면서
필요한 개수만큼만 멤버를 추출하고 중단하는 유틸리티.

tar+gzip은 순차 포맷이라 전체를 내려받지 않아도, 앞부분 스트림만 디코딩하면
초반에 등장하는 파일들을 꺼낼 수 있다. HTTP Range 요청 없이 파트 파일들을
순서대로 이어 붙여 하나의 연속된 바이트 스트림처럼 읽는다.
"""

import io
import os
import sys
import tarfile

import requests
from huggingface_hub import hf_hub_url

REPO_ID = "jungjee/spoofceleb"
REPO_TYPE = "dataset"

# HF에 올라온 파트 파일 이름 순서 (aa, ab, ..., az, ba)
PART_NAMES = [f"spoofceleb.tar.gz{c1}{c2}" for c1 in "a" for c2 in "abcdefghijklmnopqrstuvwxyz"] + [
    "spoofceleb.tar.gzba"
]


class MultiPartHTTPStream(io.RawIOBase):
    """여러 URL의 바이트를 순서대로 이어서 읽어주는 파일 유사 객체."""

    def __init__(self, urls, headers, chunk_size=1 << 20):
        self.urls = list(urls)
        self.headers = headers
        self.chunk_size = chunk_size
        self._part_index = 0
        self._iter = None
        self._buffer = b""
        self.total_read = 0

    def _open_next_part(self):
        if self._part_index >= len(self.urls):
            return False
        url = self.urls[self._part_index]
        self._part_index += 1
        resp = requests.get(url, headers=self.headers, stream=True, timeout=60)
        resp.raise_for_status()
        self._iter = resp.iter_content(chunk_size=self.chunk_size)
        return True

    def readinto(self, b):
        while len(self._buffer) == 0:
            if self._iter is None:
                if not self._open_next_part():
                    return 0
            try:
                chunk = next(self._iter)
                if chunk:
                    self._buffer = chunk
            except StopIteration:
                self._iter = None
                if not self._open_next_part():
                    return 0
        n = min(len(b), len(self._buffer))
        b[:n] = self._buffer[:n]
        self._buffer = self._buffer[n:]
        self.total_read += n
        return n

    def readable(self):
        return True


def open_stream(token, max_parts=None):
    part_urls = [
        hf_hub_url(repo_id=REPO_ID, filename=name, repo_type=REPO_TYPE)
        for name in (PART_NAMES if max_parts is None else PART_NAMES[:max_parts])
    ]
    headers = {"Authorization": f"Bearer {token}"}
    return MultiPartHTTPStream(part_urls, headers)


def inspect(token, limit=500, max_parts=3):
    stream = open_stream(token, max_parts=max_parts)
    buffered = io.BufferedReader(stream, buffer_size=1 << 20)
    tf = tarfile.open(fileobj=buffered, mode="r|gz")
    count = 0
    dirs_seen = set()
    for member in tf:
        count += 1
        top_parts = member.name.split("/")[:3]
        dirs_seen.add("/".join(top_parts))
        if count <= 40:
            print(member.name, member.size)
        if count >= limit:
            break
    print("---")
    print(f"inspected {count} members, downloaded ~{stream.total_read / 1e6:.1f} MB")
    print("distinct top-level path prefixes (up to 3 segments), sample:")
    for d in sorted(dirs_seen)[:60]:
        print(" ", d)


if __name__ == "__main__":
    token = os.environ["HF_TOKEN"]
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    max_parts = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    inspect(token, limit=limit, max_parts=max_parts)
