"""설정 로딩 + 경로 보장. config.yaml 단일 진실 공급원을 객체로 노출."""
from __future__ import annotations
import os
from pathlib import Path
import yaml

_ROOT = Path(__file__).resolve().parent.parent  # sovereign_ai/


class Config:
    def __init__(self, path: str | os.PathLike | None = None):
        cfg_path = Path(path) if path else _ROOT / "config.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.raw = yaml.safe_load(f)
        self.root = _ROOT

    # ── 경로 헬퍼 (전부 sovereign_ai/ 기준 절대경로) ──
    def _p(self, key: str) -> Path:
        return self.root / self.raw["paths"][key]

    @property
    def data_root(self) -> Path: return self._p("data_root")
    @property
    def pdf_dir(self) -> Path: return self._p("pdf_dir")
    @property
    def user_pdf_dir(self) -> Path: return self._p("user_pdf_dir")
    @property
    def text_dir(self) -> Path: return self._p("text_dir")
    @property
    def meta_dir(self) -> Path: return self._p("meta_dir")
    @property
    def index_dir(self) -> Path: return self._p("index_dir")

    def ensure_dirs(self):
        for key in ("pdf_dir", "user_pdf_dir", "text_dir", "meta_dir", "index_dir"):
            self._p(key).mkdir(parents=True, exist_ok=True)

    # ── 섹션 접근 ──
    @property
    def embedding(self) -> dict: return self.raw["embedding"]
    @property
    def llm(self) -> dict: return self.raw["llm"]
    @property
    def retrieval(self) -> dict: return self.raw["retrieval"]
    @property
    def domains(self) -> dict: return self.raw["domains"]
    @property
    def acquire(self) -> dict: return self.raw["acquire"]
    @property
    def datasets(self) -> list: return self.raw["datasets"]


def load(path: str | None = None) -> Config:
    return Config(path)
