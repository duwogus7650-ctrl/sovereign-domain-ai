"""소버린 도메인 AI — 오프라인 데스크톱 GUI (PyQt6).

터미널 대신 창에서 질의·응답한다. 무거운 엔진(RagEngine: FAISS 검색 + Ollama 생성)은
백그라운드 스레드에서 돌려, CPU에서 답변 1건이 수 분 걸려도 창이 멈추지 않게 한다.

완전 오프라인: 로컬 인덱스 + localhost Ollama 외에 네트워크를 쓰지 않는다.
(유일한 선택적 네트워크는 사용자가 출처 링크를 직접 클릭해 브라우저로 여는 것.)

실행:
  python -m sovereign.cli gui
"""
from __future__ import annotations
import html
import sys

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPlainTextEdit, QTextBrowser, QPushButton, QLabel,
)

from ..config import Config

# ── 엔지니어링 팔레트 (블루프린트 스틸 / 쿨 그레이 / 앰버 액센트) ──
STEEL = "#2f5f8f"
STEEL_DARK = "#1d3f63"
BG = "#f7f8fa"
PANEL = "#ffffff"
AMBER = "#d98a1f"
INK = "#1b2733"
MUTED = "#5b6b7b"
LINE = "#d4dbe2"

# 위젯(크롬) 스타일시트
APP_QSS = f"""
QMainWindow, QWidget {{ background: {BG}; }}
QLabel#status {{ background: {STEEL_DARK}; color: #cfe0f0; padding: 7px 12px; font-size: 12px; }}
QLabel#status[state="bad"] {{ color: #f0c089; }}
QTextBrowser#view {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 4px; padding: 4px 8px; }}
QPlainTextEdit#input {{ background: {PANEL}; border: 1px solid #c2ccd6; border-radius: 4px; padding: 6px; font-size: 14px; }}
QPlainTextEdit#input:focus {{ border: 1px solid {STEEL}; }}
QPushButton#send {{ background: {STEEL}; color: white; border: none; border-radius: 4px; padding: 8px 22px; font-weight: 600; }}
QPushButton#send:hover {{ background: {STEEL_DARK}; }}
QPushButton#send:disabled {{ background: #9bb0c4; }}
QLabel#busy {{ color: {AMBER}; font-weight: 600; padding: 2px; }}
"""

# 대화 본문(리치텍스트) 스타일 — QTextBrowser 기본 스타일시트
DOC_QSS = f"""
.q {{ color: {INK}; background: #eef2f6; padding: 8px 11px; }}
.qlabel {{ color: {STEEL}; font-weight: bold; }}
.a {{ color: {INK}; padding: 6px 11px; }}
.alabel {{ color: {STEEL_DARK}; font-weight: bold; }}
.err {{ color: #b04a2f; }}
.sources {{ color: {MUTED}; font-family: Consolas, monospace; font-size: 12px; padding: 4px 11px 12px 11px; }}
.tag {{ color: {STEEL}; }}
.score {{ color: {AMBER}; }}
a {{ color: {STEEL}; }}
"""

WELCOME = (
    "<div class='a'><span class='alabel'>소버린 도메인 AI</span><br>"
    "모터설계·제어·제어보드·AI고장진단·강화학습 분야의 오프라인 질의응답입니다.<br>"
    "아래에 질문을 입력하고 <b>전송</b>(또는 Enter)을 누르세요. "
    "이 PC는 GPU가 없어 답변 1건에 수 분이 걸릴 수 있습니다.</div>"
)


def _esc(s: str) -> str:
    return html.escape(s or "").replace("\n", "<br>")


def _sources_html(sources: list[dict]) -> str:
    if not sources:
        return ""
    rows = ["<div class='sources'><b>근거 출처</b><br>"]
    for n, h in enumerate(sources, 1):
        tag = "본문" if h.get("fulltext") else "초록"
        score = float(h.get("score", 0.0))
        title = html.escape((h.get("title", "") or "")[:110])
        url = h.get("abs_url", "") or ""
        line = (f"[{n}] <span class='tag'>{tag}</span> "
                f"<span class='score'>score={score:.3f}</span> {title}")
        if url:
            u = html.escape(url)
            line += f"<br>&nbsp;&nbsp;&nbsp;&nbsp;<a href='{u}'>{u}</a>"
        rows.append(line + "<br>")
    rows.append("</div>")
    return "".join(rows)


class InputBox(QPlainTextEdit):
    """Enter=전송, Shift+Enter=줄바꿈."""
    submitted = pyqtSignal()

    def keyPressEvent(self, e):  # noqa: N802 (Qt 시그니처)
        is_enter = e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
        shift = bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if is_enter and not shift:
            self.submitted.emit()
            return
        super().keyPressEvent(e)


class EngineWorker(QObject):
    """백그라운드 스레드에서 RagEngine을 소유·구동한다.

    init_engine(): 인덱스 로드(수 초). ask(): 검색+생성(수 분).
    UI 스레드와는 시그널로만 통신한다.
    """
    ready = pyqtSignal()
    init_failed = pyqtSignal(str)
    answered = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self._engine = None

    @pyqtSlot()
    def init_engine(self):
        try:
            from .rag import RagEngine
            self._engine = RagEngine(self.cfg)
            self.ready.emit()
        except Exception as e:  # 인덱스 없음 등 → UI에 표시
            self.init_failed.emit(str(e))

    @pyqtSlot(str)
    def ask(self, query: str):
        try:
            res = self._engine.ask(query)
            self.answered.emit(res)
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    _ask_requested = pyqtSignal(str)  # UI → worker (큐 연결)

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self._busy = False
        self._ready = False
        self._log: list[str] = [WELCOME]
        self._build_ui()
        self._start_worker()

    # ── UI ──
    def _build_ui(self):
        self.setWindowTitle("소버린 도메인 AI — 오프라인")
        self.resize(900, 680)
        self.setStyleSheet(APP_QSS)
        self.setFont(QFont("Malgun Gothic", 10))

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self.status = QLabel("엔진 준비 중…  ·  모델: " + str(self.cfg.llm.get("model", "?")))
        self.status.setObjectName("status")
        root.addWidget(self.status)

        self.view = QTextBrowser()
        self.view.setObjectName("view")
        self.view.setOpenExternalLinks(True)   # 출처 링크 클릭 → 브라우저
        self.view.document().setDefaultStyleSheet(DOC_QSS)
        root.addWidget(self.view, 1)

        self.busy = QLabel("")
        self.busy.setObjectName("busy")
        root.addWidget(self.busy)

        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        self.input = InputBox()
        self.input.setObjectName("input")
        self.input.setPlaceholderText("질문을 입력하세요 (Enter 전송 · Shift+Enter 줄바꿈)")
        self.input.setFixedHeight(72)
        self.input.submitted.connect(self._on_send)
        bottom.addWidget(self.input, 1)

        self.send_btn = QPushButton("전송")
        self.send_btn.setObjectName("send")
        self.send_btn.clicked.connect(self._on_send)
        bottom.addWidget(self.send_btn, 0, Qt.AlignmentFlag.AlignBottom)
        root.addLayout(bottom)

        self.setCentralWidget(central)
        self._render()
        self._set_enabled(False)  # 엔진 준비 전 입력 잠금

    def _start_worker(self):
        self._thread = QThread(self)
        self._worker = EngineWorker(self.cfg)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.init_engine)
        self._worker.ready.connect(self._on_ready)
        self._worker.init_failed.connect(self._on_init_failed)
        self._worker.answered.connect(self._on_answered)
        self._worker.failed.connect(self._on_failed)
        self._ask_requested.connect(self._worker.ask)
        self._thread.start()

    # ── 렌더링 ──
    def _render(self):
        self.view.setHtml("<body>" + "".join(self._log) + "</body>")
        sb = self.view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _append(self, block: str):
        self._log.append(block)
        self._render()

    def _set_enabled(self, on: bool):
        self.input.setEnabled(on)
        self.send_btn.setEnabled(on)

    def _set_busy(self, busy: bool):
        self._busy = busy
        self.busy.setText("⏳ 생성 중…  (CPU에서 수 분 걸릴 수 있습니다. 잠시만 기다려 주세요)" if busy else "")
        self._set_enabled(self._ready and not busy)
        if not busy:
            self.input.setFocus()

    # ── 슬롯 ──
    @pyqtSlot()
    def _on_ready(self):
        self._ready = True
        self.status.setText("인덱스 로드됨  ·  완전 오프라인  ·  모델: "
                            + str(self.cfg.llm.get("model", "?")))
        self.status.setProperty("state", "ok")
        self._set_enabled(True)
        self.input.setFocus()

    @pyqtSlot(str)
    def _on_init_failed(self, msg: str):
        self.status.setText("인덱스 없음 — 먼저 인덱싱이 필요합니다")
        self.status.setProperty("state", "bad")
        self.status.style().polish(self.status)
        self._append(
            "<div class='a err'><span class='alabel'>엔진을 시작할 수 없습니다</span><br>"
            + _esc(msg)
            + "<br><br>터미널에서 <b>python -m sovereign.cli index</b> 로 인덱스를 만든 뒤 다시 실행하세요.</div>"
        )

    def _on_send(self):
        if self._busy or not self._ready:
            return
        q = self.input.toPlainText().strip()
        if not q:
            return
        self.input.clear()
        self._append(f"<div class='q'><span class='qlabel'>질문</span><br>{_esc(q)}</div>")
        self._set_busy(True)
        self._ask_requested.emit(q)

    @pyqtSlot(dict)
    def _on_answered(self, res: dict):
        answer = _esc(res.get("answer", ""))
        sources = _sources_html(res.get("sources", []))
        self._append(f"<div class='a'><span class='alabel'>답변</span><br>{answer}</div>{sources}")
        self._set_busy(False)

    @pyqtSlot(str)
    def _on_failed(self, msg: str):
        self._append(f"<div class='a err'><span class='alabel'>오류</span><br>{_esc(msg)}</div>")
        self._set_busy(False)

    def closeEvent(self, e):  # noqa: N802
        self._thread.quit()
        self._thread.wait(2000)  # 생성 중이면 슬롯 종료까지 최대 대기 후 진행
        super().closeEvent(e)


def run(cfg: Config) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("소버린 도메인 AI")
    win = MainWindow(cfg)
    win.show()
    return app.exec()
