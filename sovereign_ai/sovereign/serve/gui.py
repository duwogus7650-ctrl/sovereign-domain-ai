"""YJH AI — 오프라인 데스크톱 GUI (PyQt6), Gemini 풍 세련된 라이트 테마.

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

from PyQt6.QtCore import Qt, QObject, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import (
    QFont, QPainter, QLinearGradient, QColor, QPen, QBrush, QPalette,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPlainTextEdit, QPushButton, QLabel, QFrame, QScrollArea, QStackedWidget,
    QGraphicsDropShadowEffect, QSizePolicy,
)

from ..config import Config

APP_NAME = "YJH AI"

# ── Gemini 풍 팔레트 ──
BLUE = "#1a73e8"          # Google 블루 액센트
INK = "#1f1f1f"           # 본문
SUB = "#5f6368"           # 보조 텍스트
BG = "#ffffff"            # 배경
SURFACE = "#f0f4f9"       # 입력 필/표면
USER_BUBBLE = "#e8f0fe"   # 사용자 말풍선(연한 블루)
BOT_BUBBLE = "#f5f7fa"    # 답변 말풍선(연한 그레이)
# Gemini 시그니처 그라데이션 (블루 → 퍼플 → 핑크)
GRAD = ("#4285F4", "#9B72CB", "#D96570")

FONT_STACK = '"Segoe UI", "Malgun Gothic", sans-serif'

APP_QSS = f"""
QMainWindow, QWidget {{ background: {BG}; font-family: {FONT_STACK}; }}
QScrollArea {{ border: none; background: {BG}; }}
QFrame#userBubble {{ background: {USER_BUBBLE}; border-radius: 18px; }}
QFrame#botBubble  {{ background: {BOT_BUBBLE};  border-radius: 18px; }}
QFrame#userBubble QLabel, QFrame#botBubble QLabel {{
    background: transparent; color: {INK}; font-size: 14px;
}}
QFrame#inputBar {{ background: {SURFACE}; border-radius: 26px; }}
QPlainTextEdit#input {{ background: transparent; border: none; color: {INK}; font-size: 15px; }}
QPushButton#send {{
    background: {BLUE}; color: white; border: none;
    border-radius: 20px; font-size: 18px; font-weight: bold;
}}
QPushButton#send:hover {{ background: #1765cc; }}
QPushButton#send:disabled {{ background: #c6d2e0; color: #eef2f7; }}
QLabel#statusText {{ color: {SUB}; font-size: 12px; }}
QLabel#busy {{ color: {SUB}; font-size: 12px; }}
QLabel#heroSub {{ color: {SUB}; font-size: 15px; }}
"""


def _esc(s: str) -> str:
    return html.escape(s or "").replace("\n", "<br>")


def _sources_html(sources: list[dict]) -> str:
    if not sources:
        return ""
    out = [f"<div style='margin-top:10px; color:{SUB}; font-size:12px;'>근거 출처<br>"]
    for n, h in enumerate(sources, 1):
        tag = "본문" if h.get("fulltext") else "초록"
        score = float(h.get("score", 0.0))
        title = html.escape((h.get("title", "") or "")[:110])
        url = h.get("abs_url", "") or ""
        line = (f"<span style='color:{BLUE};'>[{n}]</span> "
                f"{tag} · score={score:.3f} {title}")
        if url:
            u = html.escape(url)
            line += (f"<br>&nbsp;&nbsp;&nbsp;<a href='{u}' "
                     f"style='color:{BLUE}; text-decoration:none;'>{u}</a>")
        out.append(line + "<br>")
    out.append("</div>")
    return "".join(out)


class GradientLabel(QLabel):
    """Gemini 시그니처 그라데이션으로 글자를 칠하는 라벨."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setTextFormat(Qt.TextFormat.PlainText)

    def paintEvent(self, _e):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        grad = QLinearGradient(0, 0, max(1, self.width()), 0)
        grad.setColorAt(0.0, QColor(GRAD[0]))
        grad.setColorAt(0.5, QColor(GRAD[1]))
        grad.setColorAt(1.0, QColor(GRAD[2]))
        painter.setPen(QPen(QBrush(grad), 0))
        painter.setFont(self.font())
        painter.drawText(self.rect(), int(self.alignment().value), self.text())


class InputBox(QPlainTextEdit):
    """Enter=전송, Shift+Enter=줄바꿈."""
    submitted = pyqtSignal()

    def keyPressEvent(self, e):  # noqa: N802
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
        except Exception as e:
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
        self._pending_label: QLabel | None = None
        self._typing_timer: QTimer | None = None
        self._dots = 0
        self._build_ui()
        self._start_worker()

    # ── UI 구성 ──
    def _build_ui(self):
        self.setWindowTitle(APP_NAME)
        self.resize(920, 720)
        self.setStyleSheet(APP_QSS)
        self.setFont(QFont("Segoe UI", 10))

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 14, 18, 16)
        root.setSpacing(10)

        # 상단 바: 워드마크 + 상태
        topbar = QHBoxLayout()
        wordmark = GradientLabel(APP_NAME)
        wf = QFont("Segoe UI", 15)
        wf.setWeight(QFont.Weight.DemiBold)
        wordmark.setFont(wf)
        wordmark.setFixedHeight(28)
        wordmark.setMinimumWidth(110)
        topbar.addWidget(wordmark)
        topbar.addStretch(1)
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color:#bdc1c6; font-size:11px;")
        self.status_text = QLabel("준비 중…")
        self.status_text.setObjectName("statusText")
        topbar.addWidget(self.status_dot)
        topbar.addSpacing(5)
        topbar.addWidget(self.status_text)
        root.addLayout(topbar)

        # 본문: 히어로(빈 상태) ↔ 대화 스크롤
        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)

        # page 0: 히어로
        hero = QWidget()
        hv = QVBoxLayout(hero)
        hv.addStretch(1)
        greeting = GradientLabel("안녕하세요")
        gf = QFont("Malgun Gothic", 40)  # 한글 — 폴백 의존 없이 명시
        gf.setWeight(QFont.Weight.DemiBold)
        greeting.setFont(gf)
        greeting.setAlignment(Qt.AlignmentFlag.AlignCenter)
        greeting.setFixedHeight(64)
        hv.addWidget(greeting)
        sub = QLabel("오프라인 도메인 AI · 모터설계 · 제어 · 제어보드 · AI고장진단 · 강화학습")
        sub.setObjectName("heroSub")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hv.addWidget(sub)
        hv.addStretch(2)
        self.stack.addWidget(hero)

        # page 1: 대화
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        chat_inner = QWidget()
        self.chat_layout = QVBoxLayout(chat_inner)
        self.chat_layout.setContentsMargins(6, 6, 6, 6)
        self.chat_layout.setSpacing(4)
        self.chat_layout.addStretch(1)  # 메시지는 이 스트레치 앞에 삽입
        self.scroll.setWidget(chat_inner)
        self.stack.addWidget(self.scroll)

        # 생성 중 안내
        self.busy = QLabel("")
        self.busy.setObjectName("busy")
        self.busy.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.busy)

        # 입력 바 (필 모양 + 원형 전송)
        bar = QFrame()
        bar.setObjectName("inputBar")
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 2)
        shadow.setColor(QColor(0, 0, 0, 32))
        bar.setGraphicsEffect(shadow)
        bh = QHBoxLayout(bar)
        bh.setContentsMargins(20, 6, 8, 6)
        bh.setSpacing(8)
        self.input = InputBox()
        self.input.setObjectName("input")
        self.input.setPlaceholderText("질문을 입력하세요  (Enter 전송 · Shift+Enter 줄바꿈)")
        self.input.setFixedHeight(44)
        self.input.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.input.submitted.connect(self._on_send)
        bh.addWidget(self.input, 1)
        self.send_btn = QPushButton("↑")
        self.send_btn.setObjectName("send")
        self.send_btn.setFixedSize(40, 40)
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.clicked.connect(self._on_send)
        bh.addWidget(self.send_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        root.addWidget(bar)

        self.setCentralWidget(central)
        self._set_enabled(False)  # 엔진 준비 전 잠금

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

    # ── 메시지/스크롤 ──
    def _add_message(self, role: str, html_text: str) -> QLabel:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(2, 4, 2, 4)
        h.setSpacing(0)
        bubble = QFrame()
        bubble.setObjectName("userBubble" if role == "user" else "botBubble")
        bubble.setMaximumWidth(680)
        bl = QVBoxLayout(bubble)
        bl.setContentsMargins(15, 11, 15, 11)
        bl.setSpacing(0)
        lbl = QLabel(html_text)
        lbl.setWordWrap(True)
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        lbl.setOpenExternalLinks(True)
        bl.addWidget(lbl)
        if role == "user":
            h.addStretch(1)
            h.addWidget(bubble)
        else:
            h.addWidget(bubble)
            h.addStretch(1)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, row)
        self._scroll_bottom_soon()
        return lbl

    def _scroll_bottom_soon(self):
        def go():
            sb = self.scroll.verticalScrollBar()
            sb.setValue(sb.maximum())
        QTimer.singleShot(0, go)
        QTimer.singleShot(60, go)

    def _start_typing(self, label: QLabel):
        self._dots = 0

        def tick():
            self._dots = (self._dots % 3) + 1
            on = "●" * self._dots
            off = "●" * (3 - self._dots)
            label.setText(f"<span style='color:{SUB};'>{on}</span>"
                          f"<span style='color:#c6cdd6;'>{off}</span>")
        self._typing_timer = QTimer(self)
        self._typing_timer.timeout.connect(tick)
        self._typing_timer.start(350)
        tick()

    def _stop_typing(self):
        if self._typing_timer is not None:
            self._typing_timer.stop()
            self._typing_timer = None

    # ── 상태 토글 ──
    def _set_enabled(self, on: bool):
        self.input.setEnabled(on)
        self.send_btn.setEnabled(on)

    def _set_busy(self, busy: bool):
        self._busy = busy
        self.busy.setText("⏳ 생성 중…  CPU에서 수 분 걸릴 수 있습니다" if busy else "")
        self._set_enabled(self._ready and not busy)
        if not busy and self._ready:
            self.input.setFocus()

    def _set_status(self, color: str, text: str):
        self.status_dot.setStyleSheet(f"color:{color}; font-size:11px;")
        self.status_text.setText(text)

    # ── 슬롯 ──
    @pyqtSlot()
    def _on_ready(self):
        self._ready = True
        self._set_status("#34a853", f"오프라인 · {self.cfg.llm.get('model', '?')}")
        self._set_enabled(True)
        self.input.setFocus()

    @pyqtSlot(str)
    def _on_init_failed(self, msg: str):
        self._set_status("#ea4335", "인덱스 없음 — 인덱싱 필요")
        if self.stack.currentIndex() == 0:
            self.stack.setCurrentIndex(1)
        self._add_message(
            "bot",
            "<b>엔진을 시작할 수 없습니다</b><br>" + _esc(msg)
            + "<br><br>터미널에서 <b>python -m sovereign.cli index</b> 로 인덱스를 만든 뒤 다시 실행하세요.",
        )

    def _on_send(self):
        if self._busy or not self._ready:
            return
        q = self.input.toPlainText().strip()
        if not q:
            return
        self.input.clear()
        if self.stack.currentIndex() == 0:
            self.stack.setCurrentIndex(1)
        self._add_message("user", _esc(q))
        self._pending_label = self._add_message("bot", "")
        self._start_typing(self._pending_label)
        self._set_busy(True)
        self._ask_requested.emit(q)

    @pyqtSlot(dict)
    def _on_answered(self, res: dict):
        self._stop_typing()
        body = _esc(res.get("answer", "")) + _sources_html(res.get("sources", []))
        if self._pending_label is not None:
            self._pending_label.setText(body)
        self._pending_label = None
        self._set_busy(False)
        self._scroll_bottom_soon()

    @pyqtSlot(str)
    def _on_failed(self, msg: str):
        self._stop_typing()
        if self._pending_label is not None:
            self._pending_label.setText(f"<b>오류</b><br>{_esc(msg)}")
        self._pending_label = None
        self._set_busy(False)

    def closeEvent(self, e):  # noqa: N802
        self._stop_typing()
        self._thread.quit()
        self._thread.wait(2000)  # 생성 중이면 슬롯 종료까지 최대 대기 후 진행
        super().closeEvent(e)


def run(cfg: Config) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    # QLabel 링크 색상을 블루로
    pal = app.palette()
    pal.setColor(QPalette.ColorRole.Link, QColor(BLUE))
    app.setPalette(pal)
    win = MainWindow(cfg)
    win.show()
    return app.exec()
