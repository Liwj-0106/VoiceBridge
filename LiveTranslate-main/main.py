"""
LiveTranslate - Phase 0 Prototype
Real-time audio translation using WASAPI loopback + faster-whisper + LLM.
"""

import sys
import signal
import logging
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import yaml
import time
import numpy as np
from pathlib import Path
from datetime import datetime

from voicebridge.model_manager import (
    apply_cache_env,
    get_missing_models,
    is_asr_cached,
    ASR_DISPLAY_NAMES,
    MODELS_DIR,
)

# 【重要】必须在 import torch 之前设置缓存路径，确保 TORCH_HOME 被正确识别
apply_cache_env()

import os

# 【重要】torch 必须在 PyQt6 之前导入，避免 Windows 上的 DLL 冲突
import torch  # noqa: F401

from voicebridge.audio.capture import AudioCapture
from voicebridge.vad.processor import VADProcessor
from voicebridge.asr.whisper import ASREngine
from voicebridge.translation.translator import Translator, RepetitionError

from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QDialog, QMessageBox
from PyQt6.QtGui import QAction, QActionGroup, QIcon, QPixmap, QPainter, QColor, QFont
from PyQt6.QtCore import QTimer, Qt

from subtitle_overlay import SubtitleOverlay
from subtitle_window import SubtitleWindow
from log_window import LogWindow
from control_panel import (
    ControlPanel,
    SETTINGS_FILE,
    _load_saved_settings,
    _save_settings,
)
from dialogs import (
    SetupWizardDialog,
    ModelDownloadDialog,
    _ModelLoadDialog,
)
from voicebridge.i18n import t, set_lang, LANGUAGES, COMMON_LANG_CODES


def setup_logging():
    """
    配置日志系统：
    1. 同时输出到文件（DEBUG级别）和控制台（INFO级别）
    2. 屏蔽第三方库的冗余日志
    3. 设置全局异常钩子，捕获所有未处理异常
    """
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    # 日志文件名包含时间戳，方便区分
    log_file = log_dir / f"livetrans_{datetime.now():%Y%m%d_%H%M%S}.log"

    # 文件Handler：记录DEBUG以上所有信息
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    # 控制台Handler：只显示INFO以上信息，减少噪音
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler.setFormatter(fmt)
    console_handler.setFormatter(fmt)

    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler])

    # 屏蔽第三方库的冗余日志输出
    for noisy in (
        "httpcore",
        "httpx",
        "openai",
        "filelock",
        "huggingface_hub",
        "funasr",
        "modelscope",
        "onnxruntime",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.info(f"Log file: {log_file}")

    # FunASR/ModelScope 会污染root logger，抑制一下
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("LiveTranslate").setLevel(logging.DEBUG)

    _logger = logging.getLogger("LiveTranslate")

    # 全局异常钩子：捕获主线程未处理的异常
    def _excepthook(exc_type, exc_value, exc_tb):
        _logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook

    # 线程异常钩子：捕获所有后台线程的未处理异常
    def _thread_excepthook(args):
        _logger.critical(
            f"Uncaught exception in thread {args.thread}",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = _thread_excepthook

    return _logger


log = logging.getLogger("LiveTranslate")


def create_app_icon() -> QIcon:
    """用代码生成应用图标（蓝色圆角矩形 + "LT" 文字），避免需要外部图标文件"""
    pix = QPixmap(64, 64)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(60, 130, 240))  # 蓝色背景
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(4, 4, 56, 56, 12, 12)  # 圆角矩形
    p.setPen(QColor(255, 255, 255))  # 白色文字
    p.setFont(QFont("Consolas", 28, QFont.Weight.Bold))
    p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "LT")
    p.end()
    return QIcon(pix)


def load_config():
    """加载 config.yaml 基础配置文件"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class LiveTranslateApp:
    """
    核心应用类，管理音频捕获→VAD检测→ASR识别→翻译的完整管线

    管线由三个线程组成：
    1. _capture_thread：音频捕获 + VAD处理
    2. _asr_thread：ASR识别 + 翻译触发
    3. _tl_executor：翻译线程池（并行执行LLM调用）
    """

    def __init__(self, config):
        self._config = config
        self._running = False      # 管线是否运行
        self._paused = False       # 是否暂停
        self._asr_ready = False    # ASR模型是否加载完成

        # 音频捕获：WASAPI loopback，32ms分块
        self._audio = AudioCapture(
            device=config["audio"].get("device"),
            sample_rate=config["audio"]["sample_rate"],
            chunk_duration=config["audio"]["chunk_duration"],
        )

        # VAD语音活动检测：检测语音段落边界
        self._vad = VADProcessor(
            sample_rate=config["audio"]["sample_rate"],
            threshold=config["asr"]["vad_threshold"],
            min_speech_duration=config["asr"]["min_speech_duration"],
            max_speech_duration=config["asr"]["max_speech_duration"],
            chunk_duration=config["audio"]["chunk_duration"],
        )

        # ASR相关
        self._asr_type = None     # 当前ASR引擎类型 (whisper/sensevoice/funasr-nano/anime-whisper)
        self._asr = None          # ASR引擎实例
        self._asr_device = config["asr"]["device"]  # cuda/cpu
        self._whisper_model_size = config["asr"]["model_size"]

        # 线程安全锁：保护ASR引擎和VAD缓冲的并发访问
        self._asr_lock = threading.Lock()
        self._vad_lock = threading.Lock()

        self._target_language = config["translation"]["target_language"]

        # 翻译器：调用OpenAI兼容API
        self._translator = Translator(
            api_base=config["translation"]["api_base"],
            api_key=config["translation"]["api_key"],
            model=config["translation"]["model"],
            target_language=self._target_language,
            max_tokens=config["translation"]["max_tokens"],
            temperature=config["translation"]["temperature"],
            streaming=config["translation"]["streaming"],
            system_prompt=config["translation"].get("system_prompt"),
        )

        # UI组件引用（通过setter注入）
        self._overlay = None      # 字幕悬浮窗
        self._subwin = None        # OBS字幕窗口
        self._panel = None         # 设置面板

        # 管线线程
        self._capture_thread = None
        self._asr_thread = None
        self._asr_queue = queue.Queue(maxsize=16)  # ASR任务队列
        self._tl_executor = ThreadPoolExecutor(max_workers=8)  # 翻译线程池

        # 统计计数
        self._asr_count = 0              # ASR识别次数
        self._translate_count = 0        # 翻译次数
        self._total_prompt_tokens = 0    # 累计输入Token
        self._total_completion_tokens = 0  # 累计输出Token
        self._input_price = 0.0          # 输入价格（每1M Token）
        self._output_price = 0.0         # 输出价格（每1M Token）
        self._msg_id = 0                 # 消息ID递进
        self._last_original = ""          # 最近一次识别的原文
        self._last_msg_id = 0            # 最近一次消息ID

        # 增量ASR状态（边说边翻，降低延迟）
        self._incremental_enabled = True  # 增量ASR开关
        self._interim_interval = 2.0      # 增量检测间隔（秒），超过这个时间检测一次
        self._interim_pending = ""         # 待处理的短文本片段（≤8字符的碎片）
        self._interim_active = False       # 是否处于增量模式
        self._last_interim_samples = 0     # 上次增量检测时的缓冲样本数
        self._last_interim_check_time = 0.0  # 上次增量检测的时间
        self._interim_committed_tail = ""  # 已提交文本的尾部（用于去重）

    def set_overlay(self, overlay: SubtitleOverlay):
        """设置字幕悬浮窗引用"""
        self._overlay = overlay

    def set_subtitle_window(self, subwin: SubtitleWindow):
        """设置OBS字幕窗口引用"""
        self._subwin = subwin

    def set_panel(self, panel: ControlPanel):
        """设置控制面板引用，并连接信号"""
        self._panel = panel
        # 连接控制面板的信号，当设置变化时实时响应
        panel.settings_changed.connect(self._on_settings_changed)      # 设置变化时
        panel.model_changed.connect(self._on_model_changed)            # 模型切换时
        panel.models_list_changed.connect(self._on_models_list_changed)  # 模型列表变化时

    def _on_models_list_changed(self, models: list, active_idx: int):
        """模型列表变化时更新overlay的模型下拉框"""
        if self._overlay:
            self._overlay.set_models(models, active_idx)

    def _on_settings_changed(self, settings):
        """响应控制面板的设置变更"""
        # 更新VAD参数（阈值、静音模式等）
        self._vad.update_settings(settings)

        # 样式变更 → 更新overlay配色
        if "style" in settings and self._overlay:
            self._overlay.apply_style(settings["style"])

        # ASR语言变更 → 通知引擎
        if "asr_language" in settings and self._asr:
            self._asr.set_language(settings["asr_language"])

        # ASR设备变更 → 尝试原地迁移（to_device），失败则标记需重载
        new_device = settings.get("asr_device")
        if new_device and new_device != self._asr_device:
            old_device = self._asr_device
            self._asr_device = new_device
            if self._asr is not None and hasattr(self._asr, "to_device"):
                result = self._asr.to_device(new_device)
                if result is not False:
                    log.info(f"ASR device migrated: {old_device} -> {new_device}")
                    if self._overlay:
                        display_name = ASR_DISPLAY_NAMES.get(
                            self._asr_type, self._asr_type
                        )
                        self._overlay.update_asr_device(
                            f"{display_name} [{new_device}]"
                        )
                    import gc

                    gc.collect()
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                else:
                    self._asr_type = None  # ctranslate2: force reload
            else:
                self._asr_type = None  # no engine loaded: force reload
        new_whisper_size = settings.get("whisper_model_size")
        if new_whisper_size and new_whisper_size != self._whisper_model_size:
            self._whisper_model_size = new_whisper_size
            if self._asr_type == "whisper":
                self._asr_type = None
        if "asr_engine" in settings:
            self._switch_asr_engine(settings["asr_engine"])
        if "audio_device" in settings:
            old_device = self._audio._device_name
            self._audio.set_device(settings["audio_device"])
            if old_device != settings.get("audio_device"):
                self._vad.flush()
                self._vad._reset()
                if self._overlay:
                    self._overlay.update_monitor(0.0, 0.0)
        # 麦克风设备变更
        if "mic_device" in settings:
            self._audio.set_mic_device(settings["mic_device"])

        # 增量ASR开关和间隔
        if "incremental_asr" in settings:
            self._incremental_enabled = settings["incremental_asr"]
        if "interim_interval" in settings:
            self._interim_interval = settings["interim_interval"]

        # 目标语言变更
        if "target_language" in settings:
            self._target_language = settings["target_language"]
            if self._overlay:
                self._overlay.set_target_language(self._target_language)

        # 翻译超时变更
        if "timeout" in settings and self._translator:
            self._translator.set_timeout(settings["timeout"])

    def _on_target_language_changed(self, lang: str):
        """目标语言变化时的处理（从overlay或tray触发）"""
        self._target_language = lang
        log.info(f"Target language: {lang}")
        if self._translator:
            self._translator.set_target_language(lang)
        # 同步保存到设置文件
        if self._panel:
            settings = self._panel.get_settings()
            settings["target_language"] = lang
            from control_panel import _save_settings
            _save_settings(settings)

    def _on_model_changed(self, model_config: dict):
        """翻译模型切换：创建新的Translator实例"""
        log.info(f"Switching translator: {model_config['name']} ({model_config['model']})")
        prompt = None
        if self._panel:
            prompt = self._panel.get_settings().get("system_prompt")
        if not prompt:
            prompt = self._config["translation"].get("system_prompt")
        timeout = 10
        if self._panel:
            timeout = self._panel.get_settings().get("timeout", 10)
        self._translator = Translator(
            api_base=model_config["api_base"],
            api_key=model_config["api_key"],
            model=model_config["model"],
            target_language=self._target_language,
            max_tokens=self._config["translation"]["max_tokens"],
            temperature=self._config["translation"]["temperature"],
            streaming=model_config.get("streaming", True),
            system_prompt=prompt,
            proxy=model_config.get("proxy", "none"),
            no_system_role=model_config.get("no_system_role", False),
            no_think=model_config.get("no_think", True),
            json_response=model_config.get("json_response", False),
            timeout=timeout,
            overrides=model_config.get("overrides"),
            extra_body=model_config.get("extra_body"),
        )
        self._translator.set_context_turns(model_config.get("context_turns", 0))
        self._input_price = model_config.get("input_price", 0)
        self._output_price = model_config.get("output_price", 0)

    def _switch_asr_engine(self, engine_type: str):
        """
        切换ASR引擎（whisper/sensevoice/funasr-nano/anime-whisper）

        流程：检查缓存 → 下载缺失模型 → 后台线程加载 → 释放旧引擎
        """
        if engine_type == self._asr_type:
            return
        log.info(f"Switching ASR engine: {self._asr_type} -> {engine_type}")

        # 重置状态
        self._asr_ready = False
        self._interim_active = False
        self._interim_pending = ""
        self._last_interim_samples = 0
        self._last_interim_check_time = 0.0
        self._interim_committed_tail = ""

        # 切换期间清空VAD缓冲，防止音频积累
        self._vad.flush()
        self._vad._reset()
        device = self._asr_device
        hub = "ms"
        if self._panel:
            hub = self._panel.get_settings().get("hub", "ms")

        # 获取模型大小配置（Whisper可配置大小）
        model_size = self._config["asr"]["model_size"]
        if self._panel:
            model_size = self._panel.get_settings().get(
                "whisper_model_size", model_size
            )

        # 检查模型是否已缓存，未缓存则弹出下载对话框
        cached = is_asr_cached(engine_type, model_size, hub)
        display_name = ASR_DISPLAY_NAMES.get(engine_type, engine_type)
        if engine_type == "whisper":
            display_name = f"Whisper {model_size}"

        # 父窗口：优先使用设置面板，否则用overlay
        parent = (
            self._panel if self._panel and self._panel.isVisible() else self._overlay
        )

        # 模型未缓存？弹出下载对话框
        if not cached:
            missing = get_missing_models(engine_type, model_size, hub)
            missing = [m for m in missing if m["type"] != "silero-vad"]
            if missing:
                dlg = ModelDownloadDialog(missing, hub=hub, parent=parent)
                if dlg.exec() != QDialog.DialogCode.Accepted:
                    log.info(f"Download cancelled/failed: {engine_type}")
                    # Restore readiness if old engine is still available
                    if self._asr is not None:
                        self._asr_ready = True
                    return

        with self._asr_lock:
            old_engine = self._asr
            self._asr = None

        dlg = _ModelLoadDialog(
            t("loading_model").format(name=display_name), parent=parent
        )

        new_asr = [None]
        load_error = [None]

        def _load():
            nonlocal old_engine
            try:
                if old_engine is not None:
                    log.info(
                        f"Releasing old ASR engine: {old_engine.__class__.__name__}"
                    )
                    if hasattr(old_engine, "unload"):
                        old_engine.unload()
                    old_engine = None
                dev = device
                dev_index = 0
                if dev.startswith("cuda:"):
                    part = dev.split("(")[0].strip()  # "cuda:0"
                    dev_index = int(part.split(":")[1])
                    dev = "cuda"

                if engine_type == "sensevoice":
                    from voicebridge.asr.sensevoice import SenseVoiceEngine

                    new_asr[0] = SenseVoiceEngine(device=device, hub=hub)
                elif engine_type in ("funasr-nano", "funasr-mlt-nano"):
                    from voicebridge.asr.funasr_nano import FunASRNanoEngine

                    new_asr[0] = FunASRNanoEngine(
                        device=device, hub=hub, engine_type=engine_type
                    )
                elif engine_type == "anime-whisper":
                    from voicebridge.asr.anime_whisper import AnimeWhisperEngine

                    dev_str = dev if dev == "cpu" else f"cuda:{dev_index}"
                    new_asr[0] = AnimeWhisperEngine(device=dev_str, hub=hub)
                else:
                    download_root = str((MODELS_DIR / "huggingface" / "hub").resolve())
                    compute = self._config["asr"]["compute_type"]
                    if dev == "cpu" and compute == "float16":
                        compute = "int8"
                    new_asr[0] = ASREngine(
                        model_size=model_size,
                        device=dev,
                        device_index=dev_index,
                        compute_type=compute,
                        language=self._config["asr"]["language"],
                        download_root=download_root,
                    )
            except Exception as e:
                load_error[0] = str(e)
                log.error(f"Failed to load ASR engine: {e}", exc_info=True)

        thread = threading.Thread(target=_load, daemon=True)
        thread.start()

        poll_timer = QTimer()

        def _check():
            if not thread.is_alive():
                poll_timer.stop()
                dlg.accept()

        poll_timer.setInterval(100)
        poll_timer.timeout.connect(_check)
        poll_timer.start()

        dlg.exec()
        poll_timer.stop()

        if load_error[0]:
            QMessageBox.warning(
                parent,
                t("error_title"),
                t("error_load_asr").format(error=load_error[0]),
            )
            # Old engine was already released; mark ASR as unavailable
            self._asr_type = None
            return

        # 加载成功，更新状态
        self._asr = new_asr[0]
        self._asr_type = engine_type
        if self._panel:
            asr_lang = self._panel.get_settings().get("asr_language", "auto")
            self._asr.set_language(asr_lang)
        self._asr_ready = True
        if self._overlay:
            self._overlay.update_asr_device(f"{display_name} [{device}]")
        log.info(f"ASR engine ready: {engine_type} on {device}")

    def _compute_cost(self):
        """根据Token使用量和价格计算翻译费用"""
        if self._input_price > 0 or self._output_price > 0:
            return (self._total_prompt_tokens * self._input_price +
                    self._total_completion_tokens * self._output_price) / 1_000_000
        return 0.0

    def _translate_async(self, msg_id, text, source_lang, extra_langs=None):
        """
        异步翻译：流式调用翻译API，实时更新UI

        流程：
        1. 流式翻译，逐词更新overlay（看得见的实时效果）
        2. 翻译完成后更新最终结果和统计
        3. 字幕窗口需要额外语言时并行翻译
        """
        try:
            tl_start = time.perf_counter()
            translated = None

            # 流式翻译：逐部分yield，实时显示
            for partial in self._translator.translate_iter(text, source_lang):
                translated = partial
                if self._overlay:
                    self._overlay.update_streaming(msg_id, partial)

            tl_ms = (time.perf_counter() - tl_start) * 1000
            self._translate_count += 1

            # 统计Token使用量
            pt, ct = self._translator.last_usage
            self._total_prompt_tokens += pt
            self._total_completion_tokens += ct
            cost = self._compute_cost()

            log.info(f"Translate ({tl_ms:.0f}ms): {translated}")

            # 更新overlay：最终翻译结果 + 统计信息
            if self._overlay:
                self._overlay.update_translation(msg_id, translated, tl_ms)
                self._overlay.update_stats(
                    self._asr_count,
                    self._translate_count,
                    self._total_prompt_tokens,
                    self._total_completion_tokens,
                    cost,
                )

            # 字幕窗口：更新主语言翻译，额外语言并行翻译
            if self._subwin and self._subwin.isVisible() and translated:
                tl_dict = {self._target_language: translated}
                if extra_langs:
                    self._translate_extra_langs(text, source_lang, extra_langs, tl_dict)
                self._subwin.update_text(text, tl_dict)

        # 重复循环错误（模型输出重复内容）
        except RepetitionError:
            log.warning("Repetition loop detected, model may not support structured output well")
            if self._overlay:
                self._overlay.update_translation(
                    msg_id, f"[{t('error_repetition')}]", 0
                )
        # 网络/认证等常见错误，只记warning；其他错误记error
        except Exception as e:
            import openai
            if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError,
                              openai.AuthenticationError, openai.APIStatusError,
                              TimeoutError, ConnectionError)):
                log.warning(f"Translate error: {e}")
            else:
                log.error(f"Translate error: {e}", exc_info=True)
            if self._overlay:
                self._overlay.update_translation(msg_id, f"[error: {e}]", 0)

    def _translate_extra_langs(self, text, source_lang, extra_langs, tl_dict):
        """
        字幕窗口额外语言翻译（并行）

        字幕窗口可能需要多种语言翻译，主语言已经翻好，
        这里并行翻译其他目标语言（如日语、韩语等）
        """
        from concurrent.futures import as_completed

        def _do_translate(lang):
            # 创建临时翻译器指定目标语言
            translator = self._translator.with_target_language(lang)
            return lang, translator.translate(text, source_lang)

        # 并行提交所有额外语言的翻译任务
        futures = []
        for lang in extra_langs:
            futures.append(self._tl_executor.submit(_do_translate, lang))

        # 收集结果
        for future in as_completed(futures):
            try:
                lang, result = future.result()
                tl_dict[lang] = result
                log.info(f"Extra translate [{lang}]: {result}")
            except Exception as e:
                import openai
                if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError,
                                  openai.AuthenticationError, openai.APIStatusError,
                                  TimeoutError, ConnectionError)):
                    log.warning(f"Extra translate error: {e}")
                else:
                    log.error(f"Extra translate error: {e}", exc_info=True)

    def _translate_subwin_only(self, text, source_lang, extra_langs):
        """
        字幕窗口专用翻译（源语言==目标语言时）

        此时overlay不翻译（same language），但字幕窗口可能需要其他语言
        """
        tl_dict = {self._target_language: text}  # 同语言，直接用原文
        self._translate_extra_langs(text, source_lang, extra_langs, tl_dict)
        if self._subwin and self._subwin.isVisible():
            self._subwin.update_text(text, tl_dict)

    def start(self):
        """
        启动管线：创建线程，开始音频捕获和处理

        线程：
        - _capture_thread：音频捕获 + VAD检测
        - _asr_thread：ASR识别 + 翻译触发
        """
        if self._running:
            return

        # 根据字幕窗口语言数调整线程池大小
        n = len(self._subwin.get_target_languages()) if self._subwin else 1
        self._tl_executor = ThreadPoolExecutor(max_workers=max(8, n + 1))
        self._asr_queue = queue.Queue(maxsize=16)

        self._running = True
        self._paused = False
        self._audio.start()

        # 启动捕获线程：_capture_loop()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        # 启动ASR线程：_asr_loop()
        self._asr_thread = threading.Thread(
            target=self._asr_loop, daemon=True
        )
        self._capture_thread.start()
        self._asr_thread.start()
        log.info("Pipeline started (capture + ASR threads)")

    def stop(self):
        """
        停止管线：优雅关闭各线程，处理VAD残留音频

        注意：必须在join线程之前向ASR队列放入None结束信号
        """
        self._running = False
        self._audio.stop()

        # 等待捕获线程结束（最多3秒）
        if self._capture_thread:
            self._capture_thread.join(timeout=3)
            self._capture_thread = None

        # 发送结束信号并等待ASR线程
        self._asr_queue.put(None)
        if self._asr_thread:
            self._asr_thread.join(timeout=10)
            if self._asr_thread.is_alive():
                log.warning("ASR thread still running after timeout, proceeding with cleanup")
            self._asr_thread = None

        # 管线线程结束后，处理VAD中剩余的音频
        if self._interim_active:
            # 增量模式：强制flush
            remaining = self._vad.force_flush()
            if remaining is not None and self._asr_ready:
                self._process_interim_final(remaining)
        else:
            # 正常模式：普通flush
            remaining = self._vad.flush()
            if remaining is not None and self._asr_ready:
                self._process_segment(remaining)

        # 重置增量状态
        self._interim_active = False
        self._interim_pending = ""
        self._last_interim_samples = 0
        self._last_interim_check_time = 0.0
        self._interim_committed_tail = ""

        # 关闭翻译线程池（不等待）
        self._tl_executor.shutdown(wait=False)
        log.info("Pipeline stopped")

    def pause(self):
        """暂停管线：停止音频处理，但保留缓冲"""
        self._paused = True
        self._interim_active = False
        self._interim_pending = ""
        self._last_interim_samples = 0
        self._last_interim_check_time = 0.0
        self._interim_committed_tail = ""
        if self._overlay:
            self._overlay.update_monitor(0.0, 0.0)
        log.info("Pipeline paused")

    def resume(self):
        """恢复管线：继续音频处理"""
        self._paused = False
        log.info("Pipeline resumed")

    def _process_segment(self, speech_segment):
        """
        处理完整的语音段落：ASR识别 → 翻译 → UI更新

        由ASR线程或stop()调用，处理VAD检测到的一个完整语音段落
        """
        seg_len = len(speech_segment) / 16000
        log.info(f"Speech segment: {seg_len:.1f}s")

        # === ASR识别 ===
        asr_start = time.perf_counter()
        with self._asr_lock:
            if not self._asr_ready or self._asr is None:
                return
            try:
                result = self._asr.transcribe(speech_segment)
            except Exception as e:
                log.error(f"ASR error: {e}", exc_info=True)
                return
        asr_ms = (time.perf_counter() - asr_start) * 1000
        if asr_ms > 10000:
            log.warning(f"ASR took {asr_ms:.0f}ms, possible hang")
        if result is None:
            return

        original_text = result["text"].strip()

        # 过滤：空文本或仅标点符号
        if not original_text or not any(c.isalnum() for c in original_text):
            log.debug(f"ASR returned empty/punctuation-only, skipping: '{result['text']}'")
            return

        # 噪声过滤：长音频但文字极少（≤3个字母数字字符）
        alnum_chars = sum(1 for c in original_text if c.isalnum())
        if seg_len >= 2.0 and alnum_chars <= 3:
            log.debug(f"Noise filter: {seg_len:.1f}s segment produced only '{original_text}', skipping")
            return

        source_lang = result["language"]

        # 语言过滤：如果设置了固定ASR语言但识别结果不匹配，则丢弃
        asr_lang_setting = self._panel.get_settings().get("asr_language", "auto") if self._panel else "auto"
        if asr_lang_setting != "auto" and source_lang != asr_lang_setting:
            log.info(f"Language filter: expected '{asr_lang_setting}' but got '{source_lang}', discarding: {original_text[:60]}")
            return

        # 更新统计
        self._asr_count += 1
        self._msg_id += 1
        msg_id = self._msg_id
        timestamp = datetime.now().strftime("%H:%M:%S")
        log.info(f"ASR [{source_lang}] ({asr_ms:.0f}ms): {original_text}")

        # === 显示原文到overlay ===
        if self._overlay:
            self._overlay.add_message(msg_id, timestamp, original_text, source_lang, asr_ms)

        # 记录原文（供字幕窗口使用）
        self._last_original = original_text
        self._last_msg_id = msg_id

        target_lang = self._target_language

        # === 字幕窗口额外语言 ===
        # 字幕窗口可能需要多种语言翻译，收集主语言和源语言之外的其他语言
        extra_langs = set()
        if self._subwin and self._subwin.isVisible():
            subwin_langs = self._subwin.get_target_languages()
            extra_langs = subwin_langs - {target_lang, source_lang}

        # === 翻译决策 ===
        if source_lang == target_lang:
            # 源语言==目标语言：不翻译，只更新统计
            log.info(f"Same language ({source_lang}), no translation")
            if self._overlay:
                self._overlay.update_translation(msg_id, "", 0)
                self._overlay.update_stats(
                    self._asr_count,
                    self._translate_count,
                    self._total_prompt_tokens,
                    self._total_completion_tokens,
                    self._compute_cost(),
                )
            # 字幕窗口：仍需处理额外语言
            if self._subwin and self._subwin.isVisible():
                if extra_langs:
                    try:
                        self._tl_executor.submit(
                            self._translate_subwin_only, original_text, source_lang, extra_langs
                        )
                    except RuntimeError:
                        pass
                else:
                    self._subwin.update_text(original_text, {target_lang: original_text})
        else:
            # 不同语言：提交异步翻译任务
            try:
                self._tl_executor.submit(
                    self._translate_async, msg_id, original_text, source_lang,
                    extra_langs or None,
                )
            except RuntimeError:
                log.warning("Translation executor shut down, skipping")

    # ═══════════════════════════════════════════════════════════
    #                    增量 ASR（边说边翻）
    # ═══════════════════════════════════════════════════════════

    _pysbd_cache = {}  # lang -> pysbd.Segmenter 句子分割器缓存

    @staticmethod
    def _get_segmenter(lang: str):
        """获取指定语言的句子分割器（pysbd）"""
        import pysbd
        if lang not in LiveTranslateApp._pysbd_cache:
            pysbd_lang = lang if lang in pysbd.languages.LANGUAGE_CODES else "en"
            LiveTranslateApp._pysbd_cache[lang] = pysbd.Segmenter(
                language=pysbd_lang, clean=False
            )
        return LiveTranslateApp._pysbd_cache[lang]

    def _split_sentences(self, text: str, lang: str = "en") -> list[str]:
        """
        使用pysbd库分割句子，回退策略处理未分割的长文本

        策略：
        1. pysbd库分割（支持23种语言，高精度）
        2. CJK文本（含有"、"，>25字符）：在最后一个CJK逗号处切分
        3. 西文文本（>60字符）：在最后一个逗号处切分
        """
        seg = self._get_segmenter(lang)
        parts = [p for p in seg.segment(text) if p.strip()]
        if len(parts) > 1:
            return parts

        # 回退策略：长文本按逗号切分
        min_len = 25 if any(c == '、' for c in text) else 60
        if len(text) > min_len:
            for i in range(len(text) - 8, 5, -1):
                if text[i] in ',，;；、':
                    before = text[:i + 1].strip()
                    after = text[i + 1:].strip()
                    # 确保切分点前后都有足够内容，避免碎片
                    if before and after and len(before) > 15 and len(after) > 3:
                        return [before, after]

        return parts

    @staticmethod
    def _is_short_utterance(text: str) -> bool:
        """判断是否为短文本片段（≤8个字母数字字符），这类通常是噪声或填充词"""
        alnum = sum(1 for c in text if c.isalnum())
        return alnum <= 8

    def _strip_committed_overlap(self, text: str) -> str:
        """
        去除与已提交内容的重叠部分（回声去重）

        增量ASR中，已提交的句子尾部可能被下一段重新识别，
        这里去除这种重叠，避免重复翻译
        """
        if not self._interim_committed_tail:
            return text
        tail = self._interim_committed_tail.lower().rstrip()
        text_lower = text.lower()
        # 检查text是否以committed_tail的某后缀开头
        max_check = min(len(tail), len(text_lower))
        for overlap_len in range(max_check, 2, -1):
            if text_lower[:overlap_len] == tail[-overlap_len:]:
                stripped = text[overlap_len:].strip()
                if stripped:
                    log.debug(f"Stripped echo overlap ({overlap_len} chars): '{text[:overlap_len]}...'")
                    return stripped
                return ""
        return text

    def _do_interim_asr(self) -> bool:
        """Run ASR on current VAD buffer, output complete sentences, trim consumed audio.
        Returns True if any sentences were committed."""
        with self._vad_lock:
            peek = self._vad.peek_buffer()  # 查看当前VAD缓冲（不取出）
        if peek is None:
            return False
        audio, duration = peek

        # 音频太短（<1.5秒）不处理
        if duration < 1.5:
            return False

        # Whisper支持词级时间戳，可精确trim
        use_word_ts = self._asr_type == "whisper"

        # 执行ASR识别
        asr_start = time.perf_counter()
        with self._asr_lock:
            if not self._asr_ready or self._asr is None:
                return False
            try:
                result = self._asr.transcribe(audio, word_timestamps=use_word_ts) if use_word_ts else self._asr.transcribe(audio)
            except Exception as e:
                log.error(f"Interim ASR error: {e}", exc_info=True)
                return False
        asr_ms = (time.perf_counter() - asr_start) * 1000

        if result is None:
            return False

        full_text = result["text"].strip()
        if not full_text or not any(c.isalnum() for c in full_text):
            return False

        # 去除与已提交内容的重叠（回声去重）
        full_text = self._strip_committed_overlap(full_text)
        if not full_text:
            return False

        # 句子分割
        split_start = time.perf_counter()
        sentences = self._split_sentences(full_text, result["language"])
        split_ms = (time.perf_counter() - split_start) * 1000
        if len(sentences) <= 1:
            return False
        log.debug(f"Interim split [{result['language']}] ({split_ms:.1f}ms): {len(sentences)} parts -> {sentences}")

        # 除了最后一句，前面的都是完整句子（最后一句用户可能还在说）
        complete = sentences[:-1]

        committed_text = ""
        for sent in complete:
            committed_text += sent

        if not committed_text.strip():
            return False

        # === 计算音频修剪点 ===
        total_samples = len(audio)
        if use_word_ts and result.get("words"):
            # Whisper词级时间戳：精确找到最后一个已提交词的时间点
            words = result["words"]
            committed_lower = committed_text.lower().rstrip()
            char_pos = 0
            last_word_end = 0.0
            for w in words:
                word_text = w["word"].strip()
                idx = committed_lower.find(word_text.lower(), char_pos)
                if idx >= 0:
                    char_pos = idx + len(word_text)
                    last_word_end = w["end"]
                if char_pos >= len(committed_lower):
                    break
            trim_samples = int(last_word_end * 16000)
        else:
            # 无词级时间戳：按文本比例估算 + 0.3s安全边界
            ratio = len(committed_text) / max(len(full_text), 1)
            margin = int(0.3 * 16000)  # 额外trim减少回声
            trim_samples = int(ratio * total_samples) + margin
            # 不要过度trim：至少保留0.5秒给剩余句子
            max_trim = total_samples - int(0.5 * 16000)
            trim_samples = min(trim_samples, max(max_trim, 0))
            # 最小trim防止重识别循环
            min_trim = int(0.3 * 16000)
            if trim_samples < min_trim and trim_samples > 0:
                trim_samples = min(min_trim, total_samples // 2)

        # === 输出完整句子 ===
        actually_committed = False
        for sent in complete:
            text = sent.strip()
            if not text:
                continue
            # 短片段缓冲到下次（≤8字符的碎片）
            if self._is_short_utterance(text):
                self._interim_pending += text
                log.debug(f"Interim short utterance buffered: '{text}', pending='{self._interim_pending}'")
                continue

            # 合并之前缓冲的短片段
            if self._interim_pending:
                text = self._interim_pending + text
                self._interim_pending = ""

            self._process_segment_text(text, result["language"], asr_ms)
            actually_committed = True

        if not actually_committed:
            return False

        # 修剪VAD缓冲（去除已处理的音频）
        if trim_samples > 0:
            with self._vad_lock:
                self._vad.trim_front(trim_samples)

        # 记录已提交文本的尾部（用于下次去重）
        self._interim_committed_tail = committed_text[-50:] if len(committed_text) > 50 else committed_text

        self._interim_active = True
        log.info(f"Interim ASR: committed {len(complete)} sentence(s), trimmed {trim_samples / 16000:.2f}s")
        return True

    def _process_segment_text(self, text: str, source_lang: str, asr_ms: float = 0):
        """Output a text result (from interim or final) — similar to _process_segment but skips ASR."""
        original_text = text.strip()
        if not original_text or not any(c.isalnum() for c in original_text):
            return

        asr_lang_setting = self._panel.get_settings().get("asr_language", "auto") if self._panel else "auto"
        if asr_lang_setting != "auto" and source_lang != asr_lang_setting:
            log.info(f"Language filter: expected '{asr_lang_setting}' but got '{source_lang}', discarding: {original_text[:60]}")
            return

        self._asr_count += 1
        self._msg_id += 1
        msg_id = self._msg_id
        timestamp = datetime.now().strftime("%H:%M:%S")
        log.info(f"ASR [{source_lang}] ({asr_ms:.0f}ms, interim): {original_text}")

        if self._overlay:
            self._overlay.add_message(msg_id, timestamp, original_text, source_lang, asr_ms)

        self._last_original = original_text
        self._last_msg_id = msg_id

        # 字幕窗口额外语言
        target_lang = self._target_language
        extra_langs = set()
        if self._subwin and self._subwin.isVisible():
            subwin_langs = self._subwin.get_target_languages()
            extra_langs = subwin_langs - {target_lang, source_lang}

        if source_lang == target_lang:
            log.info(f"Same language ({source_lang}), no translation")
            if self._overlay:
                self._overlay.update_translation(msg_id, "", 0)
                self._overlay.update_stats(self._asr_count, self._translate_count, self._total_prompt_tokens, self._total_completion_tokens, self._compute_cost())
            if self._subwin and self._subwin.isVisible():
                if extra_langs:
                    try:
                        self._tl_executor.submit(self._translate_subwin_only, original_text, source_lang, extra_langs)
                    except RuntimeError:
                        pass
                else:
                    self._subwin.update_text(original_text, {target_lang: original_text})
        else:
            try:
                self._tl_executor.submit(self._translate_async, msg_id, original_text, source_lang, extra_langs or None)
            except RuntimeError:
                log.warning("Translation executor shut down, skipping")

    def _process_interim_final(self, speech_segment):
        """
        处理增量模式结束时的最终音频段

        当VAD检测到静音结束（flush）时，如果之前有增量输出，
        需要处理剩余的音频（可能包含最后一句的后半部分）
        """
        seg_len = len(speech_segment) / 16000
        log.info(f"Interim final segment: {seg_len:.1f}s")

        asr_start = time.perf_counter()
        with self._asr_lock:
            if not self._asr_ready or self._asr is None:
                return
            try:
                result = self._asr.transcribe(speech_segment)
            except Exception as e:
                log.error(f"Interim final ASR error: {e}", exc_info=True)
                return
        asr_ms = (time.perf_counter() - asr_start) * 1000

        if result is None:
            # Flush any remaining pending
            if self._interim_pending:
                text = self._interim_pending
                self._interim_pending = ""
                lang = self._panel.get_settings().get("asr_language", "auto") if self._panel else "auto"
                if lang == "auto":
                    lang = "unknown"
                self._process_segment_text(text, lang)
            return

        original_text = result["text"].strip()

        # Strip echo from previous commit's overlap
        original_text = self._strip_committed_overlap(original_text)

        # Prepend any remaining pending short utterances
        if self._interim_pending:
            original_text = self._interim_pending + original_text
            self._interim_pending = ""

        if not original_text or not any(c.isalnum() for c in original_text):
            return

        # Apply noise filter like _process_segment
        alnum_chars = sum(1 for c in original_text if c.isalnum())
        if seg_len >= 2.0 and alnum_chars <= 3:
            log.debug(f"Noise filter: {seg_len:.1f}s segment produced only '{original_text}', skipping")
            return

        self._process_segment_text(original_text, result["language"], asr_ms)

    def _capture_loop(self):
        """
        音频捕获循环（运行在后台线程）

        流程：
        1. 从WASAPI获取32ms音频块
        2. 更新音量显示
        3. VAD处理，检测语音段落
        4. 检测到完整段落 → 入ASR队列
        5. 仍在积累语音 → 检查增量ASR
        """
        # 静音块：用于VAD超时检测
        silence_chunk = np.zeros(
            int(
                self._config["audio"]["sample_rate"]
                * self._config["audio"]["chunk_duration"]
            ),
            dtype=np.float32,
        )
        while self._running:
            # 获取音频（阻塞1秒）
            item = self._audio.get_audio(timeout=1.0)
            if item is None:
                # 设备未就绪，静音块触发VAD超时检测
                if self._vad._is_speaking and not self._paused:
                    n = self._vad._get_effective_silence_limit() + 1
                    for _ in range(n):
                        with self._vad_lock:
                            seg = self._vad.process_chunk(silence_chunk)
                        if seg is not None and self._asr_ready:
                            self._enqueue_asr("vad_flush", seg)
                            break
                continue

            chunk, mic_rms = item

            # 暂停状态：跳过处理
            if self._paused:
                continue

            # 计算RMS音量并更新显示
            rms = float(np.sqrt(np.mean(chunk**2)))
            if self._overlay:
                self._overlay.update_monitor(rms, self._vad.last_confidence, mic_rms)

            # VAD处理
            with self._vad_lock:
                speech_segment = self._vad.process_chunk(chunk)

            if speech_segment is None:
                # 无语音段落：仍在积累 → 检查增量ASR
                if (self._incremental_enabled and self._asr_ready
                        and self._vad._is_speaking):
                    buf_samples = self._vad._speech_samples
                    total_dur = buf_samples / 16000
                    elapsed = (buf_samples - self._last_interim_samples) / 16000
                    now = time.perf_counter()
                    cooldown = now - self._last_interim_check_time
                    # 缓冲时长>=间隔 且 距离上次>=间隔 且 冷却>=1秒
                    if total_dur >= self._interim_interval and elapsed >= self._interim_interval and cooldown >= 1.0:
                        self._last_interim_check_time = now
                        self._enqueue_asr("interim", None)
                continue

            # 有语音段落
            if not self._asr_ready:
                log.debug("ASR not ready, dropping segment")
                continue

            self._enqueue_asr("vad_flush", speech_segment)

    def _enqueue_asr(self, seg_type: str, segment):
        """
        将ASR任务加入队列（队列满时丢弃旧任务）

        seg_type: "vad_flush"=完整段落, "interim"=增量检测
        """
        try:
            self._asr_queue.put_nowait((seg_type, segment))
        except queue.Full:
            # 队列满：丢弃最旧的任务，再尝试加入
            try:
                dropped = self._asr_queue.get_nowait()
                log.warning(f"ASR queue full, dropped {dropped[0]} segment")
            except queue.Empty:
                pass
            try:
                self._asr_queue.put_nowait((seg_type, segment))
            except queue.Full:
                log.warning("ASR queue still full after drop, skipping segment")

    def _asr_loop(self):
        """
        ASR处理循环（运行在后台线程）

        任务类型：
        - "vad_flush"：VAD检测到的完整语音段落
        - "interim"：增量ASR检测（边说边翻）
        """
        while self._running:
            try:
                item = self._asr_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:
                break  # 收到结束信号

            seg_type, segment = item

            if seg_type == "vad_flush":
                # 完整段落：增量模式已激活？→ 处理最终段；否则正常处理
                if self._interim_active:
                    self._process_interim_final(segment)
                else:
                    self._process_segment(segment)
                # 重置增量状态
                self._interim_active = False
                self._interim_pending = ""
                self._last_interim_samples = 0
                self._last_interim_check_time = 0.0
                self._interim_committed_tail = ""
            elif seg_type == "interim":
                # 增量检测：丢弃排队的重复interim任务
                self._drain_interim_duplicates()
                committed = self._do_interim_asr()
                if committed:
                    self._last_interim_samples = self._vad._speech_samples

    def _drain_interim_duplicates(self):
        """
        丢弃排队的重复interim任务

        增量检测期间可能有多个interim任务在队列中，
        只保留第一个，其余丢弃（避免重复处理）
        """
        while True:
            try:
                item = self._asr_queue.get_nowait()
            except queue.Empty:
                break
            if item is None or item[0] != "interim":
                # 非interim任务（vad_flush或结束信号）放回队列
                self._asr_queue.put(item)
                break


def main():
    """
    应用主入口

    启动流程：
    1. 配置日志
    2. 加载配置
    3. 首次启动？→ 设置向导 → 配置翻译API
    4. 非首次启动但模型缺失？→ 下载对话框
    5. 创建UI组件（LogWindow、ControlPanel、SubtitleOverlay、SubtitleWindow）
    6. 创建LiveTranslateApp并关联UI
    7. 延迟初始化（加载设置到UI）
    8. 创建系统托盘菜单
    9. 进入Qt事件循环
    """
    # === 1. 日志配置 ===
    setup_logging()
    log.info("LiveTranslate starting...")

    # === 2. 加载配置 ===
    config = load_config()
    saved = _load_saved_settings()

    # 打印实际使用的配置（方便调试）
    _asr_eng = (saved or {}).get("asr_engine", "whisper")
    _active_idx = (saved or {}).get("active_model", 0)
    _models = (saved or {}).get("models", [])
    if 0 <= _active_idx < len(_models):
        _m = _models[_active_idx]
        _model_info = f"{_m.get('name', '?')} ({_m.get('model', '?')})"
    else:
        _model_info = f"{config['translation']['model']} (default)"
    log.info(f"Config loaded: ASR={_asr_eng}, Translator={_model_info}")

    # 应用UI语言
    if saved and saved.get("ui_lang"):
        set_lang(saved["ui_lang"])

    # === 3. 创建Qt应用 ===
    os.environ["QT_LOGGING_RULES"] = "qt.text.font.db=false"
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # 关闭窗口不退出（托盘运行）
    _app_icon = create_app_icon()
    app.setWindowIcon(_app_icon)

    # === 4. 首次启动向导 ===
    if not SETTINGS_FILE.exists():
        # 显示设置向导（选择模型下载源、下载路径）
        wizard = SetupWizardDialog()
        if wizard.exec() != QDialog.DialogCode.Accepted:
            sys.exit(0)
        saved = _load_saved_settings()
        log.info("Setup wizard completed")

        # 提示用户配置翻译API
        from dialogs import ModelEditDialog

        info = QMessageBox(
            QMessageBox.Icon.Information,
            t("window_setup"),
            t("setup_api_hint"),
        )
        info.exec()

        # 弹出API配置对话框（预填了默认值）
        dlg = ModelEditDialog(None, {
            "name": "hunyuan-mt-chimera-7b",
            "api_base": "http://127.0.0.1:1234/v1",
            "api_key": "sk-lm-tHzDfNGm:dgxlip7eebn3HIMxivqN",
            "model": "hunyuan-mt-chimera-7b",
        })
        dlg.setWindowTitle(t("setup_api_title"))
        if dlg.exec() == QDialog.DialogCode.Accepted:
            data = dlg.get_data()
            if data.get("api_key"):
                saved["models"] = [data]
                saved["active_model"] = 0
                _save_settings(saved)
                log.info(f"Translation API configured: {data['name']}")

    # === 5. 非首次启动：检查模型缺失 ===
    else:
        missing = get_missing_models(
            saved.get("asr_engine", "sensevoice"),
            config["asr"]["model_size"],
            saved.get("hub", "ms"),
        )
        if missing:
            log.info(f"Missing models: {[m['name'] for m in missing]}")
            dlg = ModelDownloadDialog(missing, hub=saved.get("hub", "ms"))
            if dlg.exec() != QDialog.DialogCode.Accepted:
                sys.exit(0)

    # === 6. 创建UI组件 ===
    log_window = LogWindow()
    log_handler = log_window.get_handler()
    logging.getLogger().addHandler(log_handler)

    # 设置面板（7个标签页）
    panel = ControlPanel(config, saved_settings=saved)

    # 字幕悬浮窗
    overlay = SubtitleOverlay(config["subtitle"])
    # 恢复上次位置
    if saved:
        ox = saved.get("overlay_x")
        oy = saved.get("overlay_y")
        ow = saved.get("overlay_w")
        oh = saved.get("overlay_h")
        if ox is not None and oy is not None:
            # 检查位置是否在可见区域
            if SubtitleWindow._is_pos_visible(ox, oy):
                overlay.move(ox, oy)
            else:
                # 屏幕右下角
                screen = QApplication.primaryScreen()
                geo = screen.availableGeometry()
                overlay.move(geo.right() - overlay.width() - 20, geo.bottom() - overlay.height() - 60)
        if ow and oh:
            overlay.resize(ow, oh)
    overlay.show()

    # OBS字幕窗口
    subwin_cfg = (saved or {}).get("subtitle_mode")
    subwin = SubtitleWindow(subwin_cfg)
    subwin_was_enabled = (subwin_cfg or {}).get("enabled", False)

    # === 7. 创建核心应用并关联UI ===
    live_trans = LiveTranslateApp(config)
    live_trans.set_overlay(overlay)
    live_trans.set_subtitle_window(subwin)
    live_trans.set_panel(panel)

    # === 8. 延迟初始化（UI先显示，模型后加载） ===
    def _deferred_init():
        """在UI显示后应用所有设置，加载模型"""
        panel._apply_settings()
        models = panel.get_settings().get("models", [])
        active_idx = panel.get_settings().get("active_model", 0)
        overlay.set_models(models, active_idx)
        target = panel.get_settings().get("target_language", "zh")
        overlay.set_target_language(target)
        asr_lang = panel.get_settings().get("asr_language", "auto")
        overlay.set_source_language(asr_lang)
        style = panel.get_settings().get("style")
        if style:
            overlay.apply_style(style)
        active_model = panel.get_active_model()
        if active_model:
            live_trans._on_model_changed(active_model)

    QTimer.singleShot(100, _deferred_init)

    # === 9. 创建系统托盘 ===
    tray = QSystemTrayIcon()
    tray.setToolTip(t("tray_tooltip"))
    tray.setIcon(_app_icon)

    menu = QMenu()

    # --- 暂停/恢复 ---
    pause_action = QAction(t("tray_pause"))
    _is_running = [True]  # 用列表包装以在闭包中修改

    def on_start():
        """启动管线"""
        try:
            live_trans.start()
            overlay.set_running(True)
            _is_running[0] = True
            pause_action.setText(t("tray_pause"))
        except Exception as e:
            log.error(f"Start error: {e}", exc_info=True)

    def on_pause():
        """暂停管线"""
        live_trans.pause()
        overlay.set_running(False)
        _is_running[0] = False
        pause_action.setText(t("tray_resume"))

    def on_resume():
        """恢复管线"""
        live_trans.resume()
        overlay.set_running(True)
        _is_running[0] = True
        pause_action.setText(t("tray_pause"))

    def on_toggle_pause():
        """切换暂停/恢复"""
        if _is_running[0]:
            on_pause()
        else:
            on_resume()

    pause_action.triggered.connect(on_toggle_pause)
    menu.addAction(pause_action)
    menu.addSeparator()

    # --- 显示/隐藏字幕窗口 ---
    overlay_toggle_action = QAction(t("tray_hide_overlay"))

    _hide_notified = [False]

    def on_toggle_overlay():
        """切换字幕悬浮窗显示/隐藏"""
        if overlay.isVisible():
            overlay.hide()
            overlay_toggle_action.setText(t("tray_show_overlay"))
            # 首次隐藏时提示用户
            if not _hide_notified[0]:
                _hide_notified[0] = True
                tray.showMessage(
                    "LiveTranslate",
                    t("hide_tray_hint"),
                    QSystemTrayIcon.MessageIcon.Information,
                    3000,
                )
        else:
            overlay.show()
            overlay.raise_()
            overlay_toggle_action.setText(t("tray_hide_overlay"))

    overlay_toggle_action.triggered.connect(on_toggle_overlay)
    menu.addAction(overlay_toggle_action)

    # --- 字幕窗口切换（OBS用）---
    # 保存字幕悬浮窗位置
    def _save_overlay_pos():
        settings = panel.get_settings()
        pos = overlay.pos()
        size = overlay.size()
        settings["overlay_x"] = pos.x()
        settings["overlay_y"] = pos.y()
        settings["overlay_w"] = size.width()
        settings["overlay_h"] = size.height()
        panel._current_settings.update({
            "overlay_x": pos.x(), "overlay_y": pos.y(),
            "overlay_w": size.width(), "overlay_h": size.height(),
        })
        _save_settings(settings)

    overlay.position_changed.connect(_save_overlay_pos)

    # OBS字幕窗口开关
    subwin_toggle_action = QAction(t("subwin_show"), checkable=True)

    # 保存字幕窗口状态
    def _save_subwin_state():
        settings = panel.get_settings()
        sm = settings.get("subtitle_mode") or {}
        sm["enabled"] = subwin.isVisible()
        pos = subwin.pos()
        sm["window_x"] = pos.x()
        sm["window_y"] = pos.y()
        settings["subtitle_mode"] = sm
        panel._current_settings["subtitle_mode"] = sm
        _save_settings(settings)

    _subwin_notified = [False]

    def on_toggle_subwin(checked):
        """切换OBS字幕窗口显示"""
        if checked:
            subwin.show()
            subwin.raise_()
            # 首次显示时提示拖拽方法
            if not _subwin_notified[0]:
                _subwin_notified[0] = True
                tray.showMessage(
                    "LiveTranslate",
                    t("subwin_drag_hint"),
                    QSystemTrayIcon.MessageIcon.Information,
                    3000,
                )
        else:
            subwin.hide()
        overlay.set_subtitle_checked(checked)
        _save_subwin_state()

    subwin_toggle_action.toggled.connect(on_toggle_subwin)
    subwin.position_changed.connect(_save_subwin_state)

    # 字幕窗口手动关闭时（如Alt+F4）同步状态
    def _on_subwin_closed():
        subwin_toggle_action.blockSignals(True)
        subwin_toggle_action.setChecked(False)
        subwin_toggle_action.blockSignals(False)
        overlay.set_subtitle_checked(False)
        _save_subwin_state()

    subwin.window_closed.connect(_on_subwin_closed)

    # 恢复字幕窗口可见性状态
    if subwin_was_enabled:
        subwin_toggle_action.setChecked(True)

    menu.addAction(subwin_toggle_action)

    # overlay上的字幕按钮 → 切换字幕窗口
    def _on_overlay_subtitle_toggle():
        subwin_toggle_action.setChecked(not subwin_toggle_action.isChecked())

    overlay.subtitle_toggled.connect(_on_overlay_subtitle_toggle)

    # 控制面板字幕设置变化 → 应用到字幕窗口
    def _on_panel_subtitle_changed(s):
        subwin.apply_settings(s)

    panel.subtitle_settings_changed.connect(_on_panel_subtitle_changed)

    # 重置所有窗口位置
    def _on_reset_positions():
        screen = QApplication.primaryScreen()
        geo = screen.availableGeometry()
        subwin.move(100, 100)
        _save_subwin_state()
        ow, oh = overlay.width(), overlay.height()
        overlay.move(geo.right() - ow - 50, geo.bottom() - oh - 100)
        _save_overlay_pos()

    panel.reset_positions.connect(_on_reset_positions)

    menu.addSeparator()

    # --- 显示日志/设置面板 ---
    log_action = QAction(t("tray_show_log"))
    panel_action = QAction(t("tray_show_panel"))

    def on_toggle_log():
        """切换日志窗口显示"""
        if log_window.isVisible():
            log_window.hide()
        else:
            log_window.show()
            log_window.raise_()

    def on_toggle_panel():
        """切换设置面板显示"""
        if panel.isVisible():
            panel.hide()
        else:
            panel.show()
            panel.raise_()

    log_action.triggered.connect(on_toggle_log)
    panel_action.triggered.connect(on_toggle_panel)
    menu.addAction(panel_action)
    menu.addAction(log_action)
    menu.addSeparator()

    # --- Overlay子菜单（点击穿透、置顶、自动滚动、任务栏）---
    overlay_menu = QMenu(t("tray_menu_overlay"))

    ct_action = QAction(t("click_through"), checkable=True)  # 点击穿透
    topmost_action = QAction(t("top_most"), checkable=True)   # 置顶
    topmost_action.setChecked(True)
    autoscroll_action = QAction(t("auto_scroll"), checkable=True)  # 自动滚动
    autoscroll_action.setChecked(True)
    taskbar_action = QAction(t("taskbar"), checkable=True)  # 任务栏显示

    # 托盘 → overlay 同步
    ct_action.toggled.connect(lambda v: overlay._handle._ct_check.setChecked(v))
    topmost_action.toggled.connect(
        lambda v: overlay._handle._topmost_check.setChecked(v)
    )
    autoscroll_action.toggled.connect(
        lambda v: overlay._handle._auto_scroll.setChecked(v)
    )
    taskbar_action.toggled.connect(
        lambda v: overlay._handle._taskbar_check.setChecked(v)
    )

    # overlay → 托盘 同步（用户点击overlay上的控件时更新托盘菜单）
    overlay._handle.click_through_toggled.connect(lambda v: ct_action.setChecked(v))
    overlay._handle.topmost_toggled.connect(lambda v: topmost_action.setChecked(v))
    overlay._handle.auto_scroll_toggled.connect(
        lambda v: autoscroll_action.setChecked(v)
    )
    overlay._handle.taskbar_toggled.connect(lambda v: taskbar_action.setChecked(v))

    overlay_menu.addAction(ct_action)
    overlay_menu.addAction(topmost_action)
    overlay_menu.addAction(autoscroll_action)
    overlay_menu.addAction(taskbar_action)
    menu.addMenu(overlay_menu)

    # --- 翻译模型子菜单 ---
    model_menu = QMenu(t("tray_menu_model"))
    model_action_group = QActionGroup(model_menu)
    model_action_group.setExclusive(True)

    # 每次显示菜单前重建（动态适应模型列表变化）
    def _rebuild_model_menu():
        for a in model_action_group.actions():
            model_action_group.removeAction(a)
        model_menu.clear()
        settings = panel.get_settings()
        models = settings.get("models", [])
        active = settings.get("active_model", 0)
        for i, m in enumerate(models):
            name = m.get("name", m.get("model", "?"))
            action = QAction(name, checkable=True)
            if i == active:
                action.setChecked(True)
            model_action_group.addAction(action)
            action.triggered.connect(lambda checked, idx=i: _on_tray_model_switch(idx))
            model_menu.addAction(action)

    def _on_tray_model_switch(index):
        """托盘菜单切换翻译模型"""
        models = panel.get_settings().get("models", [])
        if 0 <= index < len(models):
            from control_panel import _save_settings

            settings = panel.get_settings()
            settings["active_model"] = index
            panel._current_settings["active_model"] = index
            _save_settings(settings)
            panel._refresh_model_list()
            live_trans._on_model_changed(models[index])
            overlay.set_models(models, index)

    def on_overlay_model_switch(index):
        models = panel.get_settings().get("models", [])
        if 0 <= index < len(models):
            from control_panel import _save_settings

            settings = panel.get_settings()
            settings["active_model"] = index
            panel._current_settings["active_model"] = index
            _save_settings(settings)
            panel._refresh_model_list()
            live_trans._on_model_changed(models[index])
        _rebuild_model_menu()

    model_menu.aboutToShow.connect(_rebuild_model_menu)
    menu.addMenu(model_menu)

    # --- 目标语言子菜单 ---
    lang_menu = QMenu(t("tray_menu_target_lang"))
    lang_action_group = QActionGroup(lang_menu)
    lang_action_group.setExclusive(True)
    _lang_actions = {}
    lang_more_menu = QMenu(t("tray_more_langs"))

    # 常用语言放主菜单，其他放"更多"子菜单
    for code, native in LANGUAGES:
        if code == "auto":
            continue
        action = QAction(f"{code} - {native}", checkable=True)
        lang_action_group.addAction(action)
        action.triggered.connect(lambda checked, lc=code: _on_tray_lang_switch(lc))
        if code in COMMON_LANG_CODES:
            lang_menu.addAction(action)
        else:
            lang_more_menu.addAction(action)
        _lang_actions[code] = action

    lang_menu.addMenu(lang_more_menu)

    # 恢复当前选中状态
    current_target = panel.get_settings().get("target_language", "zh")
    if current_target in _lang_actions:
        _lang_actions[current_target].setChecked(True)

    def _on_tray_lang_switch(lang_code):
        """托盘菜单切换目标语言"""
        overlay.set_target_language(lang_code)
        live_trans._on_target_language_changed(lang_code)
        from control_panel import _save_settings

        settings = panel.get_settings()
        settings["target_language"] = lang_code
        panel._current_settings["target_language"] = lang_code
        _save_settings(settings)

    # overlay → 托盘 同步
    def _on_overlay_lang_changed(lang_code):
        if lang_code in _lang_actions:
            _lang_actions[lang_code].setChecked(True)

    overlay.target_language_changed.connect(_on_overlay_lang_changed)

    menu.addMenu(lang_menu)

    # --- ASR语言子菜单 ---
    asr_lang_menu = QMenu(t("tray_menu_asr_lang"))
    asr_lang_action_group = QActionGroup(asr_lang_menu)
    asr_lang_action_group.setExclusive(True)
    _asr_lang_actions = {}
    asr_more_menu = QMenu(t("tray_more_langs"))

    for code, native in LANGUAGES:
        label = t("asr_lang_auto") if code == "auto" else native
        action = QAction(f"{code} - {label}", checkable=True)
        asr_lang_action_group.addAction(action)
        action.triggered.connect(lambda checked, c=code: _on_tray_asr_lang(c))
        if code in COMMON_LANG_CODES:
            asr_lang_menu.addAction(action)
        else:
            asr_more_menu.addAction(action)
        _asr_lang_actions[code] = action

    asr_lang_menu.addMenu(asr_more_menu)

    # 恢复当前选中状态
    current_asr_lang = panel.get_settings().get("asr_language", "auto")
    if current_asr_lang in _asr_lang_actions:
        _asr_lang_actions[current_asr_lang].setChecked(True)

    def _on_tray_asr_lang(code):
        """托盘菜单切换ASR语言"""
        from control_panel import _save_settings

        if live_trans._asr:
            live_trans._asr.set_language(code)
        settings = panel.get_settings()
        settings["asr_language"] = code
        panel._current_settings["asr_language"] = code
        _save_settings(settings)
        # 同步控制面板的ASR语言下拉框
        idx = panel._asr_lang.findData(code)
        if idx >= 0:
            panel._asr_lang.blockSignals(True)
            panel._asr_lang.setCurrentIndex(idx)
            panel._asr_lang.blockSignals(False)

    menu.addMenu(asr_lang_menu)
    menu.addSeparator()

    # --- 退出 ---
    quit_action = QAction(t("quit"))

    def on_quit():
        """退出应用：停止管线，退出Qt事件循环"""
        live_trans.stop()
        app.quit()

    quit_action.triggered.connect(on_quit)
    menu.addAction(quit_action)

    # --- 连接overlay信号 ---
    # overlay上的按钮 → 对应操作
    overlay.settings_requested.connect(on_toggle_panel)  # 设置按钮
    overlay.target_language_changed.connect(live_trans._on_target_language_changed)  # 目标语言变化

    # overlay源语言变化 → 同步到托盘 + ASR引擎
    def _on_overlay_source_lang(code):
        _on_tray_asr_lang(code)
        overlay.set_source_language(code)

    # 控制面板ASR语言变化 → 同步到overlay
    def _on_panel_asr_lang_changed(_index):
        code = panel._asr_lang.currentData() or "auto"
        overlay.set_source_language(code)

    overlay.source_language_changed.connect(_on_overlay_source_lang)
    panel._asr_lang.currentIndexChanged.connect(_on_panel_asr_lang_changed)
    overlay.model_switch_requested.connect(on_overlay_model_switch)  # 模型切换
    overlay.start_requested.connect(on_resume)   # 开始
    overlay.stop_requested.connect(on_pause)    # 暂停
    overlay.hide_requested.connect(on_toggle_overlay)  # 隐藏
    overlay.quit_requested.connect(on_quit)      # 退出

    # === 10. 显示托盘 ===
    tray.setContextMenu(menu)
    tray.show()

    # === 11. 延迟启动管线 ===
    QTimer.singleShot(500, on_start)

    # === 12. 信号处理 ===
    # Ctrl+C 优雅退出
    signal.signal(signal.SIGINT, lambda *_: on_quit())
    # 保持事件循环运行（防止空闲退出）
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(200)

    # === 13. 进入Qt事件循环 ===
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
