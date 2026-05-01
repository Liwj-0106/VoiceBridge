import logging
import threading
import queue
import time
import numpy as np
import pyaudiowpatch as pyaudio

# 音频捕获模块日志记录器
# 负责 WASAPI loopback 系统音频捕获的实现
log = logging.getLogger("LiveTranslate.Audio")

# 设备检查间隔：每2秒检查一次系统默认输出设备是否变更
DEVICE_CHECK_INTERVAL = 2.0  # seconds


def list_output_devices():
    """返回 WASAPI 输出设备名称列表（用于设置面板选择）"""
    pa = pyaudio.PyAudio()
    devices = []
    wasapi_idx = None
    # 查找 WASAPI Host API
    for i in range(pa.get_host_api_count()):
        info = pa.get_host_api_info_by_index(i)
        if "WASAPI" in info["name"]:
            wasapi_idx = info["index"]
            break
    if wasapi_idx is not None:
        # 遍历所有设备，筛选 WASAPI 输出设备（排除 loopback）
        for i in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(i)
            if (
                dev["hostApi"] == wasapi_idx
                and dev["maxOutputChannels"] > 0
                and not dev.get("isLoopbackDevice", False)
            ):
                devices.append(dev["name"])
    pa.terminate()
    return devices


def list_input_devices():
    """返回 WASAPI 输入设备名称列表（麦克风设备，用于设置面板选择）"""
    pa = pyaudio.PyAudio()
    devices = []
    wasapi_idx = None
    # 查找 WASAPI Host API
    for i in range(pa.get_host_api_count()):
        info = pa.get_host_api_info_by_index(i)
        if "WASAPI" in info["name"]:
            wasapi_idx = info["index"]
            break
    if wasapi_idx is not None:
        # 遍历所有设备，筛选 WASAPI 输入设备（排除 loopback）
        for i in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(i)
            if (
                dev["hostApi"] == wasapi_idx
                and dev["maxInputChannels"] > 0
                and not dev.get("isLoopbackDevice", False)
            ):
                devices.append(dev["name"])
    pa.terminate()
    return devices


class AudioCapture:
    """
    WASAPI Loopback 系统音频捕获类

    功能：
    1. 通过 WASAPI Loopback 捕获系统音频（播放的音频）
    2. 支持可选的麦克风输入混合
    3. 自动重采样到 16kHz 单声道（ASR 标准采样率）
    4. 自动检测系统默认输出设备变更并重连

    音频流程：
    系统音频(44.1/48kHz stereo) → WASAPI Loopback → 重采样(16kHz mono) → 队列 → VAD/ASR
    """
    # 初始化参数:
    # - device: 目标设备名，None=系统默认，'__disabled__'=仅麦克风模式
    # - sample_rate: 输出采样率，固定 16000Hz（ASR 标准）
    # - chunk_duration: 每次读取的音频块时长（秒），默认 0.5s
    def __init__(self, device=None, sample_rate=16000, chunk_duration=0.5):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.audio_queue = queue.Queue(maxsize=100)  # 音频块队列
        self._stream = None                          # WASAPI Loopback 流
        self._running = False                        # 运行状态标志
        self._device_name = device                   # 目标设备名
        self._pa = pyaudio.PyAudio()                 # PyAudio 实例
        self._read_thread = None                     # 读取线程
        self._native_channels = 2                    # 原始通道数（通常 2）
        self._native_rate = 44100                    # 原始采样率（通常 44100 或 48000）
        self._current_device_name = None             # 当前捕获的实际设备名
        self._loopback_disabled = False              # 是否禁用 Loopback（仅麦克风模式）
        self._lock = threading.Lock()                # 线程安全锁
        self._restart_event = threading.Event()      # 设备重启事件

        # 麦克风输入相关
        self._mic_device_name = None                 # 麦克风设备名
        self._mic_stream = None                      # 麦克风流
        self._mic_native_rate = 44100                # 麦克风原始采样率
        self._mic_native_channels = 1                # 麦克风原始通道数
        self._mic_restart_event = threading.Event()  # 麦克风设备变更事件
        self._mic_buf = np.array([], dtype=np.float32)  # 麦克风音频缓冲区

    def _get_wasapi_info(self):
        """获取 WASAPI Host API 信息"""
        for i in range(self._pa.get_host_api_count()):
            info = self._pa.get_host_api_info_by_index(i)
            if "WASAPI" in info["name"]:
                return info
        return None

    def _get_default_output_name(self):
        """获取系统默认输出设备名称"""
        wasapi_info = self._get_wasapi_info()
        if wasapi_info is None:
            return None
        default_idx = wasapi_info["defaultOutputDevice"]
        default_dev = self._pa.get_device_info_by_index(default_idx)
        return default_dev["name"]

    @staticmethod
    def _query_current_default():
        """创建新的 PyAudio 实例获取当前系统默认输出设备"""
        pa = pyaudio.PyAudio()
        try:
            for i in range(pa.get_host_api_count()):
                info = pa.get_host_api_info_by_index(i)
                if "WASAPI" in info["name"]:
                    default_idx = info["defaultOutputDevice"]
                    dev = pa.get_device_info_by_index(default_idx)
                    return dev["name"]
        finally:
            pa.terminate()
        return None

    def _find_loopback_device(self):
        """
        查找 WASAPI Loopback 设备

        WASAPI Loopback 是一种特殊的音频设备，可以捕获系统播放的音频。
        流程：
        1. 获取系统默认输出设备
        2. 在所有 loopback 设备中查找匹配默认输出设备的
        3. 如果没找到匹配，返回任意 loopback 设备作为回退
        """
        wasapi_info = self._get_wasapi_info()
        if wasapi_info is None:
            raise RuntimeError("WASAPI host API not found")

        default_output_idx = wasapi_info["defaultOutputDevice"]
        default_output = self._pa.get_device_info_by_index(default_output_idx)
        log.info(f"Default output: {default_output['name']}")

        # 目标设备名：用户指定 > 系统默认
        target_name = self._device_name or default_output["name"]

        # 查找与目标设备名匹配的 loopback 设备
        for i in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(i)
            if dev["hostApi"] == wasapi_info["index"] and dev.get(
                "isLoopbackDevice", False
            ):
                if target_name in dev["name"]:
                    return dev

        # 回退：返回任意 loopback 设备
        for i in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice", False):
                return dev

        raise RuntimeError("No WASAPI loopback device found")

    def _open_stream(self):
        """
        打开 WASAPI Loopback 音频流

        初始化流程：
        1. 查找合适的 loopback 设备
        2. 记录设备的原生采样率和通道数
        3. 打开音频流并配置重采样参数
        """
        loopback_dev = self._find_loopback_device()
        self._native_channels = loopback_dev["maxInputChannels"]  # 通常是 2（立体声）
        self._native_rate = int(loopback_dev["defaultSampleRate"])  # 通常是 44100 或 48000
        self._current_device_name = loopback_dev["name"]

        log.info(f"Loopback device: {loopback_dev['name']}")
        log.info(
            f"Native: {self._native_rate}Hz, {self._native_channels}ch -> {self.sample_rate}Hz mono"
        )

        # 计算每次读取的采样点数（基于原生采样率）
        native_chunk = int(self._native_rate * self.chunk_duration)

        # 打开 WASAPI Loopback 流
        # format=pyaudio.paFloat32: 32位浮点数格式
        # input=True: 打开为输入流（捕获）
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self._native_channels,
            rate=self._native_rate,
            input=True,
            input_device_index=loopback_dev["index"],
            frames_per_buffer=native_chunk,
        )

    def _close_stream(self):
        """关闭 WASAPI Loopback 音频流"""
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _find_mic_device(self):
        """
        查找麦克风输入设备

        支持两种模式：
        1. 指定设备名：精确匹配设备
        2. '__default__' 或 'default'：使用系统默认输入设备
        """
        wasapi_info = self._get_wasapi_info()
        if wasapi_info is None:
            raise RuntimeError("WASAPI host API not found")
        # 使用系统默认输入设备
        if self._mic_device_name in ("__default__", "default"):
            default_idx = wasapi_info["defaultInputDevice"]
            dev = self._pa.get_device_info_by_index(default_idx)
            if dev["maxInputChannels"] > 0:
                return dev
            raise RuntimeError("Default input device has no input channels")
        # 精确匹配设备名
        for i in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(i)
            if (
                dev["hostApi"] == wasapi_info["index"]
                and dev["maxInputChannels"] > 0
                and not dev.get("isLoopbackDevice", False)
                and dev["name"] == self._mic_device_name
            ):
                return dev
        raise RuntimeError(f"Mic device not found: {self._mic_device_name}")

    def _open_mic_stream(self):
        """打开麦克风输入流"""
        dev = self._find_mic_device()
        self._mic_native_channels = dev["maxInputChannels"]
        self._mic_native_rate = int(dev["defaultSampleRate"])
        native_chunk = int(self._mic_native_rate * self.chunk_duration)
        log.info(
            f"Mic device: {dev['name']} ({self._mic_native_rate}Hz, {self._mic_native_channels}ch)"
        )
        self._mic_stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self._mic_native_channels,
            rate=self._mic_native_rate,
            input=True,
            input_device_index=dev["index"],
            frames_per_buffer=native_chunk,
        )

    def _close_mic_stream(self):
        """关闭麦克风输入流"""
        if self._mic_stream:
            try:
                self._mic_stream.stop_stream()
                self._mic_stream.close()
            except Exception:
                pass
            self._mic_stream = None

    def set_mic_device(self, device_name):
        """
        运行时更改麦克风设备

        device_name: None = 禁用麦克风，其他 = 设备名
        """
        if device_name == self._mic_device_name:
            return
        log.info(f"Mic device changed: {self._mic_device_name} -> {device_name}")
        self._mic_device_name = device_name
        if self._running:
            self._mic_restart_event.set()

    def set_device(self, device_name):
        """
        运行时更改音频捕获设备

        device_name:
        - None: 使用系统默认输出设备
        - '__disabled__': 禁用 Loopback，仅使用麦克风
        - 其他字符串: 指定的设备名
        """
        if device_name == self._device_name:
            return
        log.info(f"Audio device changed: {self._device_name} -> {device_name}")
        self._device_name = device_name
        self._loopback_disabled = device_name == "__disabled__"
        if self._running:
            self._restart_event.set()

    def _resample_to_mono(self, data, native_channels, native_rate):
        """
        音频重采样：将任意采样率和通道数的音频转换为 16kHz 单声道

        处理步骤：
        1. 字节数据转换为 numpy float32 数组
        2. 如果是多通道，取平均值合并为单声道
        3. 如果采样率不是 16kHz，进行线性插值重采样

        线性插值原理：
        - 计算输出/输入采样率比值 ratio
        - 对输出样本的每个位置，计算对应的输入样本位置（浮点数）
        - 使用 floor 和 ceil 位置的值进行线性插值
        """
        # 将字节数据转换为 float32 数组
        audio = np.frombuffer(data, dtype=np.float32)

        # 通道转换：多声道 -> 单声道（取平均值）
        if native_channels > 1:
            audio = audio.reshape(-1, native_channels).mean(axis=1)

        # 采样率转换：使用线性插值
        if native_rate != self.sample_rate:
            ratio = self.sample_rate / native_rate  # 例如 16000/44100 ≈ 0.363
            n_out = int(len(audio) * ratio)  # 输出样本数
            # 计算每个输出样本对应的输入位置（浮点数）
            indices = np.arange(n_out) / ratio  # 例如 [0, 2.756, 5.512, ...]
            indices = np.clip(indices, 0, len(audio) - 1)  # 边界限制
            # 整数部分和分数部分
            idx_floor = indices.astype(np.int64)
            idx_ceil = np.minimum(idx_floor + 1, len(audio) - 1)
            frac = (indices - idx_floor).astype(np.float32)  # 插值权重
            # 线性插值：audio[idx_floor] * (1-frac) + audio[idx_ceil] * frac
            audio = audio[idx_floor] * (1 - frac) + audio[idx_ceil] * frac
        return audio

    def _restart_stream(self):
        """
        重启音频流（设备变更后调用）

        流程：
        1. 关闭当前流和 PyAudio 实例
        2. 创建新的 PyAudio 实例
        3. 重新打开 Loopback 流（如果有）
        4. 重新打开麦克风流（如果有）
        """
        with self._lock:
            self._close_stream()
            self._pa.terminate()
            self._pa = pyaudio.PyAudio()
            if not self._loopback_disabled:
                self._open_stream()
            else:
                log.info("Loopback disabled (mic-only mode)")
            # 重新打开麦克风（如果启用）
            if self._mic_device_name:
                self._close_mic_stream()
                try:
                    self._open_mic_stream()
                except Exception as e:
                    log.warning(f"Failed to re-open mic after restart: {e}")

    def _read_loop(self):
        """
        后台音频读取循环（在独立线程中运行）

        主要职责：
        1. 检查设备变更事件（Loopback 和麦克风）
        2. 定期检查系统默认输出设备是否变更
        3. 读取 Loopback 音频数据或生成静音
        4. 读取麦克风数据并与 Loopback 混合
        5. 将混合后的音频放入队列供后续处理

        队列数据格式：(audio_array, mic_rms)
        - audio_array: float32 numpy 数组，16kHz 单声道
        - mic_rms: 麦克风音频的 RMS 能量值（用于音量监测）
        """
        last_device_check = time.monotonic()

        while self._running:
            # ===== 1. 处理 Loopback 设备变更 =====
            if self._restart_event.is_set():
                self._restart_event.clear()
                try:
                    self._restart_stream()
                    # 清空队列中的旧数据
                    while not self.audio_queue.empty():
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            break
                    log.info(f"Audio capture restarted on: {self._current_device_name}")
                except Exception as e:
                    log.error(f"Restart after device change failed: {e}")
                    time.sleep(0.5)
                continue

            # ===== 2. 处理麦克风设备变更 =====
            if self._mic_restart_event.is_set():
                self._mic_restart_event.clear()
                self._close_mic_stream()
                self._mic_buf = np.array([], dtype=np.float32)
                if self._mic_device_name:
                    try:
                        self._open_mic_stream()
                        log.info(f"Mic stream opened: {self._mic_device_name}")
                    except Exception as e:
                        log.error(f"Failed to open mic: {e}")
                else:
                    log.info("Mic disabled")

            # ===== 3. 定期检查系统默认设备变更 =====
            now = time.monotonic()
            if now - last_device_check > DEVICE_CHECK_INTERVAL:
                last_device_check = now
                # 仅当使用系统默认设备时才检查
                if self._device_name is None:
                    try:
                        current_default = self._query_current_default()
                        if (
                            current_default
                            and self._current_device_name
                            and current_default not in self._current_device_name
                        ):
                            log.info(
                                f"System default output changed: "
                                f"{self._current_device_name} -> {current_default}"
                            )
                            log.info("Restarting audio capture for new device...")
                            self._restart_stream()
                            log.info(
                                f"Audio capture restarted on: {self._current_device_name}"
                            )
                    except Exception as e:
                        log.warning(f"Device check error: {e}")

            # ===== 4. 读取 Loopback 音频或生成静音 =====
            loopback_audio = None
            if self._loopback_disabled:
                # 仅麦克风模式：生成静音块
                time.sleep(self.chunk_duration)
                n_samples = int(self.sample_rate * self.chunk_duration)
                loopback_audio = np.zeros(n_samples, dtype=np.float32)
            else:
                native_chunk = int(self._native_rate * self.chunk_duration)
                try:
                    data = None
                    with self._lock:
                        if not self._stream:
                            time.sleep(0.005)
                            continue
                        # 检查是否有足够数据可读
                        if self._stream.get_read_available() >= native_chunk:
                            data = self._stream.read(
                                native_chunk, exception_on_overflow=False
                            )
                    if data is not None:
                        loopback_audio = self._resample_to_mono(
                            data, self._native_channels, self._native_rate
                        )
                except Exception as e:
                    if self._restart_event.is_set():
                        continue
                    log.warning(f"Read error (device may have changed): {e}")
                    try:
                        time.sleep(0.5)
                        self._restart_stream()
                        log.info("Stream restarted after read error")
                    except Exception as re:
                        log.error(f"Restart failed: {re}")
                        time.sleep(1)
                    continue

            # ===== 5. 读取麦克风数据到缓冲区 =====
            if self._mic_stream:
                try:
                    avail = self._mic_stream.get_read_available()
                    if avail > 0:
                        mic_data = self._mic_stream.read(
                            avail, exception_on_overflow=False
                        )
                        mic_16k = self._resample_to_mono(
                            mic_data, self._mic_native_channels, self._mic_native_rate
                        )
                        self._mic_buf = np.concatenate([self._mic_buf, mic_16k])
                except Exception as e:
                    log.warning(f"Mic read error: {e}")

            if loopback_audio is None:
                time.sleep(0.005)
                continue

            # ===== 6. 混合 Loopback 和麦克风音频 =====
            audio = loopback_audio
            mic_rms = None
            if len(self._mic_buf) > 0:
                n = len(loopback_audio)
                if len(self._mic_buf) >= n:
                    # 麦克风缓冲区有足够数据
                    mic_chunk = self._mic_buf[:n]
                    self._mic_buf = self._mic_buf[n:]
                else:
                    # 麦克风缓冲区数据不足，填充剩余部分为 0
                    mic_chunk = np.zeros(n, dtype=np.float32)
                    mic_chunk[: len(self._mic_buf)] = self._mic_buf
                    self._mic_buf = np.array([], dtype=np.float32)
                # 计算麦克风的 RMS 能量值
                mic_rms = float(np.sqrt(np.mean(mic_chunk**2)))
                # 混合：Loopback + 麦克风
                audio = loopback_audio + mic_chunk

            # ===== 7. 放入队列供后续处理 =====
            try:
                self.audio_queue.put_nowait((audio, mic_rms))
            except queue.Full:
                # 队列满时，丢弃旧数据，放入新数据
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait((audio, mic_rms))

    def start(self):
        """
        启动音频捕获

        流程：
        1. 判断是否禁用 Loopback
        2. 打开 Loopback 流（或标记仅麦克风模式）
        3. 打开麦克风流（如果已配置）
        4. 启动后台读取线程
        """
        self._loopback_disabled = self._device_name == "__disabled__"
        if not self._loopback_disabled:
            self._open_stream()
        else:
            log.info("Loopback disabled (mic-only mode)")
        if self._mic_device_name:
            try:
                self._open_mic_stream()
            except Exception as e:
                log.warning(f"Failed to open mic on start: {e}")
        self._running = True
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()
        log.info("Audio capture started")

    def stop(self):
        """停止音频捕获并清理资源"""
        self._running = False
        if self._read_thread:
            self._read_thread.join(timeout=3)
        self._close_stream()
        self._close_mic_stream()
        log.info("Audio capture stopped")

    def get_audio(self, timeout=1.0):
        """
        从队列获取一个音频块

        返回格式：(audio_array, mic_rms)
        - audio_array: float32 numpy 数组，16kHz 单声道
        - mic_rms: 麦克风 RMS 能量值（可能为 None）

        超时返回 None
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def __del__(self):
        """析构函数：确保 PyAudio 资源被释放"""
        if self._pa:
            self._pa.terminate()
