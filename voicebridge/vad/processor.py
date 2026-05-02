import logging
import collections

import numpy as np
import torch

# 设置 PyTorch 线程数为1，避免占用过多CPU资源
torch.set_num_threads(1)

# VAD模块日志记录器
log = logging.getLogger("LiveTranslate.VAD")


class VADProcessor:
    """
    语音活动检测处理器（Voice Activity Detection）

    支持三种模式：
    1. silero - 使用 Silero VAD 深度学习模型（推荐，精度最高）
    2. energy - 基于能量阈值的简单检测（轻量级备选）
    3. disabled - 禁用VAD，所有音频都视为语音

    核心功能：
    - 检测语音起止时刻
    - 渐进式静音检测：缓冲区越长，对静音越敏感（更早切分）
    - 自适应静音阈值：根据历史停顿自动调整
    - 回溯切分：最大时长时在置信度最低点切分
    - 前置缓冲：捕获语音开头的前导辅音
    """

    def __init__(
        self,
        sample_rate=16000,
        threshold=0.50,
        min_speech_duration=1.0,
        max_speech_duration=15.0,
        chunk_duration=0.032,
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold  # Silero置信度阈值
        self.energy_threshold = 0.02  # 能量检测阈值
        # 最小/最大语音样本数
        self.min_speech_samples = int(min_speech_duration * sample_rate)
        self.max_speech_samples = int(max_speech_duration * sample_rate)
        self._chunk_duration = chunk_duration  # 32ms块大小
        self.mode = "silero"  # 检测模式

        # 加载 Silero VAD 模型
        self._model, self._utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._model.eval()

        # 语音缓冲区：存储待确认的音频块
        self._speech_buffer = []
        # 置信度历史：与缓冲区同步记录每个块的置信度
        self._confidence_history = []
        self._speech_samples = 0  # 当前缓冲区样本数
        self._is_speaking = False  # 是否正在语音中
        self._silence_counter = 0  # 连续静音块计数
        self._was_trimmed = False  # 标记是否被trim_front修剪过（增量ASR用）

        # 前置缓冲：捕获VAD触发前的语音起始部分（~96ms）
        self._pre_speech_chunks = 3
        self._pre_buffer = collections.deque(maxlen=self._pre_speech_chunks)

        # 静音检测配置
        self._silence_mode = "auto"  # "auto"=自适应, "fixed"=固定
        self._fixed_silence_dur = 0.8  # 固定静音时长（秒）
        self._silence_limit = self._seconds_to_chunks(0.8)  # 静音块数阈值

        # 渐进式静音：缓冲区越长，接受越短的停顿作为切分点
        self._progressive_tiers = [
            # (缓冲区秒数, 静音限制倍数)
            (3.0, 1.0),   # < 3s: 使用完整静音限制
            (6.0, 0.5),   # 3-6s: 使用一半静音限制
            (10.0, 0.25), # 6-10s: 使用1/4静音限制
        ]

        # 自适应静音：跟踪最近停顿时长，动态调整阈值
        self._pause_history = collections.deque(maxlen=50)  # 最近50次停顿
        self._adaptive_min = 0.3  # 最小静音阈值（秒）
        self._adaptive_max = 2.0  # 最大静音阈值（秒）

        # 对外暴露：供MonitorBar显示
        self.last_confidence = 0.0

    def _seconds_to_chunks(self, seconds: float) -> int:
        """秒数转换为块数量（向上取整，最小为1）"""
        return max(1, round(seconds / self._chunk_duration))

    def _update_adaptive_limit(self):
        """
        更新自适应静音限制

        根据历史停顿时长的P75值来调整静音阈值
        这样可以适应不同说话者的停顿习惯
        """
        if len(self._pause_history) < 3:
            return
        pauses = sorted(self._pause_history)
        # 取P75分位数 × 1.2 作为目标阈值
        idx = int(len(pauses) * 0.75)
        p75 = pauses[min(idx, len(pauses) - 1)]
        target = max(self._adaptive_min, min(self._adaptive_max, p75 * 1.2))
        new_limit = self._seconds_to_chunks(target)
        if new_limit != self._silence_limit:
            log.debug(
                f"Adaptive silence: {target:.2f}s ({new_limit} chunks), P75={p75:.2f}s"
            )
            self._silence_limit = new_limit

    def update_settings(self, settings: dict):
        """从设置字典更新VAD参数"""
        if "vad_mode" in settings:
            self.mode = settings["vad_mode"]
        if "vad_threshold" in settings:
            self.threshold = settings["vad_threshold"]
        if "energy_threshold" in settings:
            self.energy_threshold = settings["energy_threshold"]
        if "min_speech_duration" in settings:
            self.min_speech_samples = int(
                settings["min_speech_duration"] * self.sample_rate
            )
        if "max_speech_duration" in settings:
            self.max_speech_samples = int(
                settings["max_speech_duration"] * self.sample_rate
            )
        if "silence_mode" in settings:
            self._silence_mode = settings["silence_mode"]
        if "silence_duration" in settings:
            self._fixed_silence_dur = settings["silence_duration"]
            if self._silence_mode == "fixed":
                self._silence_limit = self._seconds_to_chunks(self._fixed_silence_dur)
        log.info(
            f"VAD settings updated: mode={self.mode}, threshold={self.threshold}, "
            f"silence={self._silence_mode} "
            f"({self._silence_limit} chunks = {self._silence_limit * self._chunk_duration:.2f}s)"
        )

    def _silero_confidence(self, audio_chunk: np.ndarray) -> float:
        """使用 Silero VAD 模型获取置信度"""
        # Silero 窗口大小：16kHz采样率时为512样本（32ms）
        window_size = 512 if self.sample_rate == 16000 else 256
        chunk = audio_chunk[:window_size]
        if len(chunk) < window_size:
            chunk = np.pad(chunk, (0, window_size - len(chunk)))
        tensor = torch.from_numpy(chunk).float()
        return self._model(tensor, self.sample_rate).item()

    def _energy_confidence(self, audio_chunk: np.ndarray) -> float:
        """基于能量的置信度计算（简单模式用）"""
        rms = float(np.sqrt(np.mean(audio_chunk**2)))
        return min(1.0, rms / (self.energy_threshold * 2))

    def _get_confidence(self, audio_chunk: np.ndarray) -> float:
        """根据当前模式获取置信度"""
        if self.mode == "silero":
            return self._silero_confidence(audio_chunk)
        elif self.mode == "energy":
            return self._energy_confidence(audio_chunk)
        else:  # disabled
            return 1.0

    def _get_effective_silence_limit(self) -> int:
        """
        获取有效的静音限制

        渐进式静音机制：缓冲区越长，越短的停顿就能触发切分
        这样可以在用户短暂停顿时就输出翻译，降低延迟
        """
        buf_seconds = self._speech_samples / self.sample_rate
        multiplier = 1.0
        for tier_sec, tier_mult in self._progressive_tiers:
            if buf_seconds < tier_sec:
                break
            multiplier = tier_mult
        effective = max(1, round(self._silence_limit * multiplier))
        return effective

    def process_chunk(self, audio_chunk: np.ndarray):
        """
        处理一个音频块

        核心状态机：
        1. 置信度 >= 阈值 → 正在说话，添加到缓冲区
        2. 置信度 < 阈值 且 正在说话 → 静音计数增加
        3. 静音达到阈值 → 触发 flush 输出段落
        4. 置信度 < 阈值 且 未在说话 → 添加到前置缓冲

        返回值：
        - 有语音段落可输出时返回 numpy 数组
        - 无段落时返回 None
        """
        confidence = self._get_confidence(audio_chunk)
        self.last_confidence = confidence

        effective_threshold = self.threshold if self.mode == "silero" else 0.5
        eff_silence_limit = self._get_effective_silence_limit()

        # ====== 检测到语音 ======
        if confidence >= effective_threshold:
            # 记录停顿时长（用于自适应调整）
            if self._is_speaking and self._silence_counter > 0:
                pause_dur = self._silence_counter * self._chunk_duration
                if pause_dur >= 0.1:
                    self._pause_history.append(pause_dur)
                    if self._silence_mode == "auto":
                        self._update_adaptive_limit()

            # 语音开始：补充前置缓冲区的内容（捕获前导辅音）
            if not self._is_speaking:
                for pre_chunk in self._pre_buffer:
                    self._speech_buffer.append(pre_chunk)
                    self._confidence_history.append(effective_threshold)
                    self._speech_samples += len(pre_chunk)
                self._pre_buffer.clear()

            self._is_speaking = True
            self._silence_counter = 0
            self._speech_buffer.append(audio_chunk)
            self._confidence_history.append(confidence)
            self._speech_samples += len(audio_chunk)

        # ====== 静音中 ======
        elif self._is_speaking:
            self._silence_counter += 1
            self._speech_buffer.append(audio_chunk)
            self._confidence_history.append(confidence)
            self._speech_samples += len(audio_chunk)

        # ====== 非语音状态 ======
        else:
            # 添加到前置缓冲，捕获语音开头
            self._pre_buffer.append(audio_chunk)

        # ====== 最大时长检查 → 回溯切分 ======
        if self._speech_samples >= self.max_speech_samples:
            return self._split_at_best_pause()

        # ====== 静音超时 → 输出段落 ======
        if self._is_speaking and self._silence_counter >= eff_silence_limit:
            if self._speech_samples >= self.min_speech_samples:
                return self._flush_segment()
            elif self._was_trimmed:
                # 增量ASR已修剪过缓冲区，强制输出剩余部分
                log.debug(
                    f"Short segment after trim ({self._speech_samples / self.sample_rate:.1f}s), "
                    f"force flushing for interim final"
                )
                return self.force_flush()
            else:
                # 太短：软重置，保持缓冲与下一段语音合并
                log.debug(
                    f"Short segment {self._speech_samples / self.sample_rate:.1f}s "
                    f"< min {self.min_speech_samples / self.sample_rate:.1f}s, "
                    f"keeping for merge"
                )
                self._is_speaking = False
                self._silence_counter = 0
                return None

        return None

    def _find_best_split_index(self) -> int:
        """
        在缓冲区中寻找最佳切分点

        算法：
        1. 对置信度历史进行滑动平均平滑（减少单块噪声影响）
        2. 在后70%的缓冲区中查找平滑曲线的最低谷
        3. 检查这个谷是否有意义（低于阈值或显著低于平均）

        返回值：最佳切分点的块索引，-1表示找不到合适切分点
        """
        n = len(self._confidence_history)
        if n < 4:
            return -1

        # 滑动窗口平滑：~160ms（5个块，每块32ms）
        smooth_win = min(5, n // 2)
        smoothed = []
        for i in range(n):
            lo = max(0, i - smooth_win // 2)
            hi = min(n, i + smooth_win // 2 + 1)
            smoothed.append(sum(self._confidence_history[lo:hi]) / (hi - lo))

        # 在后70%的缓冲区中搜索（避免过早切分）
        search_start = max(1, n * 3 // 10)

        # 找全局最低点
        min_val = float("inf")
        min_idx = -1
        for i in range(search_start, n):
            if smoothed[i] <= min_val:
                min_val = smoothed[i]
                min_idx = i

        if min_idx <= 0:
            return -1

        # 检查这个低点是否有意义
        avg_conf = sum(smoothed[search_start:]) / max(1, n - search_start)
        dip_ratio = min_val / max(avg_conf, 1e-6)

        effective_threshold = self.threshold if self.mode == "silero" else 0.5
        if min_val < effective_threshold or dip_ratio < 0.8:
            log.debug(
                f"Split point at chunk {min_idx}/{n}: "
                f"smoothed={min_val:.3f}, avg={avg_conf:.3f}, dip_ratio={dip_ratio:.2f}"
            )
            return min_idx

        # 回退：任何低于平均值的地方都比硬切好
        if min_val < avg_conf:
            log.debug(
                f"Split point (fallback) at chunk {min_idx}/{n}: "
                f"smoothed={min_val:.3f}, avg={avg_conf:.3f}"
            )
            return min_idx

        return -1

    def _split_at_best_pause(self):
        """
        最大时长切分

        当缓冲区达到最大时长限制时：
        1. 在缓冲区中寻找最佳切分点（置信度最低处）
        2. 切分后的第一部分输出
        3. 剩余部分保留在缓冲区继续累积

        这样可以确保长语音不会无限累积，同时在自然停顿点切分
        """
        if not self._speech_buffer:
            return None

        split_idx = self._find_best_split_index()

        if split_idx <= 0:
            # 找不到好的切分点，硬切全部输出
            log.info(
                f"Max duration reached, no good split point, "
                f"hard flush {self._speech_samples / self.sample_rate:.1f}s"
            )
            return self._flush_segment()

        # 切分：第一部分输出，剩余部分保留
        first_bufs = self._speech_buffer[:split_idx]
        remain_bufs = self._speech_buffer[split_idx:]
        remain_confs = self._confidence_history[split_idx:]

        first_samples = sum(len(b) for b in first_bufs)
        remain_samples = sum(len(b) for b in remain_bufs)

        log.info(
            f"Max duration split at {first_samples / self.sample_rate:.1f}s, "
            f"keeping {remain_samples / self.sample_rate:.1f}s remainder"
        )

        segment = np.concatenate(first_bufs)

        # 保留剩余部分继续累积
        self._speech_buffer = remain_bufs
        self._confidence_history = remain_confs
        self._speech_samples = remain_samples
        self._is_speaking = True
        self._silence_counter = 0

        return segment

    def _flush_segment(self):
        """
        输出语音段落

        输出前检查语音密度：
        - 如果超过75%的块置信度低于阈值，说明可能是噪声，丢弃

        检查通过后，将缓冲区中的所有块拼接成一个连续的numpy数组输出
        """
        if not self._speech_buffer:
            return None
        # 语音密度检查：丢弃大部分块都不像语音的段落
        if len(self._confidence_history) >= 4:
            effective_threshold = self.threshold if self.mode == "silero" else 0.5
            voiced = sum(
                1 for c in self._confidence_history if c >= effective_threshold
            )
            density = voiced / len(self._confidence_history)
            if density < 0.25:
                dur = self._speech_samples / self.sample_rate
                log.debug(
                    f"Low speech density {density:.0%} ({voiced}/{len(self._confidence_history)}), "
                    f"discarding {dur:.1f}s segment"
                )
                self._reset()
                return None
        segment = np.concatenate(self._speech_buffer)
        self._reset()
        return segment

    def _reset(self):
        """重置VAD状态（输出段落后调用）"""
        self._speech_buffer = []
        self._confidence_history = []
        self._speech_samples = 0
        self._is_speaking = False
        self._silence_counter = 0
        self._was_trimmed = False

    def peek_buffer(self):
        """
        查看当前缓冲区（不刷新）

        用于增量ASR：在不干扰正常输出的情况下读取当前累积的音频
        返回 (音频数组, 时长) 或 None
        """
        if not self._speech_buffer or not self._is_speaking:
            return None
        audio = np.concatenate(self._speech_buffer)
        duration = self._speech_samples / self.sample_rate
        return audio, duration

    def trim_front(self, n_samples: int):
        """
        从缓冲区头部移除样本

        用于增量ASR：已处理的音频从缓冲区移除，避免重复识别
        同时标记 _was_trimmed=True，这样短片段会被强制输出而不是丢弃
        """
        if n_samples <= 0:
            return
        removed = 0
        while self._speech_buffer and removed < n_samples:
            chunk = self._speech_buffer[0]
            if removed + len(chunk) <= n_samples:
                self._speech_buffer.pop(0)
                self._confidence_history.pop(0)
                removed += len(chunk)
            else:
                # 部分修剪第一个块
                keep = removed + len(chunk) - n_samples
                self._speech_buffer[0] = chunk[-keep:]
                removed = n_samples
        self._speech_samples = sum(len(b) for b in self._speech_buffer)
        self._was_trimmed = True
        log.debug(f"trim_front: removed {removed} samples, remaining {self._speech_samples / self.sample_rate:.2f}s")

    def force_flush(self):
        """强制输出缓冲区（忽略最小长度限制）"""
        if not self._speech_buffer:
            return None
        segment = np.concatenate(self._speech_buffer)
        self._reset()
        return segment

    def flush(self):
        """正常flush（检查最小长度限制）"""
        if self._speech_samples >= self.min_speech_samples:
            return self._flush_segment()
        self._reset()
        return None
