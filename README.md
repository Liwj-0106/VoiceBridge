# VoiceBridge

Windows 平台实时语音识别翻译工具。

## 功能

- 系统音频捕获 (WASAPI Loopback)
- VAD 语音检测 (Silero VAD)
- ASR 语音识别 (SenseVoice)
- LLM 翻译 (OpenAI-compatible API)
- 字幕窗口显示

## 安装

```bash
conda create -n voicebridge python=3.10
conda activate voicebridge
pip install -r requirements.txt
```

## 配置

1. 复制 `.env.example` 为 `.env`，填入 API Key:
   ```env
   VOICEBRIDGE_API_KEY=sk-xxxx
   ```

2. 修改 `config.yaml` 中的参数（可选）

## 运行

```bash
conda activate voicebridge
python main.py
```

## 测试

```bash
# 测试音频捕获
python tests_manual/test_audio_capture.py

# 测试 ASR
python tests_manual/test_asr.py

# 测试翻译
python tests_manual/test_translator.py
```

## 项目结构

```
VoiceBridge/
├── main.py                 # 入口
├── config.yaml            # 配置文件
├── requirements.txt       # 依赖
├── voicebridge/
│   ├── app/               # 应用编排
│   ├── audio/             # 音频捕获
│   ├── vad/               # 语音检测
│   ├── asr/               # 语音识别
│   ├── translate/         # 文本翻译
│   ├── ui/                # 字幕窗口
│   ├── config/            # 配置加载
│   ├── models/            # 模型管理
│   └── common/            # 公共组件
└── tests_manual/          # 手动测试脚本
```
