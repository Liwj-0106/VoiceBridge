# 更新日志

## 2026-04-20
- 移除 Qwen3-ASR 引擎 (ONNX + GGUF 方案兼容性较差, 相关模型文件和 llama.cpp 运行时依赖一并清理)
- 模型配置新增「高级参数」: `temperature` / `top_p` / `max_tokens` / `frequency_penalty` / `presence_penalty` / `seed`, 每项独立「覆盖」开关, 未勾选时使用服务端默认值
- 模型配置新增 `extra_body` (JSON): 供应商专有参数透传, 如 `thinking_budget`、`reasoning_effort` 等, 保存时自动校验 JSON 格式
- 修复 Anime-Whisper 模型未缓存时下载对话框静默无动作的问题
- 修复设置面板「更新日志」Tab 空白 (正则匹配 H3 但文件用 H2, 3 月加入功能起就失效)

## 2026-04-18
- 新增 ASR 引擎: Anime-Whisper (litagin/anime-whisper), 日语动画/Galgame 特化, 擅长识别喘息/叹息等非语言发声
- 修复 HF 缓存检测误判: 下载中断留下的空目录不再被认为"已缓存"

## 2026-03-24
- 字幕窗口: 超长文本自动换行显示(不再分段), 背景高度平滑动画, 文字渲染 Pixmap 缓存
- 主悬浮窗/字幕窗口: 位置和大小自动记忆, 重启后恢复
- 主悬浮窗: 精简模式切换动画
- 设置: 移除配置文件 key 白名单限制

## 2026-03-23
- 品牌更名 LiveTrans → LiveTranslate
- 模型配置新增: 流式传输、结构化输出、上下文数、禁用思考(默认开启)
- 翻译结果流式逐字显示
- 提示词优化: 禁止多候选翻译, 编辑即时生效
- 模型重复输出检测与提示
- ASR 引擎标注[准确]/[快速]
- 设置面板新增更新日志 Tab
