# Indextts-Novel

基于 IndexTTS2 提供长文本语音合成工具，适用于制作有声小说等长语音。

## 安装指南

> 详细的环境配置请前往indextts2源项目查看，或者参考[源项目readme](./indextts-readme.md)

## 使用方法

### WebUI

```bash
uv run webui.py
```

打开浏览器访问 `http://127.0.0.1:7860`

## 技术架构

基于 IndexTTS2 自回归零样本语音合成模型，具有以下特点：

- 情感表达与时长可控
- 音色与情感解耦
- 支持多种情感控制方式（参考音频、情感向量、文本描述）

## 许可证

本项目基于 bilibili IndexTTS2 开发，遵循 bilibili Model Use License Agreement。

详细许可证信息请查看：

- [LICENSE](LICENSE) - 模型使用许可证
- [LICENSE_ZH.txt](LICENSE_ZH.txt) - 中文许可证

## 致谢

本项目基于以下开源项目开发：

- [IndexTTS2](https://github.com/index-tts/index-tts) - 情感表达与时长可控的自回归零样本语音合成模型
- [Gradio](https://github.com/gradio-app/gradio) - WebUI框架
