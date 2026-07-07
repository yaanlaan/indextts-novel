import html
import json
import os
import sys
import threading
import time

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(
    description="Indextts-Novel WebUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
parser.add_argument("--gui_seg_tokens", type=int, default=120, help="GUI: Max tokens per generation segment")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr
from indextts.infer_v2 import IndexTTS2
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="Auto")
MODE = 'local'
tts = IndexTTS2(model_dir=cmd_args.model_dir,
                cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
                use_fp16=cmd_args.fp16,
                use_deepspeed=cmd_args.deepspeed,
                use_cuda_kernel=cmd_args.cuda_kernel,
                )
# 支持的语言列表
LANGUAGES = {
    "中文": "zh_CN",
    "English": "en_US"
}
EMO_CHOICES_ALL = [i18n("与音色参考音频相同"),
                i18n("使用情感参考音频"),
                i18n("使用情感向量控制"),
                i18n("使用情感描述文本控制")]
EMO_CHOICES_OFFICIAL = EMO_CHOICES_ALL[:-1]  # skip experimental features

os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)

MAX_LENGTH_TO_USE_SPEED = 70
example_cases = []
with open("examples/cases.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        example = json.loads(line)
        if example.get("emo_audio",None):
            emo_audio_path = os.path.join("examples",example["emo_audio"])
        else:
            emo_audio_path = None

        example_cases.append([os.path.join("examples", example.get("prompt_audio", "sample_prompt.wav")),
                              EMO_CHOICES_ALL[example.get("emo_mode",0)],
                              example.get("text"),
                             emo_audio_path,
                             example.get("emo_weight",1.0),
                             example.get("emo_text",""),
                             example.get("emo_vec_1",0),
                             example.get("emo_vec_2",0),
                             example.get("emo_vec_3",0),
                             example.get("emo_vec_4",0),
                             example.get("emo_vec_5",0),
                             example.get("emo_vec_6",0),
                             example.get("emo_vec_7",0),
                             example.get("emo_vec_8",0),
                             ])

def get_example_cases(include_experimental = False):
    if include_experimental:
        return example_cases  # show every example

    # exclude emotion control mode 3 (emotion from text description)
    return [x for x in example_cases if x[1] != EMO_CHOICES_ALL[3]]

def format_glossary_markdown():
    """将词汇表转换为Markdown表格格式"""
    if not tts.normalizer.term_glossary:
        return i18n("暂无术语")

    lines = [f"| {i18n('术语')} | {i18n('中文读法')} | {i18n('英文读法')} |"]
    lines.append("|---|---|---|")

    for term, reading in tts.normalizer.term_glossary.items():
        zh = reading.get("zh", "") if isinstance(reading, dict) else reading
        en = reading.get("en", "") if isinstance(reading, dict) else reading
        lines.append(f"| {term} | {zh} | {en} |")

    return "\n".join(lines)

def gen_single(emo_control_method,prompt, text,
               emo_ref_path, emo_weight,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               emo_text,emo_random,
               max_text_tokens_per_segment=120,
                *args, progress=gr.Progress()):
    output_path = None
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    # set gradio progress
    tts.gr_progress = progress
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
        # "typical_sampling": bool(typical_sampling),
        # "typical_mass": float(typical_mass),
    }
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:  # emotion from speaker
        emo_ref_path = None  # remove external reference audio
    if emo_control_method == 1:  # emotion from reference audio
        pass
    if emo_control_method == 2:  # emotion from custom vectors
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec = tts.normalize_emo_vec(vec, apply_bias=True)
    else:
        # don't use the emotion vector inputs for the other modes
        vec = None

    if emo_text == "":
        # erase empty emotion descriptions; `infer()` will then automatically use the main prompt
        emo_text = None

    print(f"Emo control mode:{emo_control_method},weight:{emo_weight},vec:{vec}")
    output = tts.infer(spk_audio_prompt=prompt, text=text,
                       output_path=output_path,
                       emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                       emo_vector=vec,
                       use_emo_text=(emo_control_method==3), emo_text=emo_text,use_random=emo_random,
                       verbose=cmd_args.verbose,
                       max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                       **kwargs)
    return gr.update(value=output,visible=True)

def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button

def create_warning_message(warning_text):
    return gr.HTML(f"<div style=\"padding: 0.5em 0.8em; border-radius: 0.5em; background: #ffa87d; color: #000; font-weight: bold\">{html.escape(warning_text)}</div>")

def create_experimental_warning_message():
    return create_warning_message(i18n('提示：此功能为实验版，结果尚不稳定，我们正在持续优化中。'))

import re
import zipfile

def split_long_text(text):
    sentences = []
    text = text.strip()
    if not text:
        return sentences
    
    chunks = re.split(r'(\n+)', text)
    current_sentence = ""
    
    for chunk in chunks:
        if chunk.startswith('\n'):
            if current_sentence.strip():
                sentences.extend(split_by_punctuation(current_sentence.strip()))
            current_sentence = ""
        else:
            current_sentence += chunk
    
    if current_sentence.strip():
        sentences.extend(split_by_punctuation(current_sentence.strip()))
    
    sentences = [ensure_punctuation(s) for s in sentences if s.strip()]
    return sentences

def protect_quoted_content(text):
    quote_pairs = [
        ('『', '』'), ('「', '」'), ('“', '”'), ('‘', '’'),
        ('"', '"'), ("'", "'"), ('《', '》'), ('（', '）'), ('(', ')')
    ]
    
    placeholders = []
    protected_text = text
    
    for start_quote, end_quote in quote_pairs:
        if start_quote == end_quote:
            pattern = re.escape(start_quote) + r'([^' + re.escape(end_quote) + r']+)' + re.escape(end_quote)
        else:
            pattern = re.escape(start_quote) + r'(.*?)' + re.escape(end_quote)
        
        matches = list(re.finditer(pattern, protected_text))
        offset = 0
        for match in matches:
            content = match.group(1)
            placeholder = f"\x00QUOTE{len(placeholders):04d}\x00"
            placeholders.append(content)
            
            start = match.start() + offset
            end = match.end() + offset
            protected_text = protected_text[:start] + start_quote + placeholder + end_quote + protected_text[end:]
            offset += len(start_quote + placeholder + end_quote) - len(match.group(0))
    
    return protected_text, placeholders

def restore_quoted_content(text, placeholders):
    for i, content in enumerate(placeholders):
        placeholder = f"\x00QUOTE{i:04d}\x00"
        text = text.replace(placeholder, content)
    return text

def split_by_punctuation(text):
    if not text:
        return []
    
    protected_text, placeholders = protect_quoted_content(text)
    
    patterns = [
        r'([。！？])',
        r'([.!?])',
    ]
    
    for pattern in patterns:
        parts = re.split(pattern, protected_text)
        if len(parts) > 1:
            sentences = []
            current = ""
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    current = part
                else:
                    current += part
                    if current.strip():
                        restored = restore_quoted_content(current.strip(), placeholders)
                        sentences.append(restored)
                    current = ""
            if current.strip():
                restored = restore_quoted_content(current.strip(), placeholders)
                sentences.append(restored)
            return sentences
    
    restored = restore_quoted_content(text.strip(), placeholders)
    return [restored]

def ensure_punctuation(sentence):
    sentence = sentence.strip()
    if not sentence:
        return sentence
    
    punctuation = {'。', '！', '？', '.', '!', '?', '；', ';', '：', ':'}
    if sentence[-1] not in punctuation:
        sentence += '。'
    return sentence

def apply_fade(wav, sampling_rate, fade_in_ms=0, fade_out_ms=0):
    import torch
    
    num_samples = wav.shape[1]
    if fade_in_ms > 0:
        fade_in_samples = int(sampling_rate * fade_in_ms / 1000.0)
        fade_in_samples = min(fade_in_samples, num_samples)
        fade_in_curve = torch.linspace(0.0, 1.0, fade_in_samples)
        wav[:, :fade_in_samples] *= fade_in_curve
    
    if fade_out_ms > 0:
        fade_out_samples = int(sampling_rate * fade_out_ms / 1000.0)
        fade_out_samples = min(fade_out_samples, num_samples)
        fade_out_curve = torch.linspace(1.0, 0.0, fade_out_samples)
        wav[:, -fade_out_samples:] *= fade_out_curve
    
    return wav

def merge_audio_files(audio_paths, output_path, sampling_rate=22050, interval_silence_ms=200, fade_in_ms=0, fade_out_ms=0):
    import torchaudio
    import torch
    
    wavs = []
    sil_dur = int(sampling_rate * interval_silence_ms / 1000.0)
    silence = torch.zeros(1, sil_dur)
    
    for i, path in enumerate(audio_paths):
        if os.path.exists(path):
            wav, sr = torchaudio.load(path)
            if sr != sampling_rate:
                wav = torchaudio.transforms.Resample(sr, sampling_rate)(wav)
            if wav.dtype == torch.int16:
                wav = wav.float() / 32768.0
            
            wav = apply_fade(wav, sampling_rate, fade_in_ms, fade_out_ms)
            wavs.append(wav)
            if i < len(audio_paths) - 1:
                wavs.append(silence)
    
    if wavs:
        merged = torch.cat(wavs, dim=1)
        torchaudio.save(output_path, merged, sampling_rate)
        return output_path
    return None

def generate_long_text_tts(
    prompt_audio, input_text, file_upload,
    emo_control_method, emo_ref_path, emo_weight,
    vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
    emo_text, emo_random,
    max_text_tokens_per_segment,
    do_sample, top_p, top_k, temperature,
    length_penalty, num_beams, repetition_penalty, max_mel_tokens,
    interval_silence_ms, fade_in_ms, fade_out_ms,
    progress=gr.Progress()
):
    if file_upload is not None:
        with open(file_upload, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = input_text or ""
    
    if not text.strip():
        gr.Warning(i18n("请输入或上传文本"))
        return None, None, None, None, None
    
    sentences = split_long_text(text)
    if not sentences:
        gr.Warning(i18n("未检测到有效句子"))
        return None, None, None, None, None
    
    total_sentences = len(sentences)
    progress(0, desc=i18n(f"共 {total_sentences} 句，开始生成..."))
    
    task_id = str(int(time.time()))
    output_dir = os.path.join("outputs", "long_text", task_id)
    os.makedirs(output_dir, exist_ok=True)
    
    audio_paths = []
    sentence_results = []
    
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
    }
    
    emo_vector = None
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:
        emo_ref_path = None
    elif emo_control_method == 2:
        emo_vector = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        emo_vector = tts.normalize_emo_vec(emo_vector, apply_bias=True)
    
    if emo_text == "":
        emo_text = None
    
    for idx, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            sentence_results.append({"index": idx, "sentence": sentence, "status": "跳过", "audio_path": None})
            continue
        
        progress((idx + 1) / total_sentences, desc=i18n(f"正在生成第 {idx + 1}/{total_sentences} 句..."))
        
        try:
            output_path = os.path.join(output_dir, f"sentence_{idx + 1:04d}.wav")
            
            output = tts.infer(
                spk_audio_prompt=prompt_audio,
                text=sentence,
                output_path=output_path,
                emo_audio_prompt=emo_ref_path,
                emo_alpha=emo_weight,
                emo_vector=emo_vector,
                use_emo_text=(emo_control_method == 3),
                emo_text=emo_text,
                use_random=emo_random,
                verbose=cmd_args.verbose,
                max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                **kwargs
            )
            
            if output and os.path.exists(output):
                audio_paths.append(output)
                sentence_results.append({"index": idx, "sentence": sentence, "status": "成功", "audio_path": output})
            else:
                sentence_results.append({"index": idx, "sentence": sentence, "status": "失败", "audio_path": None})
        except Exception as e:
            print(f"Error generating sentence {idx}: {e}")
            sentence_results.append({"index": idx, "sentence": sentence, "status": f"错误: {str(e)[:50]}", "audio_path": None})
    
    merged_path = None
    if audio_paths:
        merged_path = os.path.join(output_dir, "merged.wav")
        merge_audio_files(audio_paths, merged_path, 
                          interval_silence_ms=interval_silence_ms,
                          fade_in_ms=fade_in_ms,
                          fade_out_ms=fade_out_ms)
    
    zip_path = None
    if audio_paths:
        zip_path = os.path.join(output_dir, "all_audio.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for path in audio_paths:
                zf.write(path, os.path.basename(path))
    
    results_df = pd.DataFrame([
        {"序号": r["index"] + 1, "句子内容": r["sentence"], "状态": r["status"]}
        for r in sentence_results
    ])
    
    return merged_path, zip_path, results_df, len(audio_paths), total_sentences

lt_stop_event = threading.Event()
lt_pause_event = threading.Event()
lt_pause_event.set()

def reset_lt_events():
    lt_stop_event.clear()
    lt_pause_event.set()

def toggle_lt_pause():
    if lt_pause_event.is_set():
        lt_pause_event.clear()
        return i18n("已暂停"), gr.update(value=i18n("暂停中"), visible=True)
    else:
        lt_pause_event.set()
        return i18n("继续"), gr.update(value=i18n("生成中"), visible=True)

def stop_lt_generation():
    lt_stop_event.set()
    lt_pause_event.set()
    return i18n("已终止"), gr.update(value=i18n("已终止"), visible=True)

with gr.Blocks(title="Indextts-Novel Demo") as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <h2><center>Indextts-Novel: Long Text to Speech for Novels and Stories</h2>
<p align="center">
<a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
</p>
    ''')

    with gr.Tab(i18n("音频生成")):
        with gr.Row():
            os.makedirs("prompts",exist_ok=True)
            prompt_audio = gr.Audio(label=i18n("音色参考音频"),key="prompt_audio",
                                    sources=["upload","microphone"],type="filepath")
            prompt_list = os.listdir("prompts")
            default = ''
            if prompt_list:
                default = prompt_list[0]
            with gr.Column():
                input_text_single = gr.TextArea(label=i18n("文本"),key="input_text_single", placeholder=i18n("请输入目标文本"), info=f"{i18n('当前模型版本')}{tts.model_version or '1.0'}")
                gen_button = gr.Button(i18n("生成语音"), key="gen_button",interactive=True)
            output_audio = gr.Audio(label=i18n("生成结果"), visible=True,key="output_audio")

        with gr.Row():
            experimental_checkbox = gr.Checkbox(label=i18n("显示实验功能"), value=False)
            glossary_checkbox = gr.Checkbox(label=i18n("开启术语词汇读音"), value=tts.normalizer.enable_glossary)
        with gr.Accordion(i18n("功能设置")):
            # 情感控制选项部分
            with gr.Row():
                emo_control_method = gr.Radio(
                    choices=EMO_CHOICES_OFFICIAL,
                    type="index",
                    value=EMO_CHOICES_OFFICIAL[0],label=i18n("情感控制方式"))
                # we MUST have an extra, INVISIBLE list of *all* emotion control
                # methods so that gr.Dataset() can fetch ALL control mode labels!
                # otherwise, the gr.Dataset()'s experimental labels would be empty!
                emo_control_method_all = gr.Radio(
                    choices=EMO_CHOICES_ALL,
                    type="index",
                    value=EMO_CHOICES_ALL[0], label=i18n("情感控制方式"),
                    visible=False)  # do not render
        # 情感参考音频部分
        with gr.Group(visible=False) as emotion_reference_group:
            with gr.Row():
                emo_upload = gr.Audio(label=i18n("上传情感参考音频"), type="filepath")

        # 情感随机采样
        with gr.Row(visible=False) as emotion_randomize_group:
            emo_random = gr.Checkbox(label=i18n("情感随机采样"), value=False)

        # 情感向量控制部分
        with gr.Group(visible=False) as emotion_vector_group:
            with gr.Row():
                with gr.Column():
                    vec1 = gr.Slider(label=i18n("喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec2 = gr.Slider(label=i18n("怒"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec3 = gr.Slider(label=i18n("哀"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec4 = gr.Slider(label=i18n("惧"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                with gr.Column():
                    vec5 = gr.Slider(label=i18n("厌恶"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec6 = gr.Slider(label=i18n("低落"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec7 = gr.Slider(label=i18n("惊喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    vec8 = gr.Slider(label=i18n("平静"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)

        with gr.Group(visible=False) as emo_text_group:
            create_experimental_warning_message()
            with gr.Row():
                emo_text = gr.Textbox(label=i18n("情感描述文本"),
                                      placeholder=i18n("请输入情绪描述（或留空以自动使用目标文本作为情绪描述）"),
                                      value="",
                                      info=i18n("例如：委屈巴巴、危险在悄悄逼近"))

        with gr.Row(visible=False) as emo_weight_group:
            emo_weight = gr.Slider(label=i18n("情感权重"), minimum=0.0, maximum=1.0, value=0.65, step=0.01)

        # 术语词汇表管理
        with gr.Accordion(i18n("自定义术语词汇读音"), open=False, visible=tts.normalizer.enable_glossary) as glossary_accordion:
            gr.Markdown(i18n("自定义个别专业术语的读音"))
            with gr.Row():
                with gr.Column(scale=1):
                    glossary_term = gr.Textbox(
                        label=i18n("术语"),
                        placeholder="IndexTTS2",
                    )
                    glossary_reading_zh = gr.Textbox(
                        label=i18n("中文读法"),
                        placeholder="Index T-T-S 二",
                    )
                    glossary_reading_en = gr.Textbox(
                        label=i18n("英文读法"),
                        placeholder="Index T-T-S two",
                    )
                    btn_add_term = gr.Button(i18n("添加术语"), scale=1)
                with gr.Column(scale=2):
                    glossary_table = gr.Markdown(
                        value=format_glossary_markdown()
                    )

        with gr.Accordion(i18n("高级生成参数设置"), open=False, visible=True) as advanced_settings_group:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"**{i18n('GPT2 采样设置')}** _{i18n('参数会影响音频多样性和生成速度详见')} [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)._")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=True, info=i18n("是否进行采样"))
                        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                        num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                        length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                    max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info=i18n("生成Token最大数量，过小导致音频被截断"), key="max_mel_tokens")
                    # with gr.Row():
                    #     typical_sampling = gr.Checkbox(label="typical_sampling", value=False, info="不建议使用")
                    #     typical_mass = gr.Slider(label="typical_mass", value=0.9, minimum=0.0, maximum=1.0, step=0.1)
                with gr.Column(scale=2):
                    gr.Markdown(f'**{i18n("分句设置")}** _{i18n("参数会影响音频质量和生成速度")}_')
                    with gr.Row():
                        initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                        max_text_tokens_per_segment = gr.Slider(
                            label=i18n("分句最大Token数"), value=initial_value, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2, key="max_text_tokens_per_segment",
                            info=i18n("建议80~200之间，值越大，分句越长；值越小，分句越碎；过小过大都可能导致音频质量不高"),
                        )
                    with gr.Accordion(i18n("预览分句结果"), open=True) as segments_settings:
                        segments_preview = gr.Dataframe(
                            headers=[i18n("序号"), i18n("分句内容"), i18n("Token数")],
                            key="segments_preview",
                            wrap=True,
                        )
            advanced_params = [
                do_sample, top_p, top_k, temperature,
                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                # typical_sampling, typical_mass,
            ]

        # we must use `gr.Dataset` to support dynamic UI rewrites, since `gr.Examples`
        # binds tightly to UI and always restores the initial state of all components,
        # such as the list of available choices in emo_control_method.
        example_table = gr.Dataset(label="Examples",
            samples_per_page=20,
            samples=get_example_cases(include_experimental=False),
            type="values",
            # these components are NOT "connected". it just reads the column labels/available
            # states from them, so we MUST link to the "all options" versions of all components,
            # such as `emo_control_method_all` (to be able to see EXPERIMENTAL text labels)!
            components=[prompt_audio,
                        emo_control_method_all,  # important: support all mode labels!
                        input_text_single,
                        emo_upload,
                        emo_weight,
                        emo_text,
                        vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        )

    def on_example_click(example):
        print(f"Example clicked: ({len(example)} values) = {example!r}")
        return (
            gr.update(value=example[0]),
            gr.update(value=example[1]),
            gr.update(value=example[2]),
            gr.update(value=example[3]),
            gr.update(value=example[4]),
            gr.update(value=example[5]),
            gr.update(value=example[6]),
            gr.update(value=example[7]),
            gr.update(value=example[8]),
            gr.update(value=example[9]),
            gr.update(value=example[10]),
            gr.update(value=example[11]),
            gr.update(value=example[12]),
            gr.update(value=example[13]),
        )

    # click() event works on both desktop and mobile UI
    example_table.click(on_example_click,
                        inputs=[example_table],
                        outputs=[prompt_audio,
                                 emo_control_method,
                                 input_text_single,
                                 emo_upload,
                                 emo_weight,
                                 emo_text,
                                 vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
    )

    def on_input_text_change(text, max_text_tokens_per_segment):
        if text and len(text) > 0:
            text_tokens_list = tts.tokenizer.tokenize(text)

            segments = tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_text_tokens_per_segment))
            data = []
            for i, s in enumerate(segments):
                segment_str = ''.join(s)
                tokens_count = len(s)
                data.append([i, segment_str, tokens_count])
            return {
                segments_preview: gr.update(value=data, visible=True, type="array"),
            }
        else:
            df = pd.DataFrame([], columns=[i18n("序号"), i18n("分句内容"), i18n("Token数")])
            return {
                segments_preview: gr.update(value=df),
            }

    # 术语词汇表事件处理函数
    def on_add_glossary_term(term, reading_zh, reading_en):
        """添加术语到词汇表并自动保存"""
        term = term.rstrip()
        reading_zh = reading_zh.rstrip()
        reading_en = reading_en.rstrip()

        if not term:
            gr.Warning(i18n("请输入术语"))
            return gr.update()
            
        if not reading_zh and not reading_en:
            gr.Warning(i18n("请至少输入一种读法"))
            return gr.update()
        

        # 构建读法数据
        if reading_zh and reading_en:
            reading = {"zh": reading_zh, "en": reading_en}
        elif reading_zh:
            reading = {"zh": reading_zh}
        elif reading_en:
            reading = {"en": reading_en}
        else:
            reading = reading_zh or reading_en

        # 添加到词汇表
        tts.normalizer.term_glossary[term] = reading

        # 自动保存到文件
        try:
            tts.normalizer.save_glossary_to_yaml(tts.glossary_path)
            gr.Info(i18n("词汇表已更新"), duration=1)
        except Exception as e:
            gr.Error(i18n("保存词汇表时出错"))
            print(f"Error details: {e}")
            return gr.update()

        # 更新Markdown表格
        return gr.update(value=format_glossary_markdown())
        

    def on_method_change(emo_control_method):
        if emo_control_method == 1:  # emotion reference audio
            return (gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 2:  # emotion vectors
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True)
                    )
        elif emo_control_method == 3:  # emotion text description
            return (gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=True)
                    )
        else:  # 0: same as speaker voice
            return (gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                    )

    emo_control_method.change(on_method_change,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group,
                 emotion_randomize_group,
                 emotion_vector_group,
                 emo_text_group,
                 emo_weight_group]
    )

    def on_experimental_change(is_experimental, current_mode_index):
        # 切换情感控制选项
        new_choices = EMO_CHOICES_ALL if is_experimental else EMO_CHOICES_OFFICIAL
        # if their current mode selection doesn't exist in new choices, reset to 0.
        # we don't verify that OLD index means the same in NEW list, since we KNOW it does.
        new_index = current_mode_index if current_mode_index < len(new_choices) else 0

        return (
            gr.update(choices=new_choices, value=new_choices[new_index]),
            gr.update(samples=get_example_cases(include_experimental=is_experimental)),
        )

    experimental_checkbox.change(
        on_experimental_change,
        inputs=[experimental_checkbox, emo_control_method],
        outputs=[emo_control_method, example_table]
    )

    def on_glossary_checkbox_change(is_enabled):
        """控制术语词汇表的可见性"""
        tts.normalizer.enable_glossary = is_enabled
        return gr.update(visible=is_enabled)

    glossary_checkbox.change(
        on_glossary_checkbox_change,
        inputs=[glossary_checkbox],
        outputs=[glossary_accordion]
    )

    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    max_text_tokens_per_segment.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )

    prompt_audio.upload(update_prompt_audio,
                         inputs=[],
                         outputs=[gen_button])

    def on_demo_load():
        """页面加载时重新加载glossary数据"""
        try:
            tts.normalizer.load_glossary_from_yaml(tts.glossary_path)
        except Exception as e:
            gr.Error(i18n("加载词汇表时出错"))
            print(f"Failed to reload glossary on page load: {e}")
        return gr.update(value=format_glossary_markdown())

    # 术语词汇表事件绑定
    btn_add_term.click(
        on_add_glossary_term,
        inputs=[glossary_term, glossary_reading_zh, glossary_reading_en],
        outputs=[glossary_table]
    )

    # 页面加载时重新加载glossary
    demo.load(
        on_demo_load,
        inputs=[],
        outputs=[glossary_table]
    )

    gen_button.click(gen_single,
                     inputs=[emo_control_method,prompt_audio, input_text_single, emo_upload, emo_weight,
                            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                             emo_text,emo_random,
                             max_text_tokens_per_segment,
                             *advanced_params,
                     ],
                     outputs=[output_audio])

    with gr.Tab(i18n("长文本配音")):
        gr.Markdown(i18n("上传TXT文件或输入长文本，系统将自动分句并逐句生成语音"))
        
        lt_sentence_results = gr.State([])
        lt_output_dir = gr.State("")
        
        with gr.Row():
            with gr.Column(scale=2):
                lt_input_text = gr.TextArea(label=i18n("输入文本"), placeholder=i18n("在此输入长文本，或上传TXT文件"), lines=10)
                lt_file_upload = gr.File(label=i18n("上传TXT文件"), file_types=[".txt"])
                
                with gr.Row():
                    lt_load_btn = gr.Button(i18n("载入文本"), variant="secondary")
                    lt_segment_btn = gr.Button(i18n("分句处理"), variant="secondary", visible=False)
            
            with gr.Column(scale=1):
                lt_prompt_audio = gr.Audio(label=i18n("音色参考音频"), sources=["upload", "microphone"], type="filepath")
                lt_gen_button = gr.Button(i18n("开始生成"), variant="primary", visible=False)
                lt_pause_button = gr.Button(i18n("暂停"), variant="secondary", visible=False)
                lt_stop_button = gr.Button(i18n("终止"), variant="stop", visible=False)
        
        with gr.Row():
            lt_experimental_checkbox = gr.Checkbox(label=i18n("显示实验功能"), value=False)
        
        with gr.Accordion(i18n("情感控制"), open=False):
            with gr.Row():
                lt_emo_control_method = gr.Radio(
                    choices=EMO_CHOICES_OFFICIAL,
                    type="index",
                    value=EMO_CHOICES_OFFICIAL[0],
                    label=i18n("情感控制方式")
                )
                lt_emo_control_method_all = gr.Radio(
                    choices=EMO_CHOICES_ALL,
                    type="index",
                    value=EMO_CHOICES_ALL[0],
                    label=i18n("情感控制方式"),
                    visible=False
                )
            
            with gr.Group(visible=False) as lt_emotion_reference_group:
                lt_emo_upload = gr.Audio(label=i18n("上传情感参考音频"), type="filepath")
            
            with gr.Row(visible=False) as lt_emotion_randomize_group:
                lt_emo_random = gr.Checkbox(label=i18n("情感随机采样"), value=False)
            
            with gr.Group(visible=False) as lt_emotion_vector_group:
                with gr.Row():
                    with gr.Column():
                        lt_vec1 = gr.Slider(label=i18n("喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        lt_vec2 = gr.Slider(label=i18n("怒"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        lt_vec3 = gr.Slider(label=i18n("哀"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        lt_vec4 = gr.Slider(label=i18n("惧"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                    with gr.Column():
                        lt_vec5 = gr.Slider(label=i18n("厌恶"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        lt_vec6 = gr.Slider(label=i18n("低落"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        lt_vec7 = gr.Slider(label=i18n("惊喜"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
                        lt_vec8 = gr.Slider(label=i18n("平静"), minimum=0.0, maximum=1.0, value=0.0, step=0.05)
            
            with gr.Group(visible=False) as lt_emo_text_group:
                create_experimental_warning_message()
                lt_emo_text = gr.Textbox(label=i18n("情感描述文本"), placeholder=i18n("请输入情绪描述"), value="")
            
            with gr.Row(visible=False) as lt_emo_weight_group:
                lt_emo_weight = gr.Slider(label=i18n("情感权重"), minimum=0.0, maximum=1.0, value=0.65, step=0.01)
        
        with gr.Accordion(i18n("高级参数"), open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"**{i18n('GPT2 采样设置')}**")
                    lt_do_sample = gr.Checkbox(label="do_sample", value=True)
                    lt_temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                    lt_top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                    lt_top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                    lt_num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                    lt_repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                    lt_length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                    lt_max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10)
                
                with gr.Column(scale=1):
                    gr.Markdown(f"**{i18n('分句设置')}**")
                    lt_max_text_tokens_per_segment = gr.Slider(
                        label=i18n("分句最大Token数"),
                        value=max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens)),
                        minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2,
                        info=i18n("建议80~200之间")
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown(f"**{i18n('音频衔接控制')}**")
                    lt_interval_silence = gr.Slider(
                        label=i18n("句间静音间隔(ms)"),
                        value=200, minimum=0, maximum=2000, step=50,
                        info=i18n("每段音频之间的静音时长")
                    )
                    lt_fade_in = gr.Slider(
                        label=i18n("淡入时长(ms)"),
                        value=0, minimum=0, maximum=1000, step=50,
                        info=i18n("每段音频开始时的淡入效果")
                    )
                    lt_fade_out = gr.Slider(
                        label=i18n("淡出时长(ms)"),
                        value=0, minimum=0, maximum=1000, step=50,
                        info=i18n("每段音频结束时的淡出效果")
                    )
        
        lt_raw_text_state = gr.State("")
        lt_sentences_state = gr.State([])
        
        with gr.Accordion(i18n("分句结果"), open=True):
            lt_sentences_df = gr.Dataframe(
                headers=[i18n("选择"), i18n("序号"), i18n("句子内容")],
                wrap=True,
                interactive=True,
                datatype=["bool", "number", "str"]
            )
            
            with gr.Row():
                lt_merge_selected_btn = gr.Button(i18n("合并选中句子"), variant="secondary")
                lt_merge_batch_count = gr.Number(label=i18n("每N句合并"), value=2, minimum=2, maximum=20, precision=0)
                lt_merge_batch_btn = gr.Button(i18n("批量合并"), variant="secondary")
            
            with gr.Row():
                lt_delete_idx_input = gr.Number(label=i18n("删除句子序号"), minimum=1, step=1)
                lt_delete_btn = gr.Button(i18n("删除句子"), variant="stop")
            
            with gr.Row():
                lt_retry_idx_input = gr.Number(label=i18n("重推句子序号"), minimum=1, step=1)
                lt_retry_btn = gr.Button(i18n("重新生成"), variant="primary")
        
        lt_status_label = gr.Label(visible=False)
        
        with gr.Row(visible=False) as lt_progress_row:
            lt_progress_info = gr.Markdown()
            lt_progress_bar = gr.Slider(
                label=i18n("生成进度"), minimum=0, maximum=100, value=0, step=1,
                interactive=False
            )
        
        lt_current_sentence = gr.Textbox(
            label=i18n("当前处理句子"), visible=False, lines=2, max_lines=3
        )
        
        lt_progress_stats = gr.Markdown(visible=False)
        
        lt_results_area = gr.Dataframe(
            headers=[i18n("序号"), i18n("句子内容"), i18n("状态"), i18n("耗时")],
            wrap=True,
            visible=False
        )
        
        with gr.Group(visible=False) as lt_results_group:
            gr.Markdown(i18n("生成结果"))
            with gr.Row():
                lt_merged_audio = gr.Audio(label=i18n("合并音频"))
                lt_download_merged = gr.DownloadButton(label=i18n("下载合并音频"))
                lt_download_all = gr.DownloadButton(label=i18n("下载全部单独音频"))
            
            with gr.Row():
                lt_stats = gr.Markdown()
        
        lt_advanced_params = [
            lt_do_sample, lt_top_p, lt_top_k, lt_temperature,
            lt_length_penalty, lt_num_beams, lt_repetition_penalty, lt_max_mel_tokens,
        ]
        
        def sentences_to_df(sentences):
            if not sentences:
                return pd.DataFrame([], columns=[i18n("选择"), i18n("序号"), i18n("句子内容")])
            return pd.DataFrame([
                {i18n("选择"): False, i18n("序号"): i + 1, i18n("句子内容"): sent}
                for i, sent in enumerate(sentences)
            ])
        
        def on_lt_load(text, file_upload):
            if file_upload is not None:
                with open(file_upload, 'r', encoding='utf-8') as f:
                    loaded_text = f.read()
            elif text.strip():
                loaded_text = text.strip()
            else:
                gr.Warning(i18n("请输入或上传文本"))
                return "", gr.update(visible=False), gr.update(visible=False)
            
            text_length = len(loaded_text)
            char_count = len(loaded_text)
            line_count = loaded_text.count('\n') + 1
            
            gr.Info(i18n(f"文本载入成功！共 {char_count} 字符，{line_count} 行"))
            return loaded_text, gr.update(visible=True), gr.update(visible=False)
        
        def on_lt_segment(raw_text):
            if not raw_text:
                gr.Warning(i18n("请先载入文本"))
                return [], sentences_to_df([]), gr.update(visible=False)
            
            sentences = split_long_text(raw_text)
            if not sentences:
                gr.Warning(i18n("未检测到有效句子"))
                return [], sentences_to_df([]), gr.update(visible=False)
            
            df = sentences_to_df(sentences)
            gr.Info(i18n(f"分句完成！共 {len(sentences)} 句"))
            return sentences, df, gr.update(visible=True)
        
        def on_lt_delete_sentence(sentences, idx):
            if idx is None:
                gr.Warning(i18n("请输入句子序号"))
                return sentences, sentences_to_df(sentences)
            
            idx = int(idx) - 1
            if 0 <= idx < len(sentences):
                sentences.pop(idx)
                df = sentences_to_df(sentences)
                return sentences, df
            else:
                gr.Warning(i18n("句子序号超出范围"))
                return sentences, sentences_to_df(sentences)
        
        def on_lt_merge_selected(sentences, df):
            if df is None or len(df) == 0:
                return sentences, sentences_to_df(sentences)
            
            selected_indices = []
            for i, row in df.iterrows():
                select_col = i18n("选择")
                if select_col in row and row[select_col]:
                    selected_indices.append(i)
            
            if len(selected_indices) < 2:
                gr.Warning(i18n("请至少选择2个句子进行合并"))
                return sentences, sentences_to_df(sentences)
            
            selected_indices.sort()
            merged_text = ''.join([sentences[idx] for idx in selected_indices])
            
            new_sentences = [sent for i, sent in enumerate(sentences) if i not in set(selected_indices)]
            new_sentences.insert(selected_indices[0], merged_text)
            
            df = sentences_to_df(new_sentences)
            gr.Info(i18n(f"已合并 {len(selected_indices)} 个句子"))
            return new_sentences, df
        
        def on_lt_retry_single(
            retry_idx, sentences, sentence_results, output_dir,
            prompt_audio,
            emo_control_method, emo_ref_path, emo_weight,
            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
            emo_text, emo_random,
            max_text_tokens_per_segment,
            do_sample, top_p, top_k, temperature,
            length_penalty, num_beams, repetition_penalty, max_mel_tokens,
        ):
            if retry_idx is None:
                gr.Warning(i18n("请输入句子序号"))
                return sentence_results, None
            
            retry_idx = int(retry_idx) - 1
            if retry_idx < 0 or retry_idx >= len(sentences):
                gr.Warning(i18n("句子序号超出范围"))
                return sentence_results, None
            
            sentence = sentences[retry_idx]
            
            kwargs = {}
            if do_sample:
                kwargs['do_sample'] = do_sample
                kwargs['top_p'] = top_p
                kwargs['top_k'] = top_k
                kwargs['temperature'] = temperature
            kwargs['length_penalty'] = length_penalty
            kwargs['num_beams'] = num_beams
            kwargs['repetition_penalty'] = repetition_penalty
            kwargs['max_mel_tokens'] = max_mel_tokens
            
            emo_vector = None
            if emo_control_method == 2:
                emo_vector = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
                emo_vector = tts.normalize_emo_vec(emo_vector, apply_bias=True)
            
            if emo_text == "":
                emo_text = None
            
            try:
                output_path = os.path.join(output_dir, f"sentence_{retry_idx + 1:04d}.wav")
                
                output = tts.infer(
                    spk_audio_prompt=prompt_audio,
                    text=sentence,
                    output_path=output_path,
                    emo_audio_prompt=emo_ref_path,
                    emo_alpha=emo_weight,
                    emo_vector=emo_vector,
                    use_emo_text=(emo_control_method == 3),
                    emo_text=emo_text,
                    use_random=emo_random,
                    verbose=cmd_args.verbose,
                    max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                    **kwargs
                )
                
                if output and os.path.exists(output):
                    for i, r in enumerate(sentence_results):
                        if r["index"] == retry_idx:
                            sentence_results[i]["status"] = "成功"
                            sentence_results[i]["audio_path"] = output
                            sentence_results[i]["duration"] = "重推"
                            break
                    else:
                        sentence_results.append({"index": retry_idx, "sentence": sentence, "status": "成功", "audio_path": output, "duration": "重推"})
                    gr.Info(i18n(f"第 {retry_idx + 1} 句重新生成成功"))
                    return sentence_results, output
                else:
                    for i, r in enumerate(sentence_results):
                        if r["index"] == retry_idx:
                            sentence_results[i]["status"] = "失败"
                            sentence_results[i]["duration"] = "重推失败"
                            break
                    gr.Warning(i18n(f"第 {retry_idx + 1} 句重新生成失败"))
                    return sentence_results, None
            except Exception as e:
                for i, r in enumerate(sentence_results):
                    if r["index"] == retry_idx:
                        sentence_results[i]["status"] = "失败"
                        sentence_results[i]["duration"] = f"重推失败: {str(e)[:20]}"
                        break
                print(f"Error retrying sentence {retry_idx}: {e}")
                gr.Warning(i18n(f"第 {retry_idx + 1} 句重新生成失败: {str(e)[:30]}"))
                return sentence_results, None
        
        def on_lt_merge_batch(sentences, batch_count):
            if not sentences:
                return sentences, sentences_to_df([])
            
            batch_count = int(batch_count) if batch_count and int(batch_count) >= 2 else 2
            
            new_sentences = []
            i = 0
            while i < len(sentences):
                batch = sentences[i:i + batch_count]
                merged_text = ''.join(batch)
                new_sentences.append(merged_text)
                i += batch_count
            
            df = sentences_to_df(new_sentences)
            return new_sentences, df
        
        def on_lt_df_change(df):
            if df is None or len(df) == 0:
                return []
            sentences = []
            for _, row in df.iterrows():
                text_col = i18n("句子内容")
                if text_col in row:
                    text = row[text_col]
                    if text and str(text).strip():
                        sentences.append(str(text).strip())
            return sentences
        
        def on_lt_method_change(emo_control_method):
            if emo_control_method == 1:
                return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True))
            elif emo_control_method == 2:
                return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True))
            elif emo_control_method == 3:
                return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True))
            else:
                return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
        
        def on_lt_experimental_change(is_experimental, current_mode_index):
            new_choices = EMO_CHOICES_ALL if is_experimental else EMO_CHOICES_OFFICIAL
            new_index = current_mode_index if current_mode_index < len(new_choices) else 0
            return gr.update(choices=new_choices, value=new_choices[new_index])
        
        def on_lt_generate(
            sentences, prompt_audio,
            emo_control_method, emo_ref_path, emo_weight,
            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
            emo_text, emo_random,
            max_text_tokens_per_segment,
            do_sample, top_p, top_k, temperature,
            length_penalty, num_beams, repetition_penalty, max_mel_tokens,
            interval_silence_ms, fade_in_ms, fade_out_ms,
        ):
            reset_lt_events()
            
            if not sentences:
                gr.Warning(i18n("请先处理文本得到分句结果"))
                return None, None, None, None, None, None, None, None, None, None, None
            
            if not prompt_audio:
                gr.Warning(i18n("请先上传音色参考音频"))
                return None, None, None, None, None, None, None, None, None, None, None
            
            total_sentences = len(sentences)
            progress_info = i18n(f"共 {total_sentences} 句，开始生成...")
            
            yield (
                gr.update(value=i18n("生成中"), visible=True),
                gr.update(visible=True),
                gr.update(value=0),
                gr.update(value="", visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value=progress_info, visible=True),
                gr.update(value="", visible=True),
                [],
                ""
            )
            
            task_id = str(int(time.time()))
            output_dir = os.path.join("outputs", "long_text", task_id)
            os.makedirs(output_dir, exist_ok=True)
            
            audio_paths = []
            sentence_results = []
            start_time = time.time()
            
            kwargs = {
                "do_sample": bool(do_sample),
                "top_p": float(top_p),
                "top_k": int(top_k) if int(top_k) > 0 else None,
                "temperature": float(temperature),
                "length_penalty": float(length_penalty),
                "num_beams": num_beams,
                "repetition_penalty": float(repetition_penalty),
                "max_mel_tokens": int(max_mel_tokens),
            }
            
            emo_vector = None
            if type(emo_control_method) is not int:
                emo_control_method = emo_control_method.value
            if emo_control_method == 0:
                emo_ref_path = None
            elif emo_control_method == 2:
                emo_vector = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
                emo_vector = tts.normalize_emo_vec(emo_vector, apply_bias=True)
            
            if emo_text == "":
                emo_text = None
            
            for idx, sentence in enumerate(sentences):
                if lt_stop_event.is_set():
                    progress_info = i18n("生成已终止")
                    progress_percent = int((idx / total_sentences) * 100)
                    
                    result_df = pd.DataFrame([
                        {i18n("序号"): r["index"] + 1, i18n("句子内容"): r["sentence"], i18n("状态"): r["status"], i18n("耗时"): r["duration"]}
                        for r in sentence_results
                    ])
                    
                    yield (
                        gr.update(value=i18n("已终止"), visible=True),
                        gr.update(visible=False),
                        gr.update(value=progress_percent),
                        gr.update(value="", visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        result_df,
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(value=f"已终止，共 {total_sentences} 句，成功 {sum(1 for r in sentence_results if r['status'] == '成功')} 句", visible=True),
                        gr.update(value="", visible=True),
                        sentence_results,
                        output_dir
                    )
                    return
                
                while lt_pause_event.is_set() == False:
                    time.sleep(0.1)
                    if lt_stop_event.is_set():
                        progress_info = i18n("生成已终止")
                        progress_percent = int((idx / total_sentences) * 100)
                        
                        result_df = pd.DataFrame([
                            {i18n("序号"): r["index"] + 1, i18n("句子内容"): r["sentence"], i18n("状态"): r["status"], i18n("耗时"): r["duration"]}
                            for r in sentence_results
                        ])
                        
                        yield (
                            gr.update(value=i18n("已终止"), visible=True),
                            gr.update(visible=False),
                            gr.update(value=progress_percent),
                            gr.update(value="", visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            result_df,
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(value=f"已终止，共 {total_sentences} 句，成功 {sum(1 for r in sentence_results if r['status'] == '成功')} 句", visible=True),
                            gr.update(value="", visible=True),
                            sentence_results,
                            output_dir
                        )
                        return
                
                sentence = sentence.strip()
                if not sentence:
                    sentence_results.append({"index": idx, "sentence": sentence, "status": "跳过", "audio_path": None, "duration": "0s"})
                    
                    progress_info = i18n(f"第 {idx + 1}/{total_sentences} 句：跳过（空句子）")
                    progress_percent = int(((idx + 1) / total_sentences) * 100)
                    
                    result_df = pd.DataFrame([
                        {i18n("序号"): r["index"] + 1, i18n("句子内容"): r["sentence"], i18n("状态"): r["status"], i18n("耗时"): r["duration"]}
                        for r in sentence_results
                    ])
                    
                    yield (
                        gr.update(value=i18n("生成中"), visible=True),
                        gr.update(visible=True),
                        gr.update(value=progress_percent),
                        gr.update(value="", visible=True),
                        gr.update(visible=True),
                        gr.update(visible=True),
                        gr.update(visible=False),
                        result_df,
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(value=f"进度: {idx + 1}/{total_sentences}, 成功: {sum(1 for r in sentence_results if r['status'] == '成功')}", visible=True),
                        gr.update(value="", visible=True),
                        sentence_results,
                        output_dir
                    )
                    continue
                
                sentence_start_time = time.time()
                progress_info = i18n(f"正在生成第 {idx + 1}/{total_sentences} 句...")
                progress_percent = int((idx / total_sentences) * 100)
                
                result_df = pd.DataFrame([
                    {i18n("序号"): r["index"] + 1, i18n("句子内容"): r["sentence"], i18n("状态"): r["status"], i18n("耗时"): r["duration"]}
                    for r in sentence_results
                ])
                
                yield (
                    gr.update(value=i18n("生成中"), visible=True),
                    gr.update(visible=True),
                    gr.update(value=progress_percent),
                    gr.update(value=sentence, visible=True),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    result_df,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(value=f"进度: {idx + 1}/{total_sentences}, 正在生成中...", visible=True),
                    gr.update(value="", visible=True),
                    sentence_results,
                    output_dir
                )
                
                try:
                    output_path = os.path.join(output_dir, f"sentence_{idx + 1:04d}.wav")
                    
                    output = tts.infer(
                        spk_audio_prompt=prompt_audio,
                        text=sentence,
                        output_path=output_path,
                        emo_audio_prompt=emo_ref_path,
                        emo_alpha=emo_weight,
                        emo_vector=emo_vector,
                        use_emo_text=(emo_control_method == 3),
                        emo_text=emo_text,
                        use_random=emo_random,
                        verbose=cmd_args.verbose,
                        max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                        **kwargs
                    )
                    
                    sentence_duration = f"{time.time() - sentence_start_time:.1f}s"
                    
                    if output and os.path.exists(output):
                        audio_paths.append(output)
                        sentence_results.append({"index": idx, "sentence": sentence, "status": "成功", "audio_path": output, "duration": sentence_duration})
                    else:
                        sentence_results.append({"index": idx, "sentence": sentence, "status": "失败", "audio_path": None, "duration": sentence_duration})
                except Exception as e:
                    sentence_duration = f"{time.time() - sentence_start_time:.1f}s"
                    print(f"Error generating sentence {idx}: {e}")
                    sentence_results.append({"index": idx, "sentence": sentence, "status": f"错误: {str(e)[:50]}", "audio_path": None, "duration": sentence_duration})
                
                progress_info = i18n(f"第 {idx + 1}/{total_sentences} 句：{sentence_results[-1]['status']}")
                progress_percent = int(((idx + 1) / total_sentences) * 100)
                
                result_df = pd.DataFrame([
                    {i18n("序号"): r["index"] + 1, i18n("句子内容"): r["sentence"], i18n("状态"): r["status"], i18n("耗时"): r["duration"]}
                    for r in sentence_results
                ])
                
                yield (
                    gr.update(value=i18n("生成中"), visible=True),
                    gr.update(visible=True),
                    gr.update(value=progress_percent),
                    gr.update(value="", visible=True),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    result_df,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(value=f"进度: {idx + 1}/{total_sentences}, 成功: {sum(1 for r in sentence_results if r['status'] == '成功')}", visible=True),
                    sentence_results,
                    output_dir
                )
            
            total_duration = f"{time.time() - start_time:.1f}s"
            progress_info = i18n(f"生成完成！总耗时: {total_duration}")
            
            merged_path = None
            if audio_paths:
                merged_path = os.path.join(output_dir, "merged.wav")
                merge_audio_files(audio_paths, merged_path, 
                                  interval_silence_ms=interval_silence_ms,
                                  fade_in_ms=fade_in_ms,
                                  fade_out_ms=fade_out_ms)
            
            zip_path = None
            if audio_paths:
                zip_path = os.path.join(output_dir, "all_audio.zip")
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for path in audio_paths:
                        zf.write(path, os.path.basename(path))
            
            success_count = sum(1 for r in sentence_results if r["status"] == "成功")
            stats_text = i18n(f"共 {total_sentences} 句，成功 {success_count} 句，失败 {total_sentences - success_count} 句，总耗时 {total_duration}")
            
            results_df = pd.DataFrame([
                {i18n("序号"): r["index"] + 1, i18n("句子内容"): r["sentence"], i18n("状态"): r["status"], i18n("耗时"): r["duration"]}
                for r in sentence_results
            ])
            
            yield (
                gr.update(value=i18n("完成"), visible=True),
                gr.update(visible=False),
                gr.update(value=100),
                gr.update(value="", visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(value=results_df, visible=True),
                gr.update(value=merged_path, visible=True) if merged_path else gr.update(visible=False),
                gr.update(value=merged_path, visible=True) if merged_path else gr.update(visible=False),
                gr.update(value=zip_path, visible=True) if zip_path else gr.update(visible=False),
                gr.update(value=stats_text, visible=True),
                gr.update(value=progress_info, visible=True),
                sentence_results,
                output_dir
            )
        
        lt_load_btn.click(on_lt_load,
            inputs=[lt_input_text, lt_file_upload],
            outputs=[lt_raw_text_state, lt_segment_btn, lt_gen_button]
        )
        
        lt_segment_btn.click(on_lt_segment,
            inputs=[lt_raw_text_state],
            outputs=[lt_sentences_state, lt_sentences_df, lt_gen_button]
        )
        
        lt_delete_btn.click(on_lt_delete_sentence,
            inputs=[lt_sentences_state, lt_delete_idx_input],
            outputs=[lt_sentences_state, lt_sentences_df]
        )
        
        lt_merge_batch_btn.click(on_lt_merge_batch,
            inputs=[lt_sentences_state, lt_merge_batch_count],
            outputs=[lt_sentences_state, lt_sentences_df]
        )
        
        lt_merge_selected_btn.click(on_lt_merge_selected,
            inputs=[lt_sentences_state, lt_sentences_df],
            outputs=[lt_sentences_state, lt_sentences_df]
        )
        
        lt_retry_btn.click(on_lt_retry_single,
            inputs=[
                lt_retry_idx_input,
                lt_sentences_state,
                lt_sentence_results,
                lt_output_dir,
                lt_prompt_audio,
                lt_emo_control_method, lt_emo_upload, lt_emo_weight,
                lt_vec1, lt_vec2, lt_vec3, lt_vec4, lt_vec5, lt_vec6, lt_vec7, lt_vec8,
                lt_emo_text, lt_emo_random,
                lt_max_text_tokens_per_segment,
                *lt_advanced_params
            ],
            outputs=[lt_sentence_results, lt_merged_audio]
        )
        
        lt_sentences_df.change(on_lt_df_change,
            inputs=[lt_sentences_df],
            outputs=[lt_sentences_state]
        )
        
        lt_pause_button.click(toggle_lt_pause, outputs=[lt_pause_button, lt_status_label])
        lt_stop_button.click(stop_lt_generation, outputs=[lt_stop_button, lt_status_label])
        
        lt_emo_control_method.change(on_lt_method_change,
            inputs=[lt_emo_control_method],
            outputs=[lt_emotion_reference_group, lt_emotion_randomize_group,
                     lt_emotion_vector_group, lt_emo_text_group, lt_emo_weight_group]
        )
        
        lt_experimental_checkbox.change(on_lt_experimental_change,
            inputs=[lt_experimental_checkbox, lt_emo_control_method],
            outputs=[lt_emo_control_method]
        )
        
        lt_gen_button.click(on_lt_generate,
            inputs=[
                lt_sentences_state,
                lt_prompt_audio,
                lt_emo_control_method, lt_emo_upload, lt_emo_weight,
                lt_vec1, lt_vec2, lt_vec3, lt_vec4, lt_vec5, lt_vec6, lt_vec7, lt_vec8,
                lt_emo_text, lt_emo_random,
                lt_max_text_tokens_per_segment,
                *lt_advanced_params,
                lt_interval_silence, lt_fade_in, lt_fade_out
            ],
            outputs=[
                lt_status_label, lt_progress_row, lt_progress_bar, lt_current_sentence,
                lt_pause_button, lt_stop_button,
                lt_results_group, lt_results_area,
                lt_merged_audio, lt_download_merged, lt_download_all,
                lt_stats, lt_progress_stats,
                lt_sentence_results, lt_output_dir
            ]
        )


if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port)
