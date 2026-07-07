import html
import json
import os
import re
import shutil
import time
import zipfile

import torch
import torchaudio

import gradio as gr


def create_warning_message(warning_text):
    return gr.HTML(f"<div style=\"padding: 0.5em 0.8em; border-radius: 0.5em; background: #ffa87d; color: #000; font-weight: bold\">{html.escape(warning_text)}</div>")


def create_experimental_warning_message():
    return create_warning_message('提示：此功能为实验版，结果尚不稳定，我们正在持续优化中。')


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


def apply_fade(wav, sampling_rate, fade_in_ms=0, fade_out_ms=0):
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
    sil_dur = int(sampling_rate * interval_silence_ms / 1000.0)
    silence = torch.zeros(1, sil_dur)
    
    wavs = []
    for path in audio_paths:
        if os.path.exists(path):
            wav, sr = torchaudio.load(path)
            if sr != sampling_rate:
                wav = torchaudio.transforms.Resample(sr, sampling_rate)(wav)
            wav = apply_fade(wav, sampling_rate, fade_in_ms, fade_out_ms)
            wavs.append(wav)
            if interval_silence_ms > 0 and path != audio_paths[-1]:
                wavs.append(silence)
    
    if wavs:
        merged = torch.cat(wavs, dim=1)
        torchaudio.save(output_path, merged, sampling_rate)
        return output_path
    return None


def download_audio(audio_path):
    if audio_path and os.path.exists(audio_path):
        return audio_path
    return None


def save_inference_metadata(output_dir, raw_text, prompt_audio_path, sentences, settings=None, emo_ref_path=None):
    metadata = {
        "task_id": os.path.basename(output_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_sentences": len(sentences),
        "settings": settings or {},
        "raw_text": raw_text,
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(output_dir, "original_text.txt"), "w", encoding="utf-8") as f:
        f.write(raw_text)
    
    with open(os.path.join(output_dir, "sentences.json"), "w", encoding="utf-8") as f:
        json.dump({"sentences": sentences}, f, ensure_ascii=False, indent=2)
    
    if prompt_audio_path and os.path.exists(prompt_audio_path):
        ext = os.path.splitext(prompt_audio_path)[1]
        dest_path = os.path.join(output_dir, f"prompt_audio{ext}")
        shutil.copy2(prompt_audio_path, dest_path)
        metadata["prompt_audio"] = f"prompt_audio{ext}"
    
    if emo_ref_path and os.path.exists(emo_ref_path):
        ext = os.path.splitext(emo_ref_path)[1]
        dest_path = os.path.join(output_dir, f"emo_ref{ext}")
        shutil.copy2(emo_ref_path, dest_path)
        metadata["emo_ref_audio"] = f"emo_ref{ext}"
    
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    return metadata


def save_progress(output_dir, sentence_results):
    progress_data = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sentence_results": sentence_results,
    }
    with open(os.path.join(output_dir, "progress.json"), "w", encoding="utf-8") as f:
        json.dump(progress_data, f, ensure_ascii=False, indent=2)


def load_inference_record(task_id):
    output_dir = os.path.join("outputs", "long_text", task_id)
    if not os.path.exists(output_dir):
        return None
    
    record = {"output_dir": output_dir}
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            record.update(metadata)
    
    if "prompt_audio" in record and record["prompt_audio"]:
        record["prompt_audio"] = os.path.join(output_dir, record["prompt_audio"])
    
    if "emo_ref_audio" in record and record["emo_ref_audio"]:
        record["emo_ref_audio"] = os.path.join(output_dir, record["emo_ref_audio"])
    
    if "raw_text" not in record:
        text_path = os.path.join(output_dir, "original_text.txt")
        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as f:
                record["raw_text"] = f.read()
    
    sentences_path = os.path.join(output_dir, "sentences.json")
    if os.path.exists(sentences_path):
        with open(sentences_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            record["sentences"] = data.get("sentences", [])
    
    progress_path = os.path.join(output_dir, "progress.json")
    if os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            record["sentence_results"] = data.get("sentence_results", [])
    
    return record


def list_inference_records():
    records = []
    base_dir = os.path.join("outputs", "long_text")
    
    if not os.path.exists(base_dir):
        return records
    
    for task_id in sorted(os.listdir(base_dir), reverse=True):
        output_dir = os.path.join(base_dir, task_id)
        if not os.path.isdir(output_dir):
            continue
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            continue
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        progress_path = os.path.join(output_dir, "progress.json")
        if os.path.exists(progress_path):
            with open(progress_path, "r", encoding="utf-8") as f:
                progress_data = json.load(f)
                sentence_results = progress_data.get("sentence_results", [])
                completed = sum(1 for r in sentence_results if r.get("status") == "成功")
                total = metadata.get("total_sentences", len(sentence_results))
                status = "完成" if completed == total else f"{completed}/{total}"
        else:
            completed = 0
            total = metadata.get("total_sentences", 0)
            status = "0/0"
        
        records.append({
            "task_id": task_id,
            "created_at": metadata.get("created_at", ""),
            "total_sentences": total,
            "completed_sentences": completed,
            "status": status,
        })
    
    return records