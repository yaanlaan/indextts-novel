import os
import re
import threading
import time
import zipfile

import gradio as gr
import pandas as pd

from .common import *
from .utils import *


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


def create_long_text_page(demo):
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
        
        with gr.Accordion(i18n("历史推理记录"), open=False):
            lt_history_list = gr.Dataframe(
                headers=[i18n("任务ID"), i18n("创建时间"), i18n("总句数"), i18n("成功数"), i18n("状态")],
                wrap=True,
                interactive=False,
            )
            lt_load_history_btn = gr.Button(i18n("加载记录"), variant="secondary")
            lt_select_history_btn = gr.Button(i18n("选中记录加载"), variant="primary")
            lt_resume_btn = gr.Button(i18n("继续生成"), variant="primary", visible=False)
            lt_merge_from_history_btn = gr.Button(i18n("合并已完成音频"), variant="secondary", visible=False)
        
        lt_results_area = gr.Dataframe(
            headers=[i18n("序号"), i18n("句子内容"), i18n("状态"), i18n("耗时")],
            wrap=True,
            visible=False
        )
        
        with gr.Row():
            lt_interval_silence = gr.Slider(
                label=i18n("句间静音间隔(ms)"),
                value=200, minimum=0, maximum=2000, step=50,
                info=i18n("每段音频之间的静音时长"),
                scale=2
            )
            lt_fade_in = gr.Slider(
                label=i18n("淡入时长(ms)"),
                value=0, minimum=0, maximum=1000, step=50,
                info=i18n("每段音频开始时的淡入效果"),
                scale=1
            )
            lt_fade_out = gr.Slider(
                label=i18n("淡出时长(ms)"),
                value=0, minimum=0, maximum=1000, step=50,
                info=i18n("每段音频结束时的淡出效果"),
                scale=1
            )
        
        with gr.Group(visible=False) as lt_results_group:
            gr.Markdown(i18n("生成结果"))
            with gr.Row():
                lt_merged_audio = gr.Audio(label=i18n("合并音频"))
                lt_download_merged = gr.DownloadButton(label=i18n("下载合并音频"))
                lt_download_all = gr.DownloadButton(label=i18n("下载全部单独音频"))
                lt_merge_audio_btn = gr.Button(i18n("重新合并音频"), variant="secondary")
            
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
        
        lt_selected_task_id = gr.State("")
        
        def on_lt_load_history():
            records = list_inference_records()
            if not records:
                return pd.DataFrame([], columns=[i18n("任务ID"), i18n("创建时间"), i18n("总句数"), i18n("成功数"), i18n("状态")])
            
            df = pd.DataFrame([
                {i18n("任务ID"): r["task_id"], i18n("创建时间"): r["created_at"],
                 i18n("总句数"): r["total_sentences"], i18n("成功数"): r["completed_sentences"],
                 i18n("状态"): r["status"]}
                for r in records
            ])
            return df
        
        def on_lt_select_history(evt: gr.SelectData):
            if evt is None or evt.value is None:
                gr.Warning(i18n("未选择记录"))
                return "", gr.update(visible=False), gr.update(visible=False)
            
            task_id = str(evt.value)
            record = load_inference_record(task_id)
            if not record:
                gr.Warning(i18n("记录不存在"))
                return "", gr.update(visible=False), gr.update(visible=False)
            
            has_pending = len(record["sentence_results"]) < len(record["sentences"])
            has_audio = any(r.get("status") == "成功" for r in record["sentence_results"])
            
            return (
                task_id,
                gr.update(visible=True) if has_pending else gr.update(visible=False),
                gr.update(visible=True) if has_audio else gr.update(visible=False),
            )
        
        def on_lt_load_selected_record(task_id):
            if not task_id:
                gr.Warning(i18n("请先选择记录"))
                return "", [], sentences_to_df([]), None, gr.update(visible=False), [], ""
            
            record = load_inference_record(task_id)
            if not record:
                gr.Warning(i18n("记录不存在"))
                return "", [], sentences_to_df([]), None, gr.update(visible=False), [], ""
            
            df = sentences_to_df(record["sentences"])
            
            has_pending = len(record["sentence_results"]) < len(record["sentences"])
            
            return (
                record["raw_text"],
                record["sentences"],
                df,
                record["prompt_audio"],
                gr.update(visible=True) if has_pending else gr.update(visible=False),
                record["sentence_results"],
                record["output_dir"],
            )
        
        def on_lt_merge_audio(sentence_results, output_dir, interval_silence_ms, fade_in_ms, fade_out_ms):
            if not sentence_results or not output_dir:
                gr.Warning(i18n("没有可合并的音频"))
                return None, None, None, None
            
            audio_paths = [r["audio_path"] for r in sentence_results if r.get("status") == "成功" and r.get("audio_path") and os.path.exists(r["audio_path"])]
            
            if not audio_paths:
                gr.Warning(i18n("没有成功生成的音频"))
                return None, None, None, None
            
            merged_path = os.path.join(output_dir, "merged.wav")
            merge_audio_files(audio_paths, merged_path,
                              interval_silence_ms=interval_silence_ms,
                              fade_in_ms=fade_in_ms,
                              fade_out_ms=fade_out_ms)
            
            zip_path = os.path.join(output_dir, "all_audio.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for path in audio_paths:
                    zf.write(path, os.path.basename(path))
            
            success_count = len(audio_paths)
            total_count = len(sentence_results)
            stats_text = i18n(f"合并完成！共 {total_count} 句，成功合并 {success_count} 句音频")
            
            return merged_path, merged_path, zip_path, stats_text
        
        def on_lt_resume(task_id):
            if not task_id:
                gr.Warning(i18n("请先选择记录"))
                return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
            
            record = load_inference_record(task_id)
            if not record:
                gr.Warning(i18n("记录不存在"))
                return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
            
            total_sentences = len(record["sentences"])
            completed = len(record["sentence_results"])
            
            if completed >= total_sentences:
                gr.Warning(i18n("该记录已全部完成"))
                return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
            
            progress_info = i18n(f"继续生成，已完成 {completed}/{total_sentences} 句...")
            
            yield (
                gr.update(value=i18n("生成中"), visible=True),
                gr.update(visible=True),
                gr.update(value=int((completed / total_sentences) * 100)),
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
                record["sentence_results"],
                record["output_dir"]
            )
            
            sentence_results = record["sentence_results"]
            audio_paths = [r["audio_path"] for r in sentence_results if r.get("status") == "成功" and r.get("audio_path")]
            start_idx = completed
            output_dir = record["output_dir"]
            sentences = record["sentences"]
            
            metadata = record.get("metadata", {})
            settings = metadata.get("settings", {})
            
            kwargs = {
                "do_sample": settings.get("do_sample", True),
                "top_p": settings.get("top_p", 0.8),
                "top_k": settings.get("top_k", 30) if settings.get("top_k", 30) > 0 else None,
                "temperature": settings.get("temperature", 0.8),
                "length_penalty": settings.get("length_penalty", 0.0),
                "num_beams": settings.get("num_beams", 3),
                "repetition_penalty": settings.get("repetition_penalty", 10.0),
                "max_mel_tokens": settings.get("max_mel_tokens", 1500),
            }
            
            emo_control_method = settings.get("emo_control_method", 0)
            emo_weight = settings.get("emo_weight", 0.65)
            max_text_tokens_per_segment = settings.get("max_text_tokens_per_segment", 120)
            use_random = settings.get("use_random", False)
            
            emo_audio_prompt = None
            emo_vector = None
            use_emo_text = False
            emo_text = None
            
            if emo_control_method == 0:
                emo_audio_prompt = None
            elif emo_control_method == 1:
                emo_ref_filename = metadata.get("emo_ref_audio", "emo_ref.wav")
                emo_audio_prompt = os.path.join(output_dir, emo_ref_filename) if os.path.exists(os.path.join(output_dir, emo_ref_filename)) else None
            elif emo_control_method == 2:
                emo_vector = settings.get("emo_vector")
                if emo_vector:
                    emo_vector = tts.normalize_emo_vec(emo_vector, apply_bias=True)
            elif emo_control_method == 3:
                use_emo_text = True
                emo_text = settings.get("emo_text")
            
            for idx in range(start_idx, total_sentences):
                if lt_stop_event.is_set():
                    save_progress(output_dir, sentence_results)
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
                        save_progress(output_dir, sentence_results)
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
                
                sentence = sentences[idx].strip()
                if not sentence:
                    sentence_results.append({"index": idx, "sentence": sentence, "status": "跳过", "audio_path": None, "duration": "0s"})
                    save_progress(output_dir, sentence_results)
                    
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
                    
                    prompt_audio_path = record.get("prompt_audio")
                    
                    output = tts.infer(
                        spk_audio_prompt=prompt_audio_path,
                        text=sentence,
                        output_path=output_path,
                        emo_audio_prompt=emo_audio_prompt,
                        emo_alpha=emo_weight,
                        emo_vector=emo_vector,
                        use_emo_text=use_emo_text,
                        emo_text=emo_text,
                        use_random=use_random,
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
                
                save_progress(output_dir, sentence_results)
                
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
            
            save_progress(output_dir, sentence_results)
            
            merged_path = None
            if audio_paths:
                merged_path = os.path.join(output_dir, "merged.wav")
                merge_audio_files(audio_paths, merged_path, 
                                  interval_silence_ms=settings.get("interval_silence_ms", 200),
                                  fade_in_ms=settings.get("fade_in_ms", 0),
                                  fade_out_ms=settings.get("fade_out_ms", 0))
            
            zip_path = None
            if audio_paths:
                zip_path = os.path.join(output_dir, "all_audio.zip")
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for path in audio_paths:
                        zf.write(path, os.path.basename(path))
            
            success_count = sum(1 for r in sentence_results if r["status"] == "成功")
            stats_text = i18n(f"继续生成完成！共 {total_sentences} 句，成功 {success_count} 句")
            
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
                gr.update(value=i18n("继续生成完成"), visible=True),
                sentence_results,
                output_dir
            )
        
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
            raw_text="",
        ):
            reset_lt_events()
            
            if not sentences:
                gr.Warning(i18n("请先处理文本得到分句结果"))
                return None, None, None, None, None, None, None, None, None, None, None
            
            if not prompt_audio:
                gr.Warning(i18n("请先上传音色参考音频"))
                return None, None, None, None, None, None, None, None, None, None, None
            
            total_sentences = len(sentences)
            
            task_id = str(int(time.time()))
            output_dir = os.path.join("outputs", "long_text", task_id)
            os.makedirs(output_dir, exist_ok=True)
            start_idx = 0
            sentence_results = []
            audio_paths = []
            progress_info = i18n(f"共 {total_sentences} 句，开始生成...")
            
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
            
            settings = {
                "emo_control_method": emo_control_method,
                "emo_weight": emo_weight,
                "do_sample": do_sample,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "length_penalty": length_penalty,
                "num_beams": num_beams,
                "repetition_penalty": repetition_penalty,
                "max_mel_tokens": max_mel_tokens,
                "interval_silence_ms": interval_silence_ms,
                "fade_in_ms": fade_in_ms,
                "fade_out_ms": fade_out_ms,
                "max_text_tokens_per_segment": max_text_tokens_per_segment,
                "use_random": emo_random,
                "emo_text": emo_text,
                "emo_vector": list(emo_vector) if emo_vector is not None else None,
            }
            save_inference_metadata(output_dir, raw_text, prompt_audio, sentences, settings, emo_ref_path)
            
            yield (
                gr.update(value=i18n("生成中"), visible=True),
                gr.update(visible=True),
                gr.update(value=int((start_idx / total_sentences) * 100)),
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
                sentence_results,
                output_dir
            )
            
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
            
            for idx in range(start_idx, total_sentences):
                sentence = sentences[idx]
                
                if lt_stop_event.is_set():
                    save_progress(output_dir, sentence_results)
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
                        save_progress(output_dir, sentence_results)
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
                    save_progress(output_dir, sentence_results)
                    
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
                
                save_progress(output_dir, sentence_results)
                
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
            save_progress(output_dir, sentence_results)
            
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
                lt_interval_silence, lt_fade_in, lt_fade_out,
                lt_raw_text_state,
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
        
        lt_load_history_btn.click(on_lt_load_history,
            outputs=[lt_history_list]
        )
        
        lt_history_list.select(on_lt_select_history,
            outputs=[lt_selected_task_id, lt_resume_btn, lt_merge_from_history_btn]
        )
        
        lt_select_history_btn.click(on_lt_load_selected_record,
            inputs=[lt_selected_task_id],
            outputs=[
                lt_input_text,
                lt_sentences_state,
                lt_sentences_df,
                lt_prompt_audio,
                lt_resume_btn,
                lt_sentence_results,
                lt_output_dir
            ]
        )
        
        lt_resume_btn.click(on_lt_resume,
            inputs=[lt_selected_task_id],
            outputs=[
                lt_status_label, lt_progress_row, lt_progress_bar, lt_current_sentence,
                lt_pause_button, lt_stop_button,
                lt_results_group, lt_results_area,
                lt_merged_audio, lt_download_merged, lt_download_all,
                lt_stats, lt_progress_stats,
                lt_sentence_results, lt_output_dir
            ]
        )
        
        lt_merge_from_history_btn.click(on_lt_merge_audio,
            inputs=[lt_sentence_results, lt_output_dir, lt_interval_silence, lt_fade_in, lt_fade_out],
            outputs=[lt_merged_audio, lt_download_merged, lt_download_all, lt_stats]
        )
        
        lt_merge_audio_btn.click(on_lt_merge_audio,
            inputs=[lt_sentence_results, lt_output_dir, lt_interval_silence, lt_fade_in, lt_fade_out],
            outputs=[lt_merged_audio, lt_download_merged, lt_download_all, lt_stats]
        )