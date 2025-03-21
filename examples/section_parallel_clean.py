from llm_dialog_manager import Agent
import multiprocessing as mp
from typing import List, Tuple
from functools import partial
import re

def clean_think_tags(text: str) -> str:
    """Remove <think>...</think> tags from text."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def split_transcript_with_time(full_transcript: str, chunk_duration=600):
    """Split timestamped transcript into chunks of specified duration."""
    # ... (previous implementation remains the same)
    pass

def process_chunk(chunk_data: Tuple[str, str, str], model_name: str = "DeepSeek-R1") -> Tuple[str, str, str]:
    """Process a single chunk in parallel"""
    start_time, end_time, chunk_text = chunk_data
    agent = Agent(model_name, memory_enabled=False)
    
    agent.add_message("system", "你是一个专业的摘要助理，会根据时间信息来输出摘要。")

    user_prompt = f"""
以下是时间段[{start_time} - {end_time}]的转录文本，请你总结重点并保留/提炼其中提到的关键时间点、主题要点等。

转录文本:
----------------
{chunk_text}
----------------

在输出中，请给出一个大纲形式，并在标题上标明本段的时间范围。例如：
[HH:MM:SS - HH:MM:SS] 主要内容
  - 主要论点
  - 重要引述(如有)
  - ...
"""
    agent.add_message("user", user_prompt)
    summary = agent.generate_response()
    # Clean any think tags from the summary
    summary = clean_think_tags(summary)
    return start_time, end_time, summary

def summarize_all_chunks_parallel(chunks: List[Tuple[str, str, str]], 
                                num_processes: int = None,
                                model_name: str = "DeepSeek-R1") -> str:
    """
    Parallel version of summarize_all_chunks using multiprocessing
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Create a process pool
    with mp.Pool(processes=num_processes) as pool:
        # Process chunks in parallel
        process_chunk_with_model = partial(process_chunk, model_name=model_name)
        chunk_summaries = pool.map(process_chunk_with_model, chunks)
    
    # Create final summary
    agent = Agent(model_name, memory_enabled=False)
    agent.add_message("system", "你是一个资深的文字编辑outline专家，擅长对多段摘要进行时序化合并。")

    merged = "\n\n".join(
        [f"【{i+1}】段时间 {st} - {et} 的摘要：\n{summ}" 
         for i, (st, et, summ) in enumerate(chunk_summaries)]
    )
    print(merged)
    
    user_prompt = f"""
下面是若干时间段的摘要，请你将它们进行合并，生成一个按时间顺序梳理的大纲，避免重复但保留关键信息。

要求：
1. 按照实际时间顺序（从最早到最晚）组织内容;
2. 在每个时段标题或段落中保留其 [开始时间 - 结束时间] 标识;
3. 若摘要间有重叠或重复，请进行适度精炼合并，但不要遗漏重点。
4. 输出时请使用分条、分段的 outline 格式，以便阅读。
5. 保证覆盖整个时间线，注意前一个章节的结束时间和第一个章节的开始时间不能有overlap也不能遗漏
6. 章节内部的段落不能包括该章节时间线以外的段落

------------
{merged}
------------

Format the outline using the following template (note that subtopics should be under the range of the section timestamp):

<outline>

## Section 1: Introduction (HH:MM:SS - HH:MM:SS)

- Topic introduction and overview
- Importance or relevance of the topic
- Brief explanation of what will be covered in the tutorial

## Section 2: [Main Topic 1] (HH:MM:SS - HH:MM:SS)

### Subtopic 1 (HH:MM:SS - HH:MM:SS)

- Key point 1
- Key point 2
- Example or demonstration

### Subtopic 2 (HH:MM:SS - HH:MM:SS)

- Key point 1
- Key point 2
- Example or demonstration

...
</outline>

另外请确保有合理的段落分配，每个段落包含的时间，不要太长也不要太短，适当即可
"""
    agent.add_message("user", user_prompt)
    final_outline = agent.generate_response()
    # Clean any think tags from the final outline
    final_outline = clean_think_tags(final_outline)
    return final_outline

if __name__ == "__main__":
    # Example usage
    sample_transcript = """
[00:00:00 -> 00:00:05] Welcome to this presentation
[00:00:05 -> 00:00:10] Today we'll discuss important topics
"""
    chunks = split_transcript_with_time(sample_transcript, chunk_duration=300)  # 5 minute chunks
    final_summary = summarize_all_chunks_parallel(chunks, num_processes=2)
    print(final_summary)
