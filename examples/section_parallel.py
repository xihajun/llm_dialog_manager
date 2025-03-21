from llm_dialog_manager import Agent
import multiprocessing as mp
from typing import List, Tuple
from functools import partial
import re

def split_transcript_with_time(full_transcript: str, chunk_duration=600):
    """
    Split timestamped transcript into chunks of specified duration.
    
    Args:
        full_transcript (str): Input transcript with timestamps in format [MM:SS -> MM:SS] or [HH:MM:SS -> HH:MM:SS]
        chunk_duration (int): Duration of each chunk in seconds (default: 600 seconds = 10 minutes)
    
    Returns:
        List[Tuple[str, str, str]]: List of (start_time, end_time, chunk_text) tuples
    """
    def time_to_seconds(time_str):
        """Convert HH:MM:SS or MM:SS format to seconds"""
        parts = time_str.strip().split(':')
        if len(parts) == 2:  # MM:SS format
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:  # HH:MM:SS format
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        else:
            raise ValueError(f"Invalid time format: {time_str}")

    def seconds_to_time(seconds, use_hours=True):
        """Convert seconds to HH:MM:SS or MM:SS format"""
        if use_hours:
            hours = seconds // 3600
            seconds %= 3600
            minutes = seconds // 60
            seconds %= 60
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            minutes = seconds // 60
            seconds %= 60
            return f"{minutes:02d}:{seconds:02d}"

    # Rest of the split_transcript_with_time function remains the same
    chunks = []
    current_chunk = []
    current_start = None
    current_end = None
    
    lines = full_transcript.strip().split('\n')
    
    for line in lines:
        if not line.strip():
            continue
            
        try:
            timestamp_part = line[line.find('[')+1:line.find(']')]
            start_time, end_time = map(str.strip, timestamp_part.split('->'))
            text = line[line.find(']')+1:].strip()
            
            start_seconds = time_to_seconds(start_time)
            end_seconds = time_to_seconds(end_time)
            
            if current_start is None:
                current_start = start_time
            
            if current_chunk and time_to_seconds(start_time) - time_to_seconds(current_start) >= chunk_duration:
                chunks.append((current_start, current_end, '\n'.join(current_chunk)))
                current_chunk = []
                current_start = start_time
            
            current_chunk.append(line)
            current_end = end_time
            
        except Exception as e:
            print(f"Warning: Could not parse line: {line}")
            continue
    
    if current_chunk:
        chunks.append((current_start, current_end, '\n'.join(current_chunk)))
    
    return chunks

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
    summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)
    return start_time, end_time, summary

def summarize_all_chunks_parallel(chunks: List[Tuple[str, str, str]], 
                                num_processes: int = None,
                                model_name: str = "DeepSeek-R1") -> str:
    """
    Parallel version of summarize_all_chunks using multiprocessing
    
    Args:
        chunks: List of (start_time, end_time, text) tuples
        num_processes: Number of parallel processes to use. If None, uses CPU count
        model_name: Name of the model to use for summarization
    
    Returns:
        str: Final outline combining all summaries
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
    
    return final_outline

# Example usage:
if __name__ == "__main__":
    # Example transcript
    sample_transcript = """
[00:00 -> 00:04]  hi everyone so in this video I'd like us
[00:02 -> 00:06]  to cover the process of tokenization in
[00:04 -> 00:08]  large language models now you see here
[00:06 -> 00:10]  that I have a set face and that's
[00:08 -> 00:11]  because uh tokenization is my least
[00:10 -> 00:13]  favorite part of working with large
[00:11 -> 00:15]  language models but unfortunately it is
[00:13 -> 00:17]  necessary to understand in some detail
[00:15 -> 00:19]  because it it is fairly hairy gnarly and
[00:17 -> 00:21]  there's a lot of hidden foot guns to be
[00:19 -> 00:24]  aware of and a lot of oddness with large
[00:21 -> 00:26]  language models typically traces back to
[00:24 -> 00:28]  tokenization so what is
[00:26 -> 00:31]  tokenization now in my previous video
[00:28 -> 00:33]  Let's Build GPT from scratch uh we
[00:31 -> 00:35]  actually already did tokenization but we
[00:33 -> 00:37]  did a very naive simple version of
[00:35 -> 00:40]  tokenization so when you go to the
[00:37 -> 00:43]  Google colab for that video uh you see
[00:40 -> 00:45]  here that we loaded our training set and
[00:43 -> 00:48]  our training set was this uh Shakespeare
[00:45 -> 00:49]  uh data set now in the beginning the
[00:48 -> 00:52]  Shakespeare data set is just a large
[00:49 -> 00:54]  string in Python it's just text and so
[00:52 -> 00:58]  the question is how do we plug text into
[00:54 -> 01:01]  large language models and in this case
[00:58 -> 01:03]  here we created a vocabulary of 65
[01:01 -> 01:05]  possible characters that we saw occur in
[01:03 -> 01:07]  this string these were the possible
[01:05 -> 01:10]  characters and we saw that there are 65
[01:07 -> 01:13]  of them and then we created a a lookup
[01:10 -> 01:16]  table for converting from every possible
[01:13 -> 01:17]  character a little string piece into a
[01:16 -> 01:20]  token an
[01:17 -> 01:23]  integer so here for example we tokenized
[01:20 -> 01:24]  the string High there and we received
[01:23 -> 01:27]  this sequence of
[01:24 -> 01:29]  tokens and here we took the first 1,000
[01:27 -> 01:32]  characters of our data set and we
[01:29 -> 01:34]  encoded it into tokens and because it is
[01:32 -> 01:38]  this is character level we received
[01:34 -> 01:40]  1,000 tokens in a sequence so token 18
[01:38 -> 01:43]  47
[01:40 -> 01:45]  Etc now later we saw that the way we
[01:43 -> 01:48]  plug these tokens into the language
[01:45 -> 01:51]  model is by using an embedding
[01:48 -> 01:53]  table and so basically if we have 65
[01:51 -> 01:56]  possible tokens then this embedding
[01:53 -> 01:58]  table is going to have 65 rows and
[01:56 -> 01:59]  roughly speaking we're taking the
[01:58 -> 02:01]  integer associated with every single
[01:59 -> 02:04]  sing Le token we're using that as a
[02:01 -> 02:06]  lookup into this table and we're
[02:04 -> 02:09]  plucking out the corresponding row and
[02:06 -> 02:10]  this row is a uh is trainable parameters
[02:09 -> 02:12]  that we're going to train using back
[02:10 -> 02:15]  propagation and this is the vector that
[02:12 -> 02:16]  then feeds into the Transformer um and
[02:15 -> 02:18]  that's how the Transformer Ser of
[02:16 -> 02:21]  perceives every single
[02:18 -> 02:23]  token so here we had a very naive
[02:21 -> 02:25]  tokenization process that was a
[02:23 -> 02:27]  character level tokenizer but in
[02:25 -> 02:28]  practice in state-ofthe-art uh language
[02:27 -> 02:30]  models people use a lot more complicated
[02:28 -> 02:34]  schemes unfortunately
[02:30 -> 02:36]  uh for constructing these uh token
[02:34 -> 02:38]  vocabularies so we're not dealing on the
[02:36 -> 02:41]  Character level we're dealing on chunk
[02:38 -> 02:43]  level and the way these um character
[02:41 -> 02:45]  chunks are constructed is using
[02:43 -> 02:46]  algorithms such as for example the bik
[02:45 -> 02:51]  pair in coding algorithm which we're
[02:46 -> 02:52]  going to go into in detail um and cover
[02:51 -> 02:54]  in this video I'd like to briefly show
[02:52 -> 02:56]  you the paper that introduced a bite
[02:54 -> 02:58]  level encoding as a mechanism for
[02:56 -> 03:00]  tokenization in the context of large
[02:58 -> 03:02]  language models and I would say that
[03:00 -> 03:05]  that's probably the gpt2 paper and if
[03:02 -> 03:07]  you scroll down here to the section
[03:05 -> 03:09]  input representation this is where they
[03:07 -> 03:10]  cover tokenization the kinds of
[03:09 -> 03:13]  properties that you'd like the
[03:10 -> 03:14]  tokenization to have and they conclude
[03:13 -> 03:17]  here that they're going to have a
[03:14 -> 03:20]  tokenizer where you have a vocabulary of
[03:17 -> 03:24]  50,2 57 possible
[03:20 -> 03:27]  tokens and the context size is going to
[03:24 -> 03:29]  be 1,24 tokens so in the in in the
[03:27 -> 03:30]  attention layer of the Transformer
[03:29 -> 03:32]  neural network
[03:30 -> 03:34]  every single token is attending to the
[03:32 -> 03:37]  previous tokens in the sequence and it's
[03:34 -> 03:40]  going to see up to 1,24 tokens so tokens
[03:37 -> 03:43]  are this like fundamental unit um the
[03:40 -> 03:44]  atom of uh large language models if you
[03:43 -> 03:47]  will and everything is in units of
[03:44 -> 03:48]  tokens everything is about tokens and
[03:47 -> 03:51]  tokenization is the process for
[03:48 -> 03:54]  translating strings or text into
[03:51 -> 03:56]  sequences of tokens and uh vice versa
[03:54 -> 03:58]  when you go into the Llama 2 paper as
[03:56 -> 04:01]  well I can show you that when you search
[03:58 -> 04:03]  token you're going to get get 63 hits um
[04:01 -> 04:05]  and that's because tokens are again
[04:03 -> 04:06]  pervasive so here they mentioned that
[04:05 -> 04:08]  they trained on two trillion tokens of
[04:06 -> 04:11]  data and so
[04:08 -> 04:13]  on so we're going to build our own
[04:11 -> 04:15]  tokenizer luckily the bite be encoding
[04:13 -> 04:16]  algorithm is not uh that super
[04:15 -> 04:18]  complicated and we can build it from
[04:16 -> 04:20]  scratch ourselves and we'll see exactly
[04:18 -> 04:22]  how this works before we dive into code
[04:20 -> 04:24]  I'd like to give you a brief Taste of
[04:22 -> 04:26]  some of the complexities that come from
[04:24 -> 04:27]  the tokenization because I just want to
[04:26 -> 04:29]  make sure that we motivate it
[04:27 -> 04:32]  sufficiently for why we are doing all
[04:29 -> 04:34]  this and why this is so gross so
[04:32 -> 04:36]  tokenization is at the heart of a lot of
[04:34 -> 04:37]  weirdness in large language models and I
[04:36 -> 04:40]  would advise that you do not brush it
[04:37 -> 04:42]  off a lot of the issues that may look
[04:40 -> 04:44]  like just issues with the new network
[04:42 -> 04:46]  architecture or the large language model
[04:44 -> 04:49]  itself are actually issues with the
[04:46 -> 04:51]  tokenization and fundamentally Trace uh
[04:49 -> 04:54]  back to it so if you've noticed any
[04:51 -> 04:56]  issues with large language models can't
[04:54 -> 04:57]  you know not able to do spelling tasks
[04:56 -> 05:00]  very easily that's usually due to
[04:57 -> 05:02]  tokenization simple string processing
[05:00 -> 05:03]  can be difficult for the large language
[05:02 -> 05:06]  model to perform
[05:03 -> 05:08]  natively uh non-english languages can
[05:06 -> 05:09]  work much worse and to a large extent
[05:08 -> 05:11]  this is due to
[05:09 -> 05:14]  tokenization sometimes llms are bad at
[05:11 -> 05:15]  simple arithmetic also can trace be
[05:14 -> 05:17]  traced to
[05:15 -> 05:19]  tokenization uh gbt2 specifically would
[05:17 -> 05:22]  have had quite a bit more issues with
[05:19 -> 05:24]  python than uh future versions of it due
[05:22 -> 05:25]  to tokenization there's a lot of other
[05:24 -> 05:27]  issues maybe you've seen weird warnings
[05:25 -> 05:30]  about a trailing whites space this is a
[05:27 -> 05:33]  tokenization issue um
[05:30 -> 05:35]  if you had asked GPT earlier about solid
[05:33 -> 05:37]  gold Magikarp and what it is you would
[05:35 -> 05:39]  see the llm go totally crazy and it
[05:37 -> 05:41]  would start going off about a completely
[05:39 -> 05:43]  unrelated tangent topic maybe you've
[05:41 -> 05:45]  been told to use yl over Json in
[05:43 -> 05:47]  structure data all of that has to do
[05:45 -> 05:49]  with tokenization so basically
[05:47 -> 05:51]  tokenization is at the heart of many
[05:49 -> 05:54]  issues I will look back around to these
[05:51 -> 05:56]  at the end of the video but for now let
[05:54 -> 05:59]  me just um skip over it a little bit and
[05:56 -> 06:02]  let's go to this web app um the Tik
[05:59 -> 06:04]  tokenizer bell.app so I have it loaded
[06:02 -> 06:06]  here and what I like about this web app
[06:04 -> 06:09]  is that tokenization is running a sort
[06:06 -> 06:11]  of live in your browser in JavaScript so
[06:09 -> 06:14]  you can just type here stuff hello world
[06:11 -> 06:18]  and the whole string
[06:14 -> 06:20]  rokenes so here what we see on uh the
[06:18 -> 06:22]  left is a string that you put in on the
[06:20 -> 06:24]  right we're currently using the gpt2
[06:22 -> 06:27]  tokenizer we see that this string that I
[06:24 -> 06:30]  pasted here is currently tokenizing into
[06:27 -> 06:32]  300 tokens and here they are sort of uh
[06:30 -> 06:35]  shown explicitly in different colors for
[06:32 -> 06:38]  every single token so for example uh
[06:35 -> 06:40]  this word tokenization became two tokens
[06:38 -> 06:43]  the token
[06:40 -> 06:43]  3,642 and
[06:44 -> 06:51]  1,634 the token um space is is token 318
[06:50 -> 06:54]  so be careful on the bottom you can show
[06:51 -> 06:57]  white space and keep in mind that there
[06:54 -> 06:59]  are spaces and uh sln new line
[06:57 -> 07:01]  characters in here but you can hide them
[06:59 -> 07:06]  for
[07:01 -> 07:11]  clarity the token space at is token 379
[07:06 -> 07:12]  the to the Token space the is 262 Etc so
[07:11 -> 07:15]  you notice here that the space is part
[07:12 -> 07:18]  of that uh token
[07:15 -> 07:21]  chunk now so this is kind of like how
[07:18 -> 07:24]  our English sentence broke up and that
[07:21 -> 07:26]  seems all well and good now now here I
[07:24 -> 07:31]  put in some arithmetic so we see that uh
[07:26 -> 07:34]  the token 127 Plus and then token six
[07:31 -> 07:36]  space 6 followed by 77 so what's
[07:34 -> 07:38]  happening here is that 127 is feeding in
[07:36 -> 07:42]  as a single token into the large
[07:38 -> 07:44]  language model but the um number 677
[07:42 -> 07:47]  will actually feed in as two separate
[07:44 -> 07:50]  tokens and so the large language model
[07:47 -> 07:53]  has to sort of um take account of that
[07:50 -> 07:56]  and process it correctly in its Network
[07:53 -> 07:57]  and see here 804 will be broken up into
[07:56 -> 07:59]  two tokens and it's is all completely
[07:57 -> 08:02]  arbitrary and here I have another
[07:59 -> 08:03]  example of four-digit numbers and they
[08:02 -> 08:05]  break up in a way that they break up and
[08:03 -> 08:08]  it's totally arbitrary sometimes you
[08:05 -> 08:10]  have um multiple digits single token
[08:08 -> 08:12]  sometimes you have individual digits as
[08:10 -> 08:14]  many tokens and it's all kind of pretty
[08:12 -> 08:17]  arbitrary and coming out of the
[08:14 -> 08:21]  tokenizer here's another example we have
[08:17 -> 08:22]  the string egg and you see here that
[08:21 -> 08:24]  this became two
[08:22 -> 08:27]  tokens but for some reason when I say I
[08:24 -> 08:30]  have an egg you see when it's a space
[08:27 -> 08:33]  egg it's two token it's sorry it's a
[08:30 -> 08:34]  single token so just egg by itself in
[08:33 -> 08:37]  the beginning of a sentence is two
[08:34 -> 08:40]  tokens but here as a space egg is
[08:37 -> 08:44]  suddenly a single token uh for the exact
[08:40 -> 08:46]  same string okay here lowercase egg
[08:44 -> 08:47]  turns out to be a single token and in
[08:46 -> 08:49]  particular notice that the color is
[08:47 -> 08:51]  different so this is a different token
[08:49 -> 08:54]  so this is case sensitive and of course
[08:51 -> 08:57]  a capital egg would also be different
[08:54 -> 09:00]  tokens and again um this would be two
[08:57 -> 09:02]  tokens arbitrarily so so for the same
[09:00 -> 09:03]  concept egg depending on if it's in the
[09:02 -> 09:06]  beginning of a sentence at the end of a
[09:03 -> 09:08]  sentence lowercase uppercase or mixed
[09:06 -> 09:10]  all this will be uh basically very
[09:08 -> 09:12]  different tokens and different IDs and
[09:10 -> 09:13]  the language model has to learn from raw
[09:12 -> 09:15]  data from all the internet text that
[09:13 -> 09:17]  it's going to be training on that these
[09:15 -> 09:19]  are actually all the exact same concept
[09:17 -> 09:21]  and it has to sort of group them in the
[09:19 -> 09:22]  parameters of the neural network and
[09:21 -> 09:24]  understand just based on the data
[09:22 -> 09:27]  patterns that these are all very similar
[09:24 -> 09:30]  but maybe not almost exactly similar but
[09:27 -> 09:32]  but very very similar
[09:30 -> 09:35]  um after the EG demonstration here I
[09:32 -> 09:41]  have um an introduction from open a eyes
[09:35 -> 09:44]  chbt in Korean so manaso Pang uh Etc uh
[09:41 -> 09:47]  so this is in Korean and the reason I
[09:44 -> 09:51]  put this here is because you'll notice
[09:47 -> 09:54]  that um non-english languages work
[09:51 -> 09:55]  slightly worse in Chachi part of this is
[09:54 -> 09:58]  because of course the training data set
[09:55 -> 09:59]  for Chachi is much larger for English
[09:58 -> 10:01]  and for everything else but the same is
[09:59 -> 10:04]  true not just for the large language
[10:01 -> 10:05]  model itself but also for the tokenizer
[10:04 -> 10:07]  so when we train the tokenizer we're
[10:05 -> 10:09]  going to see that there's a training set
[10:07 -> 10:11]  as well and there's a lot more English
[10:09 -> 10:13]  than non-english and what ends up
[10:11 -> 10:16]  happening is that we're going to have a
[10:13 -> 10:19]  lot more longer tokens for
[10:16 -> 10:21]  English so how do I put this if you have
[10:19 -> 10:23]  a single sentence in English and you
[10:21 -> 10:25]  tokenize it you might see that it's 10
[10:23 -> 10:27]  tokens or something like that but if you
[10:25 -> 10:29]  translate that sentence into say Korean
[10:27 -> 10:30]  or Japanese or something else you'll
[10:29 -> 10:33]  typically see that the number of tokens
[10:30 -> 10:36]  used is much larger and that's because
[10:33 -> 10:38]  the chunks here are a lot more broken up
[10:36 -> 10:41]  so we're using a lot more tokens for the
[10:38 -> 10:43]  exact same thing and what this does is
[10:41 -> 10:46]  it bloats up the sequence length of all
[10:43 -> 10:48]  the documents so you're using up more
[10:46 -> 10:49]  tokens and then in the attention of the
[10:48 -> 10:51]  Transformer when these tokens try to
[10:49 -> 10:55]  attend each other you are running out of
[10:51 -> 10:57]  context um in the maximum context length
[10:55 -> 11:01]  of that Transformer and so basically all
[10:57 -> 11:03]  the non-english text is stretched out
[11:01 -> 11:05]  from the perspective of the Transformer
[11:03 -> 11:07]  and this just has to do with the um
[11:05 -> 11:10]  trainings that used for the tokenizer
[11:07 -> 11:12]  and the tokenization itself so it will
[11:10 -> 11:14]  create a lot bigger tokens and a lot
[11:12 -> 11:16]  larger groups in English and it will
[11:14 -> 11:19]  have a lot of little boundaries for all
[11:16 -> 11:21]  the other non-english text um so if we
[11:19 -> 11:23]  translated this into English it would be
[11:21 -> 11:25]  significantly fewer
[11:23 -> 11:28]  tokens the final example I have here is
[11:25 -> 11:31]  a little snippet of python for doing FS
[11:28 -> 11:34]  buuz and what I'd like you to notice is
[11:31 -> 11:37]  look all these individual spaces are all
[11:34 -> 11:42]  separate tokens they are token
[11:37 -> 11:45]  220 so uh 220 220 220 220 and then space
[11:42 -> 11:46]  if is a single token and so what's going
[11:45 -> 11:49]  on here is that when the Transformer is
[11:46 -> 11:52]  going to consume or try to uh create
[11:49 -> 11:54]  this text it needs to um handle all
[11:52 -> 11:56]  these spaces individually they all feed
[11:54 -> 11:59]  in one by one into the entire
[11:56 -> 12:01]  Transformer in the sequence and so this
[11:59 -> 12:04]  is being extremely wasteful tokenizing
[12:01 -> 12:07]  it in this way and so as a result of
[12:04 -> 12:08]  that gpt2 is not very good with python
[12:07 -> 12:10]  and it's not anything to do with coding
[12:08 -> 12:12]  or the language model itself it's just
[12:10 -> 12:15]  that if he use a lot of indentation
[12:12 -> 12:17]  using space in Python like we usually do
[12:15 -> 12:19]  uh you just end up bloating out all the
[12:17 -> 12:21]  text and it's separated across way too
[12:19 -> 12:22]  much of the sequence and we are running
[12:21 -> 12:24]  out of the context length in the
[12:22 -> 12:25]  sequence uh that's roughly speaking
[12:24 -> 12:27]  what's what's happening we're being way
[12:25 -> 12:29]  too wasteful we're taking up way too
[12:27 -> 12:31]  much token space now we can also scroll
[12:29 -> 12:34]  up here and we can change the tokenizer
[12:31 -> 12:36]  so note here that gpt2 tokenizer creates
[12:34 -> 12:39]  a token count of 300 for this string
[12:36 -> 12:41]  here we can change it to CL 100K base
[12:39 -> 12:44]  which is the GPT for tokenizer and we
[12:41 -> 12:46]  see that the token count drops to 185 so
[12:44 -> 12:49]  for the exact same string we are now
[12:46 -> 12:51]  roughly having the number of tokens and
[12:49 -> 12:54]  roughly speaking this is because uh the
[12:51 -> 12:56]  number of tokens in the GPT 4 tokenizer
[12:54 -> 12:58]  is roughly double that of the number of
[12:56 -> 13:01]  tokens in the gpt2 tokenizer so we went
[12:58 -> 13:03]  went from roughly 50k to roughly 100K
[13:01 -> 13:06]  now you can imagine that this is a good
[13:03 -> 13:10]  thing because the same text is now
[13:06 -> 13:12]  squished into half as many tokens so uh
[13:10 -> 13:15]  this is a lot denser input to the
[13:12 -> 13:17]  Transformer and in the Transformer every
[13:15 -> 13:18]  single token has a finite number of
[13:17 -> 13:20]  tokens before it that it's going to pay
[13:18 -> 13:23]  attention to and so what this is doing
[13:20 -> 13:26]  is we're roughly able to see twice as
[13:23 -> 13:29]  much text as a context for what token to
[13:26 -> 13:30]  predict next uh because of this change
[13:29 -> 13:33]  but of course just increasing the number
[13:30 -> 13:35]  of tokens is uh not strictly better
[13:33 -> 13:36]  infinitely uh because as you increase
[13:35 -> 13:39]  the number of tokens now your embedding
[13:36 -> 13:41]  table is um sort of getting a lot larger
[13:39 -> 13:42]  and also at the output we are trying to
[13:41 -> 13:45]  predict the next token and there's the
[13:42 -> 13:46]  soft Max there and that grows as well
[13:45 -> 13:48]  we're going to go into more detail later
[13:46 -> 13:51]  on this but there's some kind of a Sweet
[13:48 -> 13:52]  Spot somewhere where you have a just
[13:51 -> 13:53]  right number of tokens in your
[13:52 -> 13:56]  vocabulary where everything is
[13:53 -> 13:58]  appropriately dense and still fairly
[13:56 -> 14:00]  efficient now one thing I would like you
[13:58 -> 14:03]  to note specifically for the gp4
[14:00 -> 14:05]  tokenizer is that the handling of the
[14:03 -> 14:08]  white space for python has improved a
[14:05 -> 14:10]  lot you see that here these four spaces
[14:08 -> 14:13]  are represented as one single token for
[14:10 -> 14:16]  the three spaces here and then the token
[14:13 -> 14:18]  SPF and here seven spaces were all
[14:16 -> 14:20]  grouped into a single token so we're
[14:18 -> 14:21]  being a lot more efficient in how we
[14:20 -> 14:23]  represent Python and this was a
[14:21 -> 14:27]  deliberate Choice made by open aai when
[14:23 -> 14:29]  they designed the gp4 tokenizer and they
[14:27 -> 14:32]  group a lot more space into a single
[14:29 -> 14:35]  character what this does is this
[14:32 -> 14:38]  densifies Python and therefore we can
[14:35 -> 14:39]  attend to more code before it when we're
[14:38 -> 14:42]  trying to predict the next token in the
[14:39 -> 14:45]  sequence and so the Improvement in the
[14:42 -> 14:47]  python coding ability from gbt2 to gp4
[14:45 -> 14:48]  is not just a matter of the language
[14:47 -> 14:50]  model and the architecture and the
[14:48 -> 14:52]  details of the optimization but a lot of
[14:50 -> 14:54]  the Improvement here is also coming from
[14:52 -> 14:56]  the design of the tokenizer and how it
[14:54 -> 14:59]  groups characters into tokens okay so
[14:56 -> 15:01]  let's now start writing some code
[14:59 -> 15:03]  so remember what we want to do we want
[15:01 -> 15:05]  to take strings and feed them into
[15:03 -> 15:08]  language models for that we need to
[15:05 -> 15:12]  somehow tokenize strings into some
[15:08 -> 15:14]  integers in some fixed vocabulary and
[15:12 -> 15:16]  then we will use those integers to make
[15:14 -> 15:18]  a look up into a lookup table of vectors
[15:16 -> 15:21]  and feed those vectors into the
[15:18 -> 15:22]  Transformer as an input now the reason
[15:21 -> 15:24]  this gets a little bit tricky of course
[15:22 -> 15:26]  is that we don't just want to support
[15:24 -> 15:28]  the simple English alphabet we want to
[15:26 -> 15:31]  support different kinds of languages so
[15:28 -> 15:33]  this is anango in Korean which is hello
[15:31 -> 15:34]  and we also want to support many kinds
[15:33 -> 15:37]  of special characters that we might find
[15:34 -> 15:41]  on the internet for example
[15:37 -> 15:42]  Emoji so how do we feed this text into
[15:41 -> 15:44]  uh
[15:42 -> 15:46]  Transformers well how's the what is this
[15:44 -> 15:49]  text anyway in Python so if you go to
[15:46 -> 15:51]  the documentation of a string in Python
[15:49 -> 15:54]  you can see that strings are immutable
[15:51 -> 15:57]  sequences of Unicode code
[15:54 -> 16:01]  points okay what are Unicode code points
[15:57 -> 16:04]  we can go to PDF so Unicode code points
[16:01 -> 16:07]  are defined by the Unicode Consortium as
[16:04 -> 16:09]  part of the Unicode standard and what
[16:07 -> 16:11]  this is really is that it's just a
[16:09 -> 16:14]  definition of roughly 150,000 characters
[16:11 -> 16:17]  right now and roughly speaking what they
[16:14 -> 16:19]  look like and what integers um represent
[16:17 -> 16:22]  those characters so it says 150,000
[16:19 -> 16:24]  characters across 161 scripts as of
[16:22 -> 16:26]  right now so if you scroll down here you
[16:24 -> 16:28]  can see that the standard is very much
[16:26 -> 16:30]  alive the latest standard 15.1 in
[16:28 -> 16:33]  September
[16:30 -> 16:36]  2023 and basically this is just a way to
[16:33 -> 16:39]  define lots of types of
[16:36 -> 16:41]  characters like for example all these
[16:39 -> 16:44]  characters across different scripts so
[16:41 -> 16:45]  the way we can access the unic code code
[16:44 -> 16:48]  Point given Single Character is by using
[16:45 -> 16:51]  the or function in Python so for example
[16:48 -> 16:54]  I can pass in Ord of H and I can see
[16:51 -> 16:56]  that for the Single Character H the unic
[16:54 -> 17:00]  code code point is
[16:56 -> 17:02]  104 okay um but this can be arbitr
[17:00 -> 17:04]  complicated so we can take for example
[17:02 -> 17:06]  our Emoji here and we can see that the
[17:04 -> 17:10]  code point for this one is
[17:06 -> 17:13]  128,000 or we can take
[17:10 -> 17:16]  un and this is 50,000 now keep in mind
[17:13 -> 17:18]  you can't plug in strings here because
[17:16 -> 17:20]  you uh this doesn't have a single code
[17:18 -> 17:23]  point it only takes a single uni code
[17:20 -> 17:26]  code Point character and tells you its
[17:23 -> 17:30]  integer so in this way we can look
[17:26 -> 17:32]  up all the um characters of this
[17:30 -> 17:36]  specific string and their code points so
[17:32 -> 17:40]  or of X forx in this string and we get
[17:36 -> 17:42]  this encoding here now see here we've
[17:40 -> 17:44]  already turned the raw code points
[17:42 -> 17:46]  already have integers so why can't we
[17:44 -> 17:48]  simply just use these integers and not
[17:46 -> 17:50]  have any tokenization at all why can't
[17:48 -> 17:52]  we just use this natively as is and just
[17:50 -> 17:54]  use the code Point well one reason for
[17:52 -> 17:56]  that of course is that the vocabulary in
[17:54 -> 17:58]  that case would be quite long so in this
[17:56 -> 17:59]  case for Unicode the this is a
[17:58 -> 18:02]  vocabulary of
[17:59 -> 18:05]  150,000 different code points but more
[18:02 -> 18:07]  worryingly than that I think the Unicode
[18:05 -> 18:09]  standard is very much alive and it keeps
[18:07 -> 18:11]  changing and so it's not kind of a
[18:09 -> 18:13]  stable representation necessarily that
[18:11 -> 18:15]  we may want to use directly so for those
[18:13 -> 18:17]  reasons we need something a bit better
[18:15 -> 18:19]  so to find something better we turn to
[18:17 -> 18:21]  encodings so if we go to the Wikipedia
[18:19 -> 18:23]  page here we see that the Unicode
[18:21 -> 18:27]  consortion defines three types of
[18:23 -> 18:30]  encodings utf8 UTF 16 and UTF 32 these
[18:27 -> 18:33]  encoding are the way by which we can
[18:30 -> 18:37]  take Unicode text and translate it into
[18:33 -> 18:39]  binary data or by streams utf8 is by far
[18:37 -> 18:42]  the most common uh so this is the utf8
[18:39 -> 18:44]  page now this Wikipedia page is actually
[18:42 -> 18:46]  quite long but what's important for our
[18:44 -> 18:49]  purposes is that utf8 takes every single
[18:46 -> 18:52]  Cod point and it translates it to a by
[18:49 -> 18:54]  stream and this by stream is between one
[18:52 -> 18:56]  to four bytes so it's a variable length
[18:54 -> 18:58]  encoding so depending on the Unicode
[18:56 -> 18:59]  Point according to the schema you're
[18:58 -> 19:03]  going to end up with between 1 to four
[18:59 -> 19:05]  bytes for each code point on top of that
[19:03 -> 19:08]  there's utf8 uh
[19:05 -> 19:10]  utf16 and UTF 32 UTF 32 is nice because
[19:08 -> 19:12]  it is fixed length instead of variable
[19:10 -> 19:17]  length but it has many other downsides
[19:12 -> 19:18]  as well so the full kind of spectrum of
[19:17 -> 19:20]  pros and cons of all these different
[19:18 -> 19:22]  three encodings are beyond the scope of
[19:20 -> 19:25]  this video I just like to point out that
[19:22 -> 19:27]  I enjoyed this block post and this block
[19:25 -> 19:29]  post at the end of it also has a number
[19:27 -> 19:32]  of references that can be quite useful
[19:29 -> 19:34]  uh one of them is uh utf8 everywhere
[19:32 -> 19:36]  Manifesto um and this Manifesto
[19:34 -> 19:39]  describes the reason why utf8 is
[19:36 -> 19:41]  significantly preferred and a lot nicer
[19:39 -> 19:45]  than the other encodings and why it is
[19:41 -> 19:48]  used a lot more prominently um on the
[19:45 -> 19:49]  internet one of the major advantages
[19:48 -> 19:52]  just just to give you a sense is that
[19:49 -> 19:54]  utf8 is the only one of these that is
[19:52 -> 19:57]  backwards compatible to the much simpler
[19:54 -> 19:58]  asky encoding of text um but I'm not
[19:57 -> 20:01]  going to go into the full detail in this
[19:58 -> 20:03]  video so suffice to say that we like the
[20:01 -> 20:06]  utf8 encoding and uh let's try to take
[20:03 -> 20:08]  the string and see what we get if we
[20:06 -> 20:10]  encoded into
[20:08 -> 20:12]  utf8 the string class in Python actually
[20:10 -> 20:15]  has do encode and you can give it the
[20:12 -> 20:17]  encoding which is say utf8 now we get
[20:15 -> 20:20]  out of this is not very nice because
[20:17 -> 20:22]  this is the bytes is a bytes object and
[20:20 -> 20:25]  it's not very nice in the way that it's
[20:22 -> 20:26]  printed so I personally like to take it
[20:25 -> 20:28]  through list because then we actually
[20:26 -> 20:32]  get the raw B
[20:28 -> 20:35]  of this uh encoding so this is the raw
[20:32 -> 20:38]  byes that represent this string
[20:35 -> 20:40]  according to the utf8 en coding we can
[20:38 -> 20:43]  also look at utf16 we get a slightly
[20:40 -> 20:45]  different by stream and we here we start
[20:43 -> 20:47]  to see one of the disadvantages of utf16
[20:45 -> 20:49]  you see how we have zero Z something Z
[20:47 -> 20:50]  something Z something we're starting to
[20:49 -> 20:53]  get a sense that this is a bit of a
[20:50 -> 20:56]  wasteful encoding and indeed for simple
[20:53 -> 20:58]  asky characters or English characters
[20:56 -> 21:00]  here uh we just have the structure of 0
[20:58 -> 21:04]  something Z something and it's not
[21:00 -> 21:06]  exactly nice same for UTF 32 when we
[21:04 -> 21:08]  expand this we can start to get a sense
[21:06 -> 21:10]  of the wastefulness of this encoding for
[21:08 -> 21:11]  our purposes you see a lot of zeros
[21:10 -> 21:14]  followed by
[21:11 -> 21:17]  something and so uh this is not
[21:14 -> 21:20]  desirable so suffice it to say that we
[21:17 -> 21:23]  would like to stick with utf8 for our
[21:20 -> 21:26]  purposes however if we just use utf8
[21:23 -> 21:29]  naively these are by streams so that
[21:26 -> 21:33]  would imply a vocabulary length of only
[21:29 -> 21:35]  256 possible tokens uh but this this
[21:33 -> 21:36]  vocabulary size is very very small what
[21:35 -> 21:39]  this is going to do if we just were to
[21:36 -> 21:41]  use it naively is that all of our text
[21:39 -> 21:46]  would be stretched out over very very
[21:41 -> 21:49]  long sequences of bytes and so
[21:46 -> 21:51]  um what what this does is that certainly
[21:49 -> 21:52]  the embeding table is going to be tiny
[21:51 -> 21:54]  and the prediction at the top at the
[21:52 -> 21:56]  final layer is going to be very tiny but
[21:54 -> 21:59]  our sequences are very long and remember
[21:56 -> 22:01]  that we have pretty finite um context
[21:59 -> 22:02]  length and the attention that we can
[22:01 -> 22:05]  support in a transformer for
[22:02 -> 22:07]  computational reasons and so we only
[22:05 -> 22:09]  have as much context length but now we
[22:07 -> 22:10]  have very very long sequences and this
[22:09 -> 22:12]  is just inefficient and it's not going
[22:10 -> 22:15]  to allow us to attend to sufficiently
[22:12 -> 22:18]  long text uh before us for the purposes
[22:15 -> 22:21]  of the next token prediction task so we
[22:18 -> 22:24]  don't want to use the raw bytes of the
[22:21 -> 22:26]  utf8 encoding we want to be able to
[22:24 -> 22:28]  support larger vocabulary size that we
[22:26 -> 22:30]  can tune as a hyper
[22:28 -> 22:33]  but we want to stick with the utf8
[22:30 -> 22:35]  encoding of these strings so what do we
[22:33 -> 22:37]  do well the answer of course is we turn
[22:35 -> 22:39]  to the bite pair encoding algorithm
[22:37 -> 22:42]  which will allow us to compress these
[22:39 -> 22:44]  bite sequences um to a variable amount
[22:42 -> 22:47]  so we'll get to that in a bit but I just
[22:44 -> 22:49]  want to briefly speak to the fact that I
[22:47 -> 22:52]  would love nothing more than to be able
[22:49 -> 22:54]  to feed raw bite sequences into uh
[22:52 -> 22:57]  language models in fact there's a paper
[22:54 -> 22:59]  about how this could potentially be done
[22:57 -> 23:00]  uh from Summer last last year now the
[22:59 -> 23:02]  problem is you actually have to go in
[23:00 -> 23:04]  and you have to modify the Transformer
[23:02 -> 23:06]  architecture because as I mentioned
[23:04 -> 23:08]  you're going to have a problem where the
[23:06 -> 23:10]  attention will start to become extremely
[23:08 -> 23:13]  expensive because the sequences are so
[23:10 -> 23:15]  long and so in this paper they propose
[23:13 -> 23:17]  kind of a hierarchical structuring of
[23:15 -> 23:20]  the Transformer that could allow you to
[23:17 -> 23:21]  just feed in raw bites and so at the end
[23:20 -> 23:23]  they say together these results
[23:21 -> 23:25]  establish the viability of tokenization
[23:23 -> 23:27]  free autor regressive sequence modeling
[23:25 -> 23:30]  at scale so tokenization free would
[23:27 -> 23:32]  indeed be amazing we would just feed B
[23:30 -> 23:34]  streams directly into our models but
[23:32 -> 23:36]  unfortunately I don't know that this has
[23:34 -> 23:37]  really been proven out yet by
[23:36 -> 23:39]  sufficiently many groups and a
[23:37 -> 23:40]  sufficient scale uh but something like
[23:39 -> 23:42]  this at one point would be amazing and I
[23:40 -> 23:44]  hope someone comes up with it but for
[23:42 -> 23:46]  now we have to come back and we can't
[23:44 -> 23:48]  feed this directly into language models
[23:46 -> 23:49]  and we have to compress it using the B
[23:48 -> 23:51]  paare encoding algorithm so let's see
[23:49 -> 23:53]  how that works so as I mentioned the B
[23:51 -> 23:55]  paare encoding algorithm is not all that
[23:53 -> 23:57]  complicated and the Wikipedia page is
[23:55 -> 23:59]  actually quite instructive as far as the
[23:57 -> 24:01]  basic idea goes go what we're doing is
[23:59 -> 24:03]  we have some kind of a input sequence uh
[24:01 -> 24:06]  like for example here we have only four
[24:03 -> 24:08]  elements in our vocabulary a b c and d
[24:06 -> 24:09]  and we have a sequence of them so
[24:08 -> 24:12]  instead of bytes let's say we just have
[24:09 -> 24:14]  four a vocab size of
[24:12 -> 24:16]  four the sequence is too long and we'd
[24:14 -> 24:20]  like to compress it so what we do is
[24:16 -> 24:23]  that we iteratively find the pair of uh
[24:20 -> 24:25]  tokens that occur the most
[24:23 -> 24:28]  frequently and then once we've
[24:25 -> 24:30]  identified that pair we repl replace
[24:28 -> 24:33]  that pair with just a single new token
[24:30 -> 24:36]  that we append to our vocabulary so for
[24:33 -> 24:38]  example here the bite pair AA occurs
[24:36 -> 24:41]  most often so we mint a new token let's
[24:38 -> 24:46]  call it capital Z and we replace every
[24:41 -> 24:48]  single occurrence of AA by Z so now we
[24:46 -> 24:51]  have two Z's here so here we took a
[24:48 -> 24:54]  sequence of 11 characters with
[24:51 -> 24:58]  vocabulary size four and we've converted
[24:54 -> 25:00]  it to a um sequence of only nine tokens
[24:58 -> 25:02]  but now with a vocabulary of five
[25:00 -> 25:04]  because we have a fifth vocabulary
[25:02 -> 25:07]  element that we just created and it's Z
[25:04 -> 25:10]  standing for concatination of AA and we
[25:07 -> 25:12]  can again repeat this process so we
[25:10 -> 25:15]  again look at the sequence and identify
[25:12 -> 25:19]  the pair of tokens that are most
[25:15 -> 25:20]  frequent let's say that that is now AB
[25:19 -> 25:23]  well we are going to replace AB with a
[25:20 -> 25:25]  new token that we meant call Y so y
[25:23 -> 25:28]  becomes ab and then every single
[25:25 -> 25:31]  occurrence of ab is now replaced with y
[25:28 -> 25:35]  so we end up with this so now we only
[25:31 -> 25:40]  have 1 2 3 4 5 6 seven characters in our
[25:35 -> 25:42]  sequence but we have not just um four
[25:40 -> 25:45]  vocabulary elements or five but now we
[25:42 -> 25:47]  have six and for the final round we
[25:45 -> 25:50]  again look through the sequence find
[25:47 -> 25:53]  that the phrase zy or the pair zy is
[25:50 -> 25:56]  most common and replace it one more time
[25:53 -> 25:59]  with another um character let's say x so
[25:56 -> 26:02]  X is z y and we replace all curses of zy
[25:59 -> 26:03]  and we get this following sequence so
[26:02 -> 26:08]  basically after we have gone through
[26:03 -> 26:09]  this process instead of having a um
[26:08 -> 26:13]  sequence of
[26:09 -> 26:18]  11 uh tokens with a vocabulary length of
[26:13 -> 26:21]  four we now have a sequence of 1 2 3
[26:18 -> 26:25]  four five tokens but our vocabulary
[26:21 -> 26:27]  length now is seven and so in this way
[26:25 -> 26:30]  we can iteratively compress our sequence
[26:27 -> 26:32]  I we Mint new tokens so in the in the
[26:30 -> 26:36]  exact same way we start we start out
[26:32 -> 26:38]  with bite sequences so we have 256
[26:36 -> 26:40]  vocabulary size but we're now going to
[26:38 -> 26:42]  go through these and find the bite pairs
[26:40 -> 26:44]  that occur the most and we're going to
[26:42 -> 26:46]  iteratively start minting new tokens
[26:44 -> 26:48]  appending them to our vocabulary and
[26:46 -> 26:50]  replacing things and in this way we're
[26:48 -> 26:52]  going to end up with a compressed
[26:50 -> 26:55]  training data set and also an algorithm
[26:52 -> 26:58]  for taking any arbitrary sequence and
[26:55 -> 27:01]  encoding it using this uh vocabul
[26:58 -> 27:03]  and also decoding it back to Strings so
[27:01 -> 27:05]  let's now Implement all that so here's
[27:03 -> 27:07]  what I did I went to this block post
[27:05 -> 27:10]  that I enjoyed and I took the first
[27:07 -> 27:13]  paragraph and I copy pasted it here into
[27:10 -> 27:15]  text so this is one very long line
[27:13 -> 27:17]  here now to get the tokens as I
[27:15 -> 27:20]  mentioned we just take our text and we
[27:17 -> 27:22]  encode it into utf8 the tokens here at
[27:20 -> 27:25]  this point will be a raw bites single
[27:22 -> 27:27]  stream of bytes and just so that it's
[27:25 -> 27:29]  easier to work with instead of just a
[27:27 -> 27:32]  bytes object I'm going to convert all
[27:29 -> 27:34]  those bytes to integers and then create
[27:32 -> 27:35]  a list of it just so it's easier for us
[27:34 -> 27:38]  to manipulate and work with in Python
[27:35 -> 27:42]  and visualize and here I'm printing all
[27:38 -> 27:45]  of that so this is the original um this
[27:42 -> 27:45]  is the original paragraph and its length
[27:45 -> 27:49]  is
[27:45 -> 27:53]  533 uh code points and then here are the
[27:49 -> 27:56]  bytes encoded in ut utf8 and we see that
[27:53 -> 27:59]  this has a length of 616 bytes at this
[27:56 -> 28:01]  point or 616 tokens and the reason this
[27:59 -> 28:04]  is more is because a lot of these simple
[28:01 -> 28:06]  asky characters or simple characters
[28:04 -> 28:08]  they just become a single bite but a lot
[28:06 -> 28:11]  of these Unicode more complex characters
[28:08 -> 28:12]  become multiple bytes up to four and so
[28:11 -> 28:14]  we are expanding that
[28:12 -> 28:16]  size so now what we'd like to do as a
[28:14 -> 28:18]  first step of the algorithm is we'd like
[28:16 -> 28:22]  to iterate over here and find the pair
[28:18 -> 28:24]  of bites that occur most frequently
[28:22 -> 28:25]  because we're then going to merge it so
[28:24 -> 28:27]  if you are working long on a notebook on
[28:25 -> 28:29]  a side then I encourage you to basically
[28:27 -> 28:31]  click on the link find this notebook and
[28:29 -> 28:32]  try to write that function yourself
[28:31 -> 28:34]  otherwise I'm going to come here and
[28:32 -> 28:36]  Implement first the function that finds
[28:34 -> 28:38]  the most common pair okay so here's what
[28:36 -> 28:40]  I came up with there are many different
[28:38 -> 28:42]  ways to implement this but I'm calling
[28:40 -> 28:44]  the function get stats it expects a list
[28:42 -> 28:46]  of integers I'm using a dictionary to
[28:44 -> 28:48]  keep track of basically the counts and
[28:46 -> 28:51]  then this is a pythonic way to iterate
[28:48 -> 28:53]  consecutive elements of this list uh
[28:51 -> 28:55]  which we covered in the previous video
[28:53 -> 28:58]  and then here I'm just keeping track of
[28:55 -> 29:00]  just incrementing by one um for all the
[28:58 -> 29:03]  pairs so if I call this on all the
[29:00 -> 29:06]  tokens here then the stats comes out
[29:03 -> 29:08]  here so this is the dictionary the keys
[29:06 -> 29:11]  are these topples of consecutive
[29:08 -> 29:14]  elements and this is the count so just
[29:11 -> 29:17]  to uh print it in a slightly better way
[29:14 -> 29:20]  this is one way that I like to do that
[29:17 -> 29:22]  where you it's a little bit compound
[29:20 -> 29:25]  here so you can pause if you like but we
[29:22 -> 29:27]  iterate all all the items the items
[29:25 -> 29:31]  called on dictionary returns pairs of
[29:27 -> 29:35]  key value and instead I create a list
[29:31 -> 29:37]  here of value key because if it's a
[29:35 -> 29:41]  value key list then I can call sort on
[29:37 -> 29:43]  it and by default python will uh use the
[29:41 -> 29:46]  first element which in this case will be
[29:43 -> 29:48]  value to sort by if it's given tles and
[29:46 -> 29:50]  then reverse so it's descending and
[29:48 -> 29:53]  print that so basically it looks like
[29:50 -> 29:55]  101 comma 32 was the most commonly
[29:53 -> 29:58]  occurring consecutive pair and it
[29:55 -> 30:00]  occurred 20 times we can double check
[29:58 -> 30:02]  that that makes reasonable sense so if I
[30:00 -> 30:05]  just search
[30:02 -> 30:10]  10132 then you see that these are the 20
[30:05 -> 30:11]  occurrences of that um pair and if we'd
[30:10 -> 30:14]  like to take a look at what exactly that
[30:11 -> 30:17]  pair is we can use Char which is the
[30:14 -> 30:22]  opposite of or in Python so we give it a
[30:17 -> 30:25]  um unic code Cod point so 101 and of 32
[30:22 -> 30:28]  and we see that this is e and space so
[30:25 -> 30:29]  basically there's a lot of E space here
[30:28 -> 30:32]  meaning that a lot of these words seem
[30:29 -> 30:34]  to end with e so here's eace as an
[30:32 -> 30:36]  example so there's a lot of that going
[30:34 -> 30:38]  on here and this is the most common pair
[30:36 -> 30:40]  so now that we've identified the most
[30:38 -> 30:42]  common pair we would like to iterate
[30:40 -> 30:44]  over this sequence we're going to Mint a
[30:42 -> 30:47]  new token with the ID of
[30:44 -> 30:50]  256 right because these tokens currently
[30:47 -> 30:52]  go from Z to 255 so when we create a new
[30:50 -> 30:56]  token it will have an ID of
[30:52 -> 30:59]  256 and we're going to iterate over this
[30:56 -> 31:02]  entire um list and every every time we
[30:59 -> 31:03]  see 101 comma 32 we're going to swap
[31:02 -> 31:07]  that out for
[31:03 -> 31:09]  256 so let's Implement that now and feel
[31:07 -> 31:11]  free to uh do that yourself as well so
[31:09 -> 31:14]  first I commented uh this just so we
[31:11 -> 31:17]  don't pollute uh the notebook too much
[31:14 -> 31:20]  this is a nice way of in Python
[31:17 -> 31:23]  obtaining the highest ranking pair so
[31:20 -> 31:26]  we're basically calling the Max on this
[31:23 -> 31:27]  dictionary stats and this will return
[31:26 -> 31:30]  the maximum
[31:27 -> 31:32]  key and then the question is how does it
[31:30 -> 31:35]  rank keys so you can provide it with a
[31:32 -> 31:38]  function that ranks keys and that
[31:35 -> 31:41]  function is just stats. getet uh stats.
[31:38 -> 31:42]  getet would basically return the value
[31:41 -> 31:45]  and so we're ranking by the value and
[31:42 -> 31:49]  getting the maximum key so it's 101
[31:45 -> 31:51]  comma 32 as we saw now to actually merge
[31:49 -> 31:53]  10132 um this is the function that I
[31:51 -> 31:55]  wrote but again there are many different
[31:53 -> 31:57]  versions of it so we're going to take a
[31:55 -> 31:59]  list of IDs and the the pair that we
[31:57 -> 32:02]  want to replace and that pair will be
[31:59 -> 32:05]  replaced with the new index
[32:02 -> 32:08]  idx so iterating through IDs if we find
[32:05 -> 32:10]  the pair swap it out for idx so we
[32:08 -> 32:12]  create this new list and then we start
[32:10 -> 32:14]  at zero and then we go through this
[32:12 -> 32:17]  entire list sequentially from left to
[32:14 -> 32:19]  right and here we are checking for
[32:17 -> 32:20]  equality at the current position with
[32:19 -> 32:23]  the
[32:20 -> 32:25]  pair um so here we are checking that the
[32:23 -> 32:27]  pair matches now here is a bit of a
[32:25 -> 32:29]  tricky condition that you have to append
[32:27 -> 32:31]  if you're trying to be careful and that
[32:29 -> 32:33]  is that um you don't want this here to
[32:31 -> 32:35]  be out of Bounds at the very last
[32:33 -> 32:37]  position when you're on the rightmost
[32:35 -> 32:39]  element of this list otherwise this
[32:37 -> 32:40]  would uh give you an autof bounds error
[32:39 -> 32:44]  so we have to make sure that we're not
[32:40 -> 32:46]  at the very very last element so uh this
[32:44 -> 32:51]  would be false for that so if we find a
[32:46 -> 32:53]  match we append to this new list that
[32:51 -> 32:54]  replacement index and we increment the
[32:53 -> 32:57]  position by two so we skip over that
[32:54 -> 32:59]  entire pair but otherwise if we we
[32:57 -> 33:02]  haven't found a matching pair we just
[32:59 -> 33:05]  sort of copy over the um element at that
[33:02 -> 33:07]  position and increment by one then
[33:05 -> 33:10]  return this so here's a very small toy
[33:07 -> 33:12]  example if we have a list 566 791 and we
[33:10 -> 33:16]  want to replace the occurrences of 67
[33:12 -> 33:18]  with 99 then calling this on that will
[33:16 -> 33:21]  give us what we're asking for so here
[33:18 -> 33:23]  the 67 is replaced with
[33:21 -> 33:27]  99 so now I'm going to uncomment this
[33:23 -> 33:29]  for our actual use case where we want to
[33:27 -> 33:33]  take our tokens we want to take the top
[33:29 -> 33:37]  pair here and replace it with 256 to get
[33:33 -> 33:40]  tokens to if we run this we get the
[33:37 -> 33:45]  following so recall that previously we
[33:40 -> 33:48]  had a length 616 in this list and now we
[33:45 -> 33:50]  have a length 596 right so this
[33:48 -> 33:52]  decreased by 20 which makes sense
[33:50 -> 33:55]  because there are 20 occurrences
[33:52 -> 33:58]  moreover we can try to find 256 here and
[33:55 -> 33:59]  we see plenty of occurrences on off it
[33:58 -> 34:02]  and moreover just double check there
[33:59 -> 34:05]  should be no occurrence of 10132 so this
[34:02 -> 34:06]  is the original array plenty of them and
[34:05 -> 34:08]  in the second array there are no
[34:06 -> 34:11]  occurrences of 1032 so we've
[34:08 -> 34:13]  successfully merged this single pair and
[34:11 -> 34:15]  now we just uh iterate this so we are
[34:13 -> 34:17]  going to go over the sequence again find
[34:15 -> 34:19]  the most common pair and replace it so
[34:17 -> 34:21]  let me now write a y Loop that uses
[34:19 -> 34:24]  these functions to do this um sort of
[34:21 -> 34:26]  iteratively and how many times do we do
[34:24 -> 34:27]  it four well that's totally up to us as
[34:26 -> 34:30]  a hyper parameter
[34:27 -> 34:33]  the more um steps we take the larger
[34:30 -> 34:35]  will be our vocabulary and the shorter
[34:33 -> 34:37]  will be our sequence and there is some
[34:35 -> 34:39]  sweet spot that we usually find works
[34:37 -> 34:41]  the best in practice and so this is kind
[34:39 -> 34:44]  of a hyperparameter and we tune it and
[34:41 -> 34:46]  we find good vocabulary sizes as an
[34:44 -> 34:49]  example gp4 currently uses roughly
[34:46 -> 34:51]  100,000 tokens and um bpark that those
[34:49 -> 34:53]  are reasonable numbers currently instead
[34:51 -> 34:55]  the are large language models so let me
[34:53 -> 34:58]  now write uh putting putting it all
[34:55 -> 35:00]  together and uh iterating these steps
[34:58 -> 35:03]  okay now before we dive into the Y loop
[35:00 -> 35:04]  I wanted to add one more cell here where
[35:03 -> 35:07]  I went to the block post and instead of
[35:04 -> 35:08]  grabbing just the first paragraph or two
[35:07 -> 35:10]  I took the entire block post and I
[35:08 -> 35:12]  stretched it out in a single line and
[35:10 -> 35:13]  basically just using longer text will
[35:12 -> 35:16]  allow us to have more representative
[35:13 -> 35:18]  statistics for the bite Pairs and we'll
[35:16 -> 35:21]  just get a more sensible results out of
[35:18 -> 35:24]  it because it's longer text um so here
[35:21 -> 35:27]  we have the raw text we encode it into
[35:24 -> 35:30]  bytes using the utf8 encoding
[35:27 -> 35:31]  and then here as before we are just
[35:30 -> 35:33]  changing it into a list of integers in
[35:31 -> 35:36]  Python just so it's easier to work with
[35:33 -> 35:40]  instead of the raw byes objects and then
[35:36 -> 35:44]  this is the code that I came up with uh
[35:40 -> 35:45]  to actually do the merging in Loop these
[35:44 -> 35:48]  two functions here are identical to what
[35:45 -> 35:49]  we had above I only included them here
[35:48 -> 35:53]  just so that you have the point of
[35:49 -> 35:55]  reference here so uh these two are
[35:53 -> 35:57]  identical and then this is the new code
[35:55 -> 35:58]  that I added so the first first thing we
[35:57 -> 36:01]  want to do is we want to decide on the
[35:58 -> 36:02]  final vocabulary size that we want our
[36:01 -> 36:04]  tokenizer to have and as I mentioned
[36:02 -> 36:06]  this is a hyper parameter and you set it
[36:04 -> 36:08]  in some way depending on your best
[36:06 -> 36:10]  performance so let's say for us we're
[36:08 -> 36:13]  going to use 276 because that way we're
[36:10 -> 36:15]  going to be doing exactly 20
[36:13 -> 36:16]  merges and uh 20 merges because we
[36:15 -> 36:20]  already have
[36:16 -> 36:23]  256 tokens for the raw bytes and to
[36:20 -> 36:25]  reach 276 we have to do 20 merges uh to
[36:23 -> 36:28]  add 20 new
[36:25 -> 36:31]  tokens here uh this is uh one way in
[36:28 -> 36:33]  Python to just create a copy of a list
[36:31 -> 36:35]  so I'm taking the tokens list and by
[36:33 -> 36:37]  wrapping it in a list python will
[36:35 -> 36:38]  construct a new list of all the
[36:37 -> 36:39]  individual elements so this is just a
[36:38 -> 36:42]  copy
[36:39 -> 36:44]  operation then here I'm creating a
[36:42 -> 36:46]  merges uh dictionary so this merges
[36:44 -> 36:49]  dictionary is going to maintain
[36:46 -> 36:52]  basically the child one child two
[36:49 -> 36:53]  mapping to a new uh token and so what
[36:52 -> 36:56]  we're going to be building up here is a
[36:53 -> 36:59]  binary tree of merges but actually it's
[36:56 -> 37:01]  not exactly a tree because a tree would
[36:59 -> 37:03]  have a single root node with a bunch of
[37:01 -> 37:05]  leaves for us we're starting with the
[37:03 -> 37:06]  leaves on the bottom which are the
[37:05 -> 37:09]  individual bites those are the starting
[37:06 -> 37:11]  256 tokens and then we're starting to
[37:09 -> 37:14]  like merge two of them at a time and so
[37:11 -> 37:18]  it's not a tree it's more like a forest
[37:14 -> 37:22]  um uh as we merge these elements
[37:18 -> 37:25]  so for 20 merges we're going to find the
[37:22 -> 37:28]  most commonly occurring pair we're going
[37:25 -> 37:30]  to Mint a new token integer for it so I
[37:28 -> 37:32]  here will start at zero so we'll going
[37:30 -> 37:34]  to start at 256 we're going to print
[37:32 -> 37:36]  that we're merging it and we're going to
[37:34 -> 37:39]  replace all of the occurrences of that
[37:36 -> 37:42]  pair with the new new lied token and
[37:39 -> 37:45]  we're going to record that this pair of
[37:42 -> 37:49]  integers merged into this new
[37:45 -> 37:51]  integer so running this gives us the
[37:49 -> 37:54]  following
[37:51 -> 37:56]  output so we did 20 merges and for
[37:54 -> 37:58]  example the first merge was exactly as
[37:56 -> 38:01]  before the
[37:58 -> 38:04]  10132 um tokens merging into a new token
[38:01 -> 38:06]  2556 now keep in mind that the
[38:04 -> 38:08]  individual uh tokens 101 and 32 can
[38:06 -> 38:10]  still occur in the sequence after
[38:08 -> 38:12]  merging it's only when they occur
[38:10 -> 38:13]  exactly consecutively that that becomes
[38:12 -> 38:16]  256
[38:13 -> 38:19]  now um and in particular the other thing
[38:16 -> 38:21]  to notice here is that the token 256
[38:19 -> 38:23]  which is the newly minted token is also
[38:21 -> 38:26]  eligible for merging so here on the
[38:23 -> 38:28]  bottom the 20th merge was a merge of 25
[38:26 -> 38:31]  and 259 becoming
[38:28 -> 38:33]  275 so every time we replace these
[38:31 -> 38:35]  tokens they become eligible for merging
[38:33 -> 38:37]  in the next round of data ration so
[38:35 -> 38:38]  that's why we're building up a small
[38:37 -> 38:40]  sort of binary Forest instead of a
[38:38 -> 38:42]  single individual
[38:40 -> 38:44]  tree one thing we can take a look at as
[38:42 -> 38:46]  well is we can take a look at the
[38:44 -> 38:48]  compression ratio that we've achieved so
[38:46 -> 38:51]  in particular we started off with this
[38:48 -> 38:56]  tokens list um so we started off with
[38:51 -> 38:58]  24,000 bytes and after merging 20 times
[38:56 -> 39:01]  uh we now have only
[38:58 -> 39:03]  19,000 um tokens and so therefore the
[39:01 -> 39:06]  compression ratio simply just dividing
[39:03 -> 39:07]  the two is roughly 1.27 so that's the
[39:06 -> 39:10]  amount of compression we were able to
[39:07 -> 39:13]  achieve of this text with only 20
[39:10 -> 39:15]  merges um and of course the more
[39:13 -> 39:19]  vocabulary elements you add uh the
[39:15 -> 39:23]  greater the compression ratio here would
[39:19 -> 39:25]  be finally so that's kind of like um the
[39:23 -> 39:28]  training of the tokenizer if you will
[39:25 -> 39:31]  now 1 Point I wanted to make is that and
[39:28 -> 39:33]  maybe this is a diagram that can help um
[39:31 -> 39:34]  kind of illustrate is that tokenizer is
[39:33 -> 39:37]  a completely separate object from the
[39:34 -> 39:38]  large language model itself so
[39:37 -> 39:40]  everything in this lecture we're not
[39:38 -> 39:41]  really touching the llm itself uh we're
[39:40 -> 39:43]  just training the tokenizer this is a
[39:41 -> 39:46]  completely separate pre-processing stage
[39:43 -> 39:47]  usually so the tokenizer will have its
[39:46 -> 39:49]  own training set just like a large
[39:47 -> 39:52]  language model has a potentially
[39:49 -> 39:53]  different training set so the tokenizer
[39:52 -> 39:54]  has a training set of documents on which
[39:53 -> 39:57]  you're going to train the
[39:54 -> 39:58]  tokenizer and then and um we're
[39:57 -> 40:01]  performing The Bite pair encoding
[39:58 -> 40:02]  algorithm as we saw above to train the
[40:01 -> 40:04]  vocabulary of this
[40:02 -> 40:06]  tokenizer so it has its own training set
[40:04 -> 40:09]  it is a pre-processing stage that you
[40:06 -> 40:11]  would run a single time in the beginning
[40:09 -> 40:14]  um and the tokenizer is trained using
[40:11 -> 40:16]  bipar coding algorithm once you have the
[40:14 -> 40:19]  tokenizer once it's trained and you have
[40:16 -> 40:22]  the vocabulary and you have the merges
[40:19 -> 40:24]  uh we can do both encoding and decoding
[40:22 -> 40:27]  so these two arrows here so the
[40:24 -> 40:30]  tokenizer is a translation layer between
[40:27 -> 40:32]  raw text which is as we saw the sequence
[40:30 -> 40:35]  of Unicode code points it can take raw
[40:32 -> 40:37]  text and turn it into a token sequence
[40:35 -> 40:40]  and vice versa it can take a token
[40:37 -> 40:43]  sequence and translate it back into raw
[40:40 -> 40:45]  text so now that we have trained uh
[40:43 -> 40:47]  tokenizer and we have these merges we
[40:45 -> 40:49]  are going to turn to how we can do the
[40:47 -> 40:51]  encoding and the decoding step if you
[40:49 -> 40:53]  give me text here are the tokens and
[40:51 -> 40:55]  vice versa if you give me tokens here's
[40:53 -> 40:57]  the text once we have that we can
[40:55 -> 40:58]  translate between these two Realms and
[40:57 -> 41:01]  then the language model is going to be
[40:58 -> 41:03]  trained as a step two afterwards and
[41:01 -> 41:05]  typically in a in a sort of a
[41:03 -> 41:06]  state-of-the-art application you might
[41:05 -> 41:08]  take all of your training data for the
[41:06 -> 41:10]  language model and you might run it
[41:08 -> 41:11]  through the tokenizer and sort of
[41:10 -> 41:13]  translate everything into a massive
[41:11 -> 41:15]  token sequence and then you can throw
[41:13 -> 41:17]  away the raw text you're just left with
[41:15 -> 41:19]  the tokens themselves and those are
[41:17 -> 41:21]  stored on disk and that is what the
[41:19 -> 41:23]  large language model is actually reading
[41:21 -> 41:24]  when it's training on them so this one
[41:23 -> 41:26]  approach that you can take as a single
[41:24 -> 41:30]  massive pre-processing step a
[41:26 -> 41:31]  stage um so yeah basically I think the
[41:30 -> 41:32]  most important thing I want to get
[41:31 -> 41:34]  across is that this is completely
[41:32 -> 41:36]  separate stage it usually has its own
[41:34 -> 41:38]  entire uh training set you may want to
[41:36 -> 41:39]  have those training sets be different
[41:38 -> 41:41]  between the tokenizer and the logge
[41:39 -> 41:43]  language model so for example when
[41:41 -> 41:45]  you're training the tokenizer as I
[41:43 -> 41:46]  mentioned we don't just care about the
[41:45 -> 41:49]  performance of English text we care
[41:46 -> 41:51]  about uh multi many different languages
[41:49 -> 41:53]  and we also care about code or not code
[41:51 -> 41:55]  so you may want to look into different
[41:53 -> 41:57]  kinds of mixtures of different kinds of
[41:55 -> 42:00]  languages and different amounts of code
[41:57 -> 42:01]  and things like that because the amount
[42:00 -> 42:03]  of different language that you have in
[42:01 -> 42:06]  your tokenizer training set will
[42:03 -> 42:08]  determine how many merges of it there
[42:06 -> 42:11]  will be and therefore that determines
[42:08 -> 42:15]  the density with which uh this type of
[42:11 -> 42:17]  data is um sort of has in the token
[42:15 -> 42:19]  space and so roughly speaking
[42:17 -> 42:21]  intuitively if you add some amount of
[42:19 -> 42:24]  data like say you have a ton of Japanese
[42:21 -> 42:25]  data in your uh tokenizer training set
[42:24 -> 42:26]  then that means that more Japanese
[42:25 -> 42:28]  tokens will get merged
[42:26 -> 42:30]  and therefore Japanese will have shorter
[42:28 -> 42:32]  sequences uh and that's going to be
[42:30 -> 42:34]  beneficial for the large language model
[42:32 -> 42:36]  which has a finite context length on
[42:34 -> 42:39]  which it can work on in in the token
[42:36 -> 42:41]  space uh so hopefully that makes sense
[42:39 -> 42:43]  so we're now going to turn to encoding
[42:41 -> 42:46]  and decoding now that we have trained a
[42:43 -> 42:48]  tokenizer so we have our merges and now
[42:46 -> 42:50]  how do we do encoding and decoding okay
[42:48 -> 42:52]  so let's begin with decoding which is
[42:50 -> 42:54]  this Arrow over here so given a token
[42:52 -> 42:57]  sequence let's go through the tokenizer
[42:54 -> 42:59]  to get back a python string object so
[42:57 -> 43:01]  the raw text so this is the function
[42:59 -> 43:03]  that we' like to implement um we're
[43:01 -> 43:05]  given the list of integers and we want
[43:03 -> 43:06]  to return a python string if you'd like
[43:05 -> 43:08]  uh try to implement this function
[43:06 -> 43:11]  yourself it's a fun exercise otherwise
[43:08 -> 43:13]  I'm going to start uh pasting in my own
[43:11 -> 43:16]  solution so there are many different
[43:13 -> 43:18]  ways to do it um here's one way I will
[43:16 -> 43:21]  create an uh kind of pre-processing
[43:18 -> 43:24]  variable that I will call
[43:21 -> 43:27]  vocab and vocab is a mapping or a
[43:24 -> 43:31]  dictionary in Python for from the token
[43:27 -> 43:33]  uh ID to the bytes object for that token
[43:31 -> 43:36]  so we begin with the raw bytes for
[43:33 -> 43:39]  tokens from 0 to 255 and then we go in
[43:36 -> 43:42]  order of all the merges and we sort of
[43:39 -> 43:45]  uh populate this vocab list by doing an
[43:42 -> 43:47]  addition here so this is the basically
[43:45 -> 43:50]  the bytes representation of the first
[43:47 -> 43:52]  child followed by the second one and
[43:50 -> 43:54]  remember these are bytes objects so this
[43:52 -> 43:57]  addition here is an addition of two
[43:54 -> 43:58]  bytes objects just concatenation
[43:57 -> 44:01]  so that's what we get
[43:58 -> 44:02]  here one tricky thing to be careful with
[44:01 -> 44:06]  by the way is that I'm iterating a
[44:02 -> 44:08]  dictionary in Python using a DOT items
[44:06 -> 44:11]  and uh it really matters that this runs
[44:08 -> 44:13]  in the order in which we inserted items
[44:11 -> 44:15]  into the merous dictionary luckily
[44:13 -> 44:17]  starting with python 3.7 this is
[44:15 -> 44:19]  guaranteed to be the case but before
[44:17 -> 44:20]  python 3.7 this iteration may have been
[44:19 -> 44:23]  out of order with respect to how we
[44:20 -> 44:25]  inserted elements into merges and this
[44:23 -> 44:28]  may not have worked but we are using an
[44:25 -> 44:31]  um modern python so we're okay and then
[44:28 -> 44:35]  here uh given the IDS the first thing
[44:31 -> 44:37]  we're going to do is get the
[44:35 -> 44:39]  tokens so the way I implemented this
[44:37 -> 44:41]  here is I'm taking I'm iterating over
[44:39 -> 44:44]  all the IDS I'm using vocap to look up
[44:41 -> 44:46]  their bytes and then here this is one
[44:44 -> 44:49]  way in Python to concatenate all these
[44:46 -> 44:51]  bytes together to create our tokens and
[44:49 -> 44:56]  then these tokens here at this point are
[44:51 -> 44:59]  raw bytes so I have to decode using UTF
[44:56 -> 45:01]  F now back into python strings so
[44:59 -> 45:03]  previously we called that encode on a
[45:01 -> 45:05]  string object to get the bytes and now
[45:03 -> 45:07]  we're doing it Opposite we're taking the
[45:05 -> 45:11]  bytes and calling a decode on the bytes
[45:07 -> 45:13]  object to get a string in Python and
[45:11 -> 45:16]  then we can return
[45:13 -> 45:20]  text so um this is how we can do it now
[45:16 -> 45:22]  this actually has a um issue um in the
[45:20 -> 45:24]  way I implemented it and this could
[45:22 -> 45:26]  actually throw an error so try to think
[45:24 -> 45:30]  figure out why this code could actually
[45:26 -> 45:32]  result in an error if we plug in um uh
[45:30 -> 45:35]  some sequence of IDs that is
[45:32 -> 45:37]  unlucky so let me demonstrate the issue
[45:35 -> 45:41]  when I try to decode just something like
[45:37 -> 45:44]  97 I am going to get letter A here back
[45:41 -> 45:48]  so nothing too crazy happening but when
[45:44 -> 45:51]  I try to decode 128 as a single element
[45:48 -> 45:55]  the token 128 is what in string or in
[45:51 -> 46:00]  Python object uni Cod decoder utfa can't
[45:55 -> 46:01]  Decode by um 0x8 which is this in HEX in
[46:00 -> 46:03]  position zero invalid start bite what
[46:01 -> 46:04]  does that mean well to understand what
[46:03 -> 46:07]  this means we have to go back to our
[46:04 -> 46:10]  utf8 page uh that I briefly showed
[46:07 -> 46:13]  earlier and this is Wikipedia utf8 and
[46:10 -> 46:16]  basically there's a specific schema that
[46:13 -> 46:19]  utfa bytes take so in particular if you
[46:16 -> 46:21]  have a multi-te object for some of the
[46:19 -> 46:24]  Unicode characters they have to have
[46:21 -> 46:26]  this special sort of envelope in how the
[46:24 -> 46:30]  encoding works and so what's happening
[46:26 -> 46:31]  here is that invalid start pite that's
[46:30 -> 46:33]  because
[46:31 -> 46:37]  128 the binary representation of it is
[46:33 -> 46:39]  one followed by all zeros so we have one
[46:37 -> 46:41]  and then all zero and we see here that
[46:39 -> 46:42]  that doesn't conform to the format
[46:41 -> 46:44]  because one followed by all zero just
[46:42 -> 46:47]  doesn't fit any of these rules so to
[46:44 -> 46:50]  speak so it's an invalid start bite
[46:47 -> 46:52]  which is byte one this one must have a
[46:50 -> 46:54]  one following it and then a zero
[46:52 -> 46:57]  following it and then the content of
[46:54 -> 46:59]  your uni codee in x here so basically we
[46:57 -> 47:02]  don't um exactly follow the utf8
[46:59 -> 47:06]  standard and this cannot be decoded and
[47:02 -> 47:11]  so the way to fix this um is to
[47:06 -> 47:13]  use this errors equals in bytes. decode
[47:11 -> 47:17]  function of python and by default errors
[47:13 -> 47:20]  is strict so we will throw an error if
[47:17 -> 47:21]  um it's not valid utf8 bytes encoding
[47:20 -> 47:23]  but there are many different things that
[47:21 -> 47:25]  you could put here on error handling
[47:23 -> 47:27]  this is the full list of all the errors
[47:25 -> 47:29]  that you can use and in particular
[47:27 -> 47:32]  instead of strict let's change it to
[47:29 -> 47:35]  replace and that will replace uh with
[47:32 -> 47:40]  this special marker this replacement
[47:35 -> 47:43]  character so errors equals replace and
[47:40 -> 47:46]  now we just get that character
[47:43 -> 47:48]  back so basically not every single by
[47:46 -> 47:51]  sequence is valid
[47:48 -> 47:53]  utf8 and if it happens that your large
[47:51 -> 47:56]  language model for example predicts your
[47:53 -> 48:00]  tokens in a bad manner then they might
[47:56 -> 48:02]  not fall into valid utf8 and then we
[48:00 -> 48:05]  won't be able to decode them so the
[48:02 -> 48:07]  standard practice is to basically uh use
[48:05 -> 48:10]  errors equals replace and this is what
[48:07 -> 48:12]  you will also find in the openai um code
[48:10 -> 48:14]  that they released as well but basically
[48:12 -> 48:16]  whenever you see um this kind of a
[48:14 -> 48:18]  character in your output in that case uh
[48:16 -> 48:21]  something went wrong and the LM output
[48:18 -> 48:23]  not was not valid uh sort of sequence of
[48:21 -> 48:25]  tokens okay and now we're going to go
[48:23 -> 48:26]  the other way so we are going to
[48:25 -> 48:27]  implement
[48:26 -> 48:29]  this Arrow right here where we are going
[48:27 -> 48:31]  to be given a string and we want to
[48:29 -> 48:33]  encode it into
[48:31 -> 48:36]  tokens so this is the signature of the
[48:33 -> 48:38]  function that we're interested in and um
[48:36 -> 48:41]  this should basically print a list of
[48:38 -> 48:43]  integers of the tokens so again uh try
[48:41 -> 48:45]  to maybe implement this yourself if
[48:43 -> 48:46]  you'd like a fun exercise uh and pause
[48:45 -> 48:47]  here otherwise I'm going to start
[48:46 -> 48:50]  putting in my
[48:47 -> 48:53]  solution so again there are many ways to
[48:50 -> 48:57]  do this so um this is one of the ways
[48:53 -> 48:59]  that sort of I came came up with so the
[48:57 -> 49:00]  first thing we're going to do is we are
[48:59 -> 49:03]  going
[49:00 -> 49:05]  to uh take our text encode it into utf8
[49:03 -> 49:07]  to get the raw bytes and then as before
[49:05 -> 49:10]  we're going to call list on the bytes
[49:07 -> 49:12]  object to get a list of integers of
[49:10 -> 49:14]  those bytes so those are the starting
[49:12 -> 49:16]  tokens those are the raw bytes of our
[49:14 -> 49:19]  sequence but now of course according to
[49:16 -> 49:21]  the merges dictionary above and recall
[49:19 -> 49:23]  this was the
[49:21 -> 49:26]  merges some of the bytes may be merged
[49:23 -> 49:28]  according to this lookup in addition to
[49:26 -> 49:29]  that remember that the merges was built
[49:28 -> 49:31]  from top to bottom and this is sort of
[49:29 -> 49:34]  the order in which we inserted stuff
[49:31 -> 49:36]  into merges and so we prefer to do all
[49:34 -> 49:39]  these merges in the beginning before we
[49:36 -> 49:40]  do these merges later because um for
[49:39 -> 49:44]  example this merge over here relies on
[49:40 -> 49:46]  the 256 which got merged here so we have
[49:44 -> 49:48]  to go in the order from top to bottom
[49:46 -> 49:51]  sort of if we are going to be merging
[49:48 -> 49:54]  anything now we expect to be doing a few
[49:51 -> 49:58]  merges so we're going to be doing W
[49:54 -> 50:00]  true um and now we want to find a pair
[49:58 -> 50:03]  of byes that is consecutive that we are
[50:00 -> 50:05]  allowed to merge according to this in
[50:03 -> 50:06]  order to reuse some of the functionality
[50:05 -> 50:09]  that we've already written I'm going to
[50:06 -> 50:12]  reuse the function uh get
[50:09 -> 50:14]  stats so recall that get stats uh will
[50:12 -> 50:16]  give us the we'll basically count up how
[50:14 -> 50:18]  many times every single pair occurs in
[50:16 -> 50:22]  our sequence of tokens and return that
[50:18 -> 50:25]  as a dictionary and the dictionary was a
[50:22 -> 50:27]  mapping from all the different uh by
[50:25 -> 50:30]  pairs to the number of times that they
[50:27 -> 50:32]  occur right um at this point we don't
[50:30 -> 50:34]  actually care how many times they occur
[50:32 -> 50:36]  in the sequence we only care what the
[50:34 -> 50:38]  raw pairs are in that sequence and so
[50:36 -> 50:40]  I'm only going to be using basically the
[50:38 -> 50:42]  keys of the dictionary I only care about
[50:40 -> 50:43]  the set of possible merge candidates if
[50:42 -> 50:46]  that makes
[50:43 -> 50:47]  sense now we want to identify the pair
[50:46 -> 50:50]  that we're going to be merging at this
[50:47 -> 50:53]  stage of the loop so what do we want we
[50:50 -> 50:57]  want to find the pair or like the a key
[50:53 -> 50:59]  inside stats that has the lowest index
[50:57 -> 51:01]  in the merges uh dictionary because we
[50:59 -> 51:03]  want to do all the early merges before
[51:01 -> 51:05]  we work our way to the late
[51:03 -> 51:07]  merges so again there are many different
[51:05 -> 51:11]  ways to implement this but I'm going to
[51:07 -> 51:14]  do something a little bit fancy
[51:11 -> 51:16]  here so I'm going to be using the Min
[51:14 -> 51:18]  over an iterator in Python when you call
[51:16 -> 51:20]  Min on an iterator and stats here as a
[51:18 -> 51:24]  dictionary we're going to be iterating
[51:20 -> 51:27]  the keys of this dictionary in Python so
[51:24 -> 51:29]  we're looking at all the pairs inside
[51:27 -> 51:32]  stats um which are all the consecutive
[51:29 -> 51:34]  Pairs and we're going to be taking the
[51:32 -> 51:38]  consecutive pair inside tokens that has
[51:34 -> 51:40]  the minimum what the Min takes a key
[51:38 -> 51:42]  which gives us the function that is
[51:40 -> 51:44]  going to return a value over which we're
[51:42 -> 51:46]  going to do the Min and the one we care
[51:44 -> 51:50]  about is we're we care about taking
[51:46 -> 51:52]  merges and basically getting um that
[51:50 -> 51:57]  pairs
[51:52 -> 51:59]  index so basically for any pair inside
[51:57 -> 52:03]  stats we are going to be looking into
[51:59 -> 52:05]  merges at what index it has and we want
[52:03 -> 52:07]  to get the pair with the Min number so
[52:05 -> 52:10]  as an example if there's a pair 101 and
[52:07 -> 52:11]  32 we definitely want to get that pair
[52:10 -> 52:15]  uh we want to identify it here and
[52:11 -> 52:15]  return it and pair would become 10132 if
[52:15 -> 52:17]  it
[52:15 -> 52:21]  occurs and the reason that I'm putting a
[52:17 -> 52:24]  float INF here as a fall back is that in
[52:21 -> 52:26]  the get function when we call uh when we
[52:24 -> 52:29]  basically consider a pair that doesn't
[52:26 -> 52:31]  occur in the merges then that pair is
[52:29 -> 52:33]  not eligible to be merged right so if in
[52:31 -> 52:35]  the token sequence there's some pair
[52:33 -> 52:38]  that is not a merging pair it cannot be
[52:35 -> 52:40]  merged then uh it doesn't actually occur
[52:38 -> 52:42]  here and it doesn't have an index and uh
[52:40 -> 52:45]  it cannot be merged which we will denote
[52:42 -> 52:46]  as float INF and the reason Infinity is
[52:45 -> 52:48]  nice here is because for sure we're
[52:46 -> 52:50]  guaranteed that it's not going to
[52:48 -> 52:53]  participate in the list of candidates
[52:50 -> 52:55]  when we do the men so uh so this is one
[52:53 -> 52:58]  way to do it so B basically long story
[52:55 -> 53:01]  short this Returns the most eligible
[52:58 -> 53:04]  merging candidate pair uh that occurs in
[53:01 -> 53:07]  the tokens now one thing to be careful
[53:04 -> 53:09]  with here is this uh function here might
[53:07 -> 53:13]  fail in the following way if there's
[53:09 -> 53:16]  nothing to merge then uh uh then there's
[53:13 -> 53:18]  nothing in merges um that satisfi that
[53:16 -> 53:21]  is satisfied anymore there's nothing to
[53:18 -> 53:23]  merge everything just returns float imps
[53:21 -> 53:26]  and then the pair I think will just
[53:23 -> 53:28]  become the very first element of stats
[53:26 -> 53:31]  um but this pair is not actually a
[53:28 -> 53:33]  mergeable pair it just becomes the first
[53:31 -> 53:36]  pair inside stats arbitrarily because
[53:33 -> 53:38]  all of these pairs evaluate to float in
[53:36 -> 53:40]  for the merging Criterion so basically
[53:38 -> 53:41]  it could be that this this doesn't look
[53:40 -> 53:44]  succeed because there's no more merging
[53:41 -> 53:46]  pairs so if this pair is not in merges
[53:44 -> 53:48]  that was returned then this is a signal
[53:46 -> 53:50]  for us that actually there was nothing
[53:48 -> 53:53]  to merge no single pair can be merged
[53:50 -> 53:57]  anymore in that case we will break
[53:53 -> 53:57]  out um nothing else can be
[53:57 -> 54:01]  merged you may come up with a different
[53:59 -> 54:03]  implementation by the way this is kind
[54:01 -> 54:05]  of like really trying hard in
[54:03 -> 54:07]  Python um but really we're just trying
[54:05 -> 54:09]  to find a pair that can be merged with
[54:07 -> 54:13]  the lowest index
[54:09 -> 54:16]  here now if we did find a pair that is
[54:13 -> 54:19]  inside merges with the lowest index then
[54:16 -> 54:19]  we can merge it
[54:19 -> 54:24]  so we're going to look into the merger
[54:22 -> 54:27]  dictionary for that pair to look up the
[54:24 -> 54:29]  index and we're going to now merge that
[54:27 -> 54:32]  into that index so we're going to do
[54:29 -> 54:34]  tokens equals and we're going to
[54:32 -> 54:36]  replace the original tokens we're going
[54:34 -> 54:38]  to be replacing the pair pair and we're
[54:36 -> 54:41]  going to be replacing it with index idx
[54:38 -> 54:43]  and this returns a new list of tokens
[54:41 -> 54:46]  where every occurrence of pair is
[54:43 -> 54:47]  replaced with idx so we're doing a merge
[54:46 -> 54:49]  and we're going to be continuing this
[54:47 -> 54:51]  until eventually nothing can be merged
[54:49 -> 54:53]  we'll come out here and we'll break out
[54:51 -> 54:55]  and here we just return
[54:53 -> 54:57]  tokens and so that that's the
[54:55 -> 55:02]  implementation I think so hopefully this
[54:57 -> 55:04]  runs okay cool um yeah and this looks uh
[55:02 -> 55:09]  reasonable so for example 32 is a space
[55:04 -> 55:11]  in asky so that's here um so this looks
[55:09 -> 55:13]  like it worked great okay so let's wrap
[55:11 -> 55:14]  up this section of the video at least I
[55:13 -> 55:16]  wanted to point out that this is not
[55:14 -> 55:17]  quite the right implementation just yet
[55:16 -> 55:20]  because we are leaving out a special
[55:17 -> 55:23]  case so in particular if uh we try to do
[55:20 -> 55:25]  this this would give us an error and the
[55:23 -> 55:28]  issue is that um if we only have a
[55:25 -> 55:29]  single character or an empty string then
[55:28 -> 55:32]  stats is empty and that causes an issue
[55:29 -> 55:36]  inside Min so one way to fight this is
[55:32 -> 55:37]  if L of tokens is at least two because
[55:36 -> 55:40]  if it's less than two it's just a single
[55:37 -> 55:41]  token or no tokens then let's just uh
[55:40 -> 55:44]  there's nothing to merge so we just
[55:41 -> 55:48]  return so that would fix uh that
[55:44 -> 55:50]  case Okay and then second I have a few
[55:48 -> 55:53]  test cases here for us as well so first
[55:50 -> 55:56]  let's make sure uh about or let's note
[55:53 -> 55:58]  the following if we take a string and we
[55:56 -> 56:00]  try to encode it and then decode it back
[55:58 -> 56:03]  you'd expect to get the same string back
[56:00 -> 56:03]  right is that true for all
[56:04 -> 56:08]  strings so I think uh so here it is the
[56:07 -> 56:12]  case and I think in general this is
[56:08 -> 56:14]  probably the case um but notice that
[56:12 -> 56:15]  going backwards is not is not you're not
[56:14 -> 56:19]  going to have an identity going
[56:15 -> 56:22]  backwards because as I mentioned us not
[56:19 -> 56:25]  all token sequences are valid utf8 uh
[56:22 -> 56:27]  sort of by streams and so so therefore
[56:25 -> 56:30]  you're some of them can't even be
[56:27 -> 56:32]  decodable um so this only goes in One
[56:30 -> 56:34]  Direction but for that one direction we
[56:32 -> 56:36]  can check uh here if we take the
[56:34 -> 56:38]  training text which is the text that we
[56:36 -> 56:39]  train to tokenizer around we can make
[56:38 -> 56:41]  sure that when we encode and decode we
[56:39 -> 56:43]  get the same thing back which is true
[56:41 -> 56:45]  and here I took some validation data so
[56:43 -> 56:47]  I went to I think this web page and I
[56:45 -> 56:49]  grabbed some text so this is text that
[56:47 -> 56:52]  the tokenizer has not seen and we can
[56:49 -> 56:53]  make sure that this also works um okay
[56:52 -> 56:56]  so that gives us some confidence that
[56:53 -> 56:58]  this was correctly implemented
[56:56 -> 57:00]  so those are the basics of the bite pair
[56:58 -> 57:03]  encoding algorithm we saw how we can uh
[57:00 -> 57:05]  take some training set train a tokenizer
[57:03 -> 57:08]  the parameters of this tokenizer really
[57:05 -> 57:09]  are just this dictionary of merges and
[57:08 -> 57:11]  that basically creates the little binary
[57:09 -> 57:14]  Forest on top of raw
[57:11 -> 57:16]  bites once we have this the merges table
[57:14 -> 57:19]  we can both encode and decode between
[57:16 -> 57:21]  raw text and token sequences so that's
[57:19 -> 57:23]  the the simplest setting of The
[57:21 -> 57:24]  tokenizer what we're going to do now
[57:23 -> 57:26]  though is we're going to look at some of
[57:24 -> 57:28]  the St the art lar language models and
[57:26 -> 57:29]  the kinds of tokenizers that they use
[57:28 -> 57:31]  and we're going to see that this picture
[57:29 -> 57:34]  complexifies very quickly so we're going
[57:31 -> 57:37]  to go through the details of this comp
[57:34 -> 57:39]  complexification one at a time so let's
[57:37 -> 57:41]  kick things off by looking at the GPD
[57:39 -> 57:44]  Series so in particular I have the gpt2
[57:41 -> 57:48]  paper here um and this paper is from
[57:44 -> 57:51]  2019 or so so 5 years ago and let's
[57:48 -> 57:52]  scroll down to input representation this
[57:51 -> 57:55]  is where they talk about the tokenizer
[57:52 -> 57:57]  that they're using for gpd2 now this is
[57:55 -> 58:00]  all fairly readable so I encourage you
[57:57 -> 58:02]  to pause and um read this yourself but
[58:00 -> 58:04]  this is where they motivate the use of
[58:02 -> 58:07]  the bite pair encoding algorithm on the
[58:04 -> 58:09]  bite level representation of utf8
[58:07 -> 58:11]  encoding so this is where they motivate
[58:09 -> 58:13]  it and they talk about the vocabulary
[58:11 -> 58:15]  sizes and everything now everything here
[58:13 -> 58:18]  is exactly as we've covered it so far
[58:15 -> 58:20]  but things start to depart around here
[58:18 -> 58:22]  so what they mention is that they don't
[58:20 -> 58:25]  just apply the naive algorithm as we
[58:22 -> 58:27]  have done it and in particular here's a
[58:25 -> 58:29]  example suppose that you have common
[58:27 -> 58:31]  words like dog what will happen is that
[58:29 -> 58:34]  dog of course occurs very frequently in
[58:31 -> 58:36]  the text and it occurs right next to all
[58:34 -> 58:39]  kinds of punctuation as an example so
[58:36 -> 58:42]  doc dot dog exclamation mark dog
[58:39 -> 58:43]  question mark Etc and naively you might
[58:42 -> 58:45]  imagine that the BP algorithm could
[58:43 -> 58:47]  merge these to be single tokens and then
[58:45 -> 58:49]  you end up with lots of tokens that are
[58:47 -> 58:50]  just like dog with a slightly different
[58:49 -> 58:52]  punctuation and so it feels like you're
[58:50 -> 58:53]  clustering things that shouldn't be
[58:52 -> 58:55]  clustered you're combining kind of
[58:53 -> 58:58]  semantics with
[58:55 -> 59:00]  uation and this uh feels suboptimal and
[58:58 -> 59:02]  indeed they also say that this is
[59:00 -> 59:04]  suboptimal according to some of the
[59:02 -> 59:06]  experiments so what they want to do is
[59:04 -> 59:09]  they want to top down in a manual way
[59:06 -> 59:12]  enforce that some types of um characters
[59:09 -> 59:14]  should never be merged together um so
[59:12 -> 59:17]  they want to enforce these merging rules
[59:14 -> 59:19]  on top of the bite PA encoding algorithm
[59:17 -> 59:21]  so let's take a look um at their code
[59:19 -> 59:23]  and see how they actually enforce this
[59:21 -> 59:25]  and what kinds of mergy they actually do
[59:23 -> 59:29]  perform so I have to to tab open here
[59:25 -> 59:30]  for gpt2 under open AI on GitHub and
[59:29 -> 59:34]  when we go to
[59:30 -> 59:35]  Source there is an encoder thatp now I
[59:34 -> 59:37]  don't personally love that they call it
[59:35 -> 59:39]  encoder dopy because this is the
[59:37 -> 59:41]  tokenizer and the tokenizer can do both
[59:39 -> 59:43]  encode and decode uh so it feels kind of
[59:41 -> 59:45]  awkward to me that it's called encoder
[59:43 -> 59:47]  but that is the tokenizer and there's a
[59:45 -> 59:49]  lot going on here and we're going to
[59:47 -> 59:51]  step through it in detail at one point
[59:49 -> 59:54]  for now I just want to focus on this
[59:51 -> 59:56]  part here the create a rigix pattern
[59:54 -> 59:58]  here that looks very complicated and
[59:56 -> 01:00:00]  we're going to go through it in a bit uh
[59:58 -> 01:00:04]  but this is the core part that allows
[01:00:00 -> 01:00:05]  them to enforce rules uh for what parts
[01:00:04 -> 01:00:08]  of the text Will Never Be merged for
[01:00:05 -> 01:00:10]  sure now notice that re. compile here is
[01:00:08 -> 01:00:12]  a little bit misleading because we're
[01:00:10 -> 01:00:14]  not just doing import re which is the
[01:00:12 -> 01:00:17]  python re module we're doing import reex
[01:00:14 -> 01:00:20]  as re and reex is a python package that
[01:00:17 -> 01:00:22]  you can install P install r x and it's
[01:00:20 -> 01:00:23]  basically an extension of re so it's a
[01:00:22 -> 01:00:26]  bit more powerful
[01:00:23 -> 01:00:28]  re um
[01:00:26 -> 01:00:30]  so let's take a look at this pattern and
[01:00:28 -> 01:00:32]  what it's doing and why this is actually
[01:00:30 -> 01:00:34]  doing the separation that they are
[01:00:32 -> 01:00:37]  looking for okay so I've copy pasted the
[01:00:34 -> 01:00:39]  pattern here to our jupit notebook where
[01:00:37 -> 01:00:42]  we left off and let's take this pattern
[01:00:39 -> 01:00:44]  for a spin so in the exact same way that
[01:00:42 -> 01:00:47]  their code does we're going to call an
[01:00:44 -> 01:00:49]  re. findall for this pattern on any
[01:00:47 -> 01:00:50]  arbitrary string that we are interested
[01:00:49 -> 01:00:55]  so this is the string that we want to
[01:00:50 -> 01:00:59]  encode into tokens um to feed into n llm
[01:00:55 -> 01:01:01]  like gpt2 so what exactly is this doing
[01:00:59 -> 01:01:02]  well re. findall will take this pattern
[01:01:01 -> 01:01:06]  and try to match it against a
[01:01:02 -> 01:01:07]  string um the way this works is that you
[01:01:06 -> 01:01:10]  are going from left to right in the
[01:01:07 -> 01:01:13]  string and you're trying to match the
[01:01:10 -> 01:01:16]  pattern and R.F find all will get all
[01:01:13 -> 01:01:19]  the occurrences and organize them into a
[01:01:16 -> 01:01:20]  list now when you look at the um when
[01:01:19 -> 01:01:23]  you look at this pattern first of all
[01:01:20 -> 01:01:26]  notice that this is a raw string um and
[01:01:23 -> 01:01:28]  then these are three double quotes just
[01:01:26 -> 01:01:31]  to start the string so really the string
[01:01:28 -> 01:01:34]  itself this is the pattern itself
[01:01:31 -> 01:01:36]  right and notice that it's made up of a
[01:01:34 -> 01:01:40]  lot of ores so see these vertical bars
[01:01:36 -> 01:01:41]  those are ores in reg X and so you go
[01:01:40 -> 01:01:43]  from left to right in this pattern and
[01:01:41 -> 01:01:46]  try to match it against the string
[01:01:43 -> 01:01:48]  wherever you are so we have hello and
[01:01:46 -> 01:01:50]  we're going to try to match it well it's
[01:01:48 -> 01:01:53]  not apostrophe s it's not apostrophe t
[01:01:50 -> 01:01:58]  or any of these but it is an optional
[01:01:53 -> 01:02:02]  space followed by- P of uh sorry SL P of
[01:01:58 -> 01:02:04]  L one or more times what is/ P of L it
[01:02:02 -> 01:02:08]  is coming to some documentation that I
[01:02:04 -> 01:02:11]  found um there might be other sources as
[01:02:08 -> 01:02:15]  well uh SLP is a letter any kind of
[01:02:11 -> 01:02:19]  letter from any language and hello is
[01:02:15 -> 01:02:21]  made up of letters h e l Etc so optional
[01:02:19 -> 01:02:24]  space followed by a bunch of letters one
[01:02:21 -> 01:02:27]  or more letters is going to match hello
[01:02:24 -> 01:02:31]  but then the match ends because a white
[01:02:27 -> 01:02:33]  space is not a letter so from there on
[01:02:31 -> 01:02:36]  begins a new sort of attempt to match
[01:02:33 -> 01:02:38]  against the string again and starting in
[01:02:36 -> 01:02:40]  here we're going to skip over all of
[01:02:38 -> 01:02:42]  these again until we get to the exact
[01:02:40 -> 01:02:44]  same Point again and we see that there's
[01:02:42 -> 01:02:46]  an optional space this is the optional
[01:02:44 -> 01:02:48]  space followed by a bunch of letters one
[01:02:46 -> 01:02:52]  or more of them and so that matches so
[01:02:48 -> 01:02:55]  when we run this we get a list of two
[01:02:52 -> 01:02:58]  elements hello and then space world
[01:02:55 -> 01:03:01]  so how are you if we add more letters we
[01:02:58 -> 01:03:03]  would just get them like this now what
[01:03:01 -> 01:03:05]  is this doing and why is this important
[01:03:03 -> 01:03:09]  we are taking our string and instead of
[01:03:05 -> 01:03:11]  directly encoding it um for
[01:03:09 -> 01:03:13]  tokenization we are first splitting it
[01:03:11 -> 01:03:15]  up and when you actually step through
[01:03:13 -> 01:03:17]  the code and we'll do that in a bit more
[01:03:15 -> 01:03:20]  detail what really is doing on a high
[01:03:17 -> 01:03:24]  level is that it first splits your text
[01:03:20 -> 01:03:26]  into a list of texts just like this one
[01:03:24 -> 01:03:29]  and all these elements of this list are
[01:03:26 -> 01:03:30]  processed independently by the tokenizer
[01:03:29 -> 01:03:32]  and all of the results of that
[01:03:30 -> 01:03:35]  processing are simply
[01:03:32 -> 01:03:39]  concatenated so hello world oh I I
[01:03:35 -> 01:03:41]  missed how hello world how are you we
[01:03:39 -> 01:03:44]  have five elements of list all of these
[01:03:41 -> 01:03:47]  will independent
[01:03:44 -> 01:03:49]  independently go from text to a token
[01:03:47 -> 01:03:50]  sequence and then that token sequence is
[01:03:49 -> 01:03:54]  going to be concatenated it's all going
[01:03:50 -> 01:03:56]  to be joined up and roughly speaking
[01:03:54 -> 01:03:58]  what that does is you're only ever
[01:03:56 -> 01:04:00]  finding merges between the elements of
[01:03:58 -> 01:04:01]  this list so you can only ever consider
[01:04:00 -> 01:04:03]  merges within every one of these
[01:04:01 -> 01:04:06]  elements in
[01:04:03 -> 01:04:07]  individually and um after you've done
[01:04:06 -> 01:04:09]  all the possible merging for all of
[01:04:07 -> 01:04:13]  these elements individually the results
[01:04:09 -> 01:04:16]  of all that will be joined um by
[01:04:13 -> 01:04:18]  concatenation and so you are basically
[01:04:16 -> 01:04:21]  what what you're doing effectively is
[01:04:18 -> 01:04:23]  you are never going to be merging this e
[01:04:21 -> 01:04:25]  with this space because they are now
[01:04:23 -> 01:04:27]  parts of the separate elements of this
[01:04:25 -> 01:04:28]  list and so you are saying we are never
[01:04:27 -> 01:04:32]  going to merge
[01:04:28 -> 01:04:35]  eace um because we're breaking it up in
[01:04:32 -> 01:04:37]  this way so basically using this regx
[01:04:35 -> 01:04:41]  pattern to Chunk Up the text is just one
[01:04:37 -> 01:04:43]  way of enforcing that some merges are
[01:04:41 -> 01:04:45]  not to happen and we're going to go into
[01:04:43 -> 01:04:46]  more of this text and we'll see that
[01:04:45 -> 01:04:48]  what this is trying to do on a high
[01:04:46 -> 01:04:50]  level is we're trying to not merge
[01:04:48 -> 01:04:53]  across letters across numbers across
[01:04:50 -> 01:04:54]  punctuation and so on so let's see in
[01:04:53 -> 01:04:58]  more detail how that works so let's
[01:04:54 -> 01:05:01]  continue now we have/ P ofn if you go to
[01:04:58 -> 01:05:04]  the documentation SLP of n is any kind
[01:05:01 -> 01:05:06]  of numeric character in any script so
[01:05:04 -> 01:05:08]  it's numbers so we have an optional
[01:05:06 -> 01:05:10]  space followed by numbers and those
[01:05:08 -> 01:05:12]  would be separated out so letters and
[01:05:10 -> 01:05:15]  numbers are being separated so if I do
[01:05:12 -> 01:05:17]  Hello World 123 how are you then world
[01:05:15 -> 01:05:20]  will stop matching here because one is
[01:05:17 -> 01:05:22]  not a letter anymore but one is a number
[01:05:20 -> 01:05:26]  so this group will match for that and
[01:05:22 -> 01:05:26]  we'll get it as a separate entity
[01:05:26 -> 01:05:31]  uh let's see how these apostrophes work
[01:05:28 -> 01:05:35]  so here if we have
[01:05:31 -> 01:05:38]  um uh Slash V or I mean apostrophe V as
[01:05:35 -> 01:05:39]  an example then apostrophe here is not a
[01:05:38 -> 01:05:42]  letter or a
[01:05:39 -> 01:05:44]  number so hello will stop matching and
[01:05:42 -> 01:05:48]  then we will exactly match this with
[01:05:44 -> 01:05:50]  that so that will come out as a separate
[01:05:48 -> 01:05:52]  thing so why are they doing the
[01:05:50 -> 01:05:53]  apostrophes here honestly I think that
[01:05:52 -> 01:05:56]  these are just like very common
[01:05:53 -> 01:05:59]  apostrophes p uh that are used um
[01:05:56 -> 01:06:00]  typically I don't love that they've done
[01:05:59 -> 01:06:03]  this
[01:06:00 -> 01:06:05]  because uh let me show you what happens
[01:06:03 -> 01:06:07]  when you have uh some Unicode
[01:06:05 -> 01:06:10]  apostrophes like for example you can
[01:06:07 -> 01:06:13]  have if you have house then this will be
[01:06:10 -> 01:06:15]  separated out because of this matching
[01:06:13 -> 01:06:16]  but if you use the Unicode apostrophe
[01:06:15 -> 01:06:19]  like
[01:06:16 -> 01:06:21]  this then suddenly this does not work
[01:06:19 -> 01:06:24]  and so this apostrophe will actually
[01:06:21 -> 01:06:26]  become its own thing now and so so um
[01:06:24 -> 01:06:29]  it's basically hardcoded for this
[01:06:26 -> 01:06:31]  specific kind of apostrophe and uh
[01:06:29 -> 01:06:34]  otherwise they become completely
[01:06:31 -> 01:06:38]  separate tokens in addition to this you
[01:06:34 -> 01:06:40]  can go to the gpt2 docs and here when
[01:06:38 -> 01:06:43]  they Define the pattern they say should
[01:06:40 -> 01:06:44]  have added re. ignore case so BP merges
[01:06:43 -> 01:06:46]  can happen for capitalized versions of
[01:06:44 -> 01:06:47]  contractions so what they're pointing
[01:06:46 -> 01:06:50]  out is that you see how this is
[01:06:47 -> 01:06:52]  apostrophe and then lowercase letters
[01:06:50 -> 01:06:56]  well because they didn't do re. ignore
[01:06:52 -> 01:06:58]  case then then um these rules will not
[01:06:56 -> 01:07:01]  separate out the apostrophes if it's
[01:06:58 -> 01:07:06]  uppercase so
[01:07:01 -> 01:07:10]  house would be like this but if I did
[01:07:06 -> 01:07:12]  house if I'm uppercase then notice
[01:07:10 -> 01:07:15]  suddenly the apostrophe comes by
[01:07:12 -> 01:07:17]  itself so the tokenization will work
[01:07:15 -> 01:07:19]  differently in uppercase and lower case
[01:07:17 -> 01:07:21]  inconsistently separating out these
[01:07:19 -> 01:07:24]  apostrophes so it feels extremely gnarly
[01:07:21 -> 01:07:27]  and slightly gross um but that's that's
[01:07:24 -> 01:07:28]  how that works okay so let's come back
[01:07:27 -> 01:07:30]  after trying to match a bunch of
[01:07:28 -> 01:07:32]  apostrophe Expressions by the way the
[01:07:30 -> 01:07:34]  other issue here is that these are quite
[01:07:32 -> 01:07:35]  language specific probably so I don't
[01:07:34 -> 01:07:37]  know that all the languages for example
[01:07:35 -> 01:07:39]  use or don't use apostrophes but that
[01:07:37 -> 01:07:42]  would be inconsistently tokenized as a
[01:07:39 -> 01:07:44]  result then we try to match letters then
[01:07:42 -> 01:07:47]  we try to match numbers and then if that
[01:07:44 -> 01:07:49]  doesn't work we fall back to here and
[01:07:47 -> 01:07:50]  what this is saying is again optional
[01:07:49 -> 01:07:53]  space followed by something that is not
[01:07:50 -> 01:07:55]  a letter number or a space in one or
[01:07:53 -> 01:07:57]  more of that so what this is doing
[01:07:55 -> 01:07:59]  effectively is this is trying to match
[01:07:57 -> 01:08:02]  punctuation roughly speaking not letters
[01:07:59 -> 01:08:04]  and not numbers so this group will try
[01:08:02 -> 01:08:08]  to trigger for that so if I do something
[01:08:04 -> 01:08:09]  like this then these parts here are not
[01:08:08 -> 01:08:12]  letters or numbers but they will
[01:08:09 -> 01:08:14]  actually they are uh they will actually
[01:08:12 -> 01:08:17]  get caught here and so they become its
[01:08:14 -> 01:08:20]  own group so we've separated out the
[01:08:17 -> 01:08:22]  punctuation and finally this um this is
[01:08:20 -> 01:08:25]  also a little bit confusing so this is
[01:08:22 -> 01:08:29]  matching white space but this is using a
[01:08:25 -> 01:08:30]  negative look ahead assertion in regex
[01:08:29 -> 01:08:33]  so what this is doing is it's matching
[01:08:30 -> 01:08:35]  wh space up to but not including the
[01:08:33 -> 01:08:37]  last Whit space
[01:08:35 -> 01:08:40]  character why is this important um this
[01:08:37 -> 01:08:41]  is pretty subtle I think so you see how
[01:08:40 -> 01:08:45]  the white space is always included at
[01:08:41 -> 01:08:48]  the beginning of the word so um space r
[01:08:45 -> 01:08:49]  space u Etc suppose we have a lot of
[01:08:48 -> 01:08:52]  spaces
[01:08:49 -> 01:08:54]  here what's going to happen here is that
[01:08:52 -> 01:08:57]  these spaces up to not including the
[01:08:54 -> 01:08:59]  last character will get caught by this
[01:08:57 -> 01:09:01]  and what that will do is it will
[01:08:59 -> 01:09:03]  separate out the spaces up to but not
[01:09:01 -> 01:09:05]  including the last character so that the
[01:09:03 -> 01:09:09]  last character can come here and join
[01:09:05 -> 01:09:11]  with the um space you and the reason
[01:09:09 -> 01:09:13]  that's nice is because space you is the
[01:09:11 -> 01:09:15]  common token so if I didn't have these
[01:09:13 -> 01:09:18]  Extra Spaces here you would just have
[01:09:15 -> 01:09:20]  space you and if I add tokens if I add
[01:09:18 -> 01:09:22]  spaces we still have a space view but
[01:09:20 -> 01:09:24]  now we have all this extra white space
[01:09:22 -> 01:09:27]  so basically the GB to tokenizer really
[01:09:24 -> 01:09:30]  likes to have a space letters or numbers
[01:09:27 -> 01:09:31]  um and it it preens these spaces and
[01:09:30 -> 01:09:33]  this is just something that it is
[01:09:31 -> 01:09:36]  consistent about so that's what that is
[01:09:33 -> 01:09:38]  for and then finally we have all the the
[01:09:36 -> 01:09:42]  last fallback is um whites space
[01:09:38 -> 01:09:46]  characters uh so um that would be
[01:09:42 -> 01:09:48]  just um if that doesn't get caught then
[01:09:46 -> 01:09:50]  this thing will catch any trailing
[01:09:48 -> 01:09:53]  spaces and so on I wanted to show one
[01:09:50 -> 01:09:54]  more real world example here so if we
[01:09:53 -> 01:09:56]  have this string which is a piece of
[01:09:54 -> 01:09:58]  python code and then we try to split it
[01:09:56 -> 01:10:00]  up then this is the kind of output we
[01:09:58 -> 01:10:02]  get so you'll notice that the list has
[01:10:00 -> 01:10:05]  many elements here and that's because we
[01:10:02 -> 01:10:07]  are splitting up fairly often uh every
[01:10:05 -> 01:10:09]  time sort of a category
[01:10:07 -> 01:10:10]  changes um so there will never be any
[01:10:09 -> 01:10:13]  merges Within These
[01:10:10 -> 01:10:16]  elements and um that's what you are
[01:10:13 -> 01:10:17]  seeing here now you might think that in
[01:10:16 -> 01:10:21]  order to train the
[01:10:17 -> 01:10:23]  tokenizer uh open AI has used this to
[01:10:21 -> 01:10:25]  split up text into chunks and then run
[01:10:23 -> 01:10:27]  just a BP algorithm within all the
[01:10:25 -> 01:10:30]  chunks but that is not exactly what
[01:10:27 -> 01:10:33]  happened and the reason is the following
[01:10:30 -> 01:10:35]  notice that we have the spaces here uh
[01:10:33 -> 01:10:38]  those Spaces end up being entire
[01:10:35 -> 01:10:40]  elements but these spaces never actually
[01:10:38 -> 01:10:42]  end up being merged by by open Ai and
[01:10:40 -> 01:10:44]  the way you can tell is that if you copy
[01:10:42 -> 01:10:47]  paste the exact same chunk here into Tik
[01:10:44 -> 01:10:49]  token U Tik tokenizer you see that all
[01:10:47 -> 01:10:51]  the spaces are kept independent and
[01:10:49 -> 01:10:53]  they're all token
[01:10:51 -> 01:10:56]  220 so I think opena at some point Point
[01:10:53 -> 01:10:59]  en Force some rule that these spaces
[01:10:56 -> 01:11:01]  would never be merged and so um there's
[01:10:59 -> 01:11:04]  some additional rules on top of just
[01:11:01 -> 01:11:06]  chunking and bpe that open ey is not uh
[01:11:04 -> 01:11:08]  clear about now the training code for
[01:11:06 -> 01:11:10]  the gpt2 tokenizer was never released so
[01:11:08 -> 01:11:13]  all we have is uh the code that I've
[01:11:10 -> 01:11:14]  already shown you but this code here
[01:11:13 -> 01:11:17]  that they've released is only the
[01:11:14 -> 01:11:19]  inference code for the tokens so this is
[01:11:17 -> 01:11:21]  not the training code you can't give it
[01:11:19 -> 01:11:23]  a piece of text and training tokenizer
[01:11:21 -> 01:11:25]  this is just the inference code which
[01:11:23 -> 01:11:28]  Tak takes the merges that we have up
[01:11:25 -> 01:11:30]  above and applies them to a new piece of
[01:11:28 -> 01:11:32]  text and so we don't know exactly how
[01:11:30 -> 01:11:34]  opening ey trained um train the
[01:11:32 -> 01:11:38]  tokenizer but it wasn't as simple as
[01:11:34 -> 01:11:40]  chunk it up and BP it uh whatever it was
[01:11:38 -> 01:11:42]  next I wanted to introduce you to the
[01:11:40 -> 01:11:44]  Tik token library from openai which is
[01:11:42 -> 01:11:48]  the official library for tokenization
[01:11:44 -> 01:11:51]  from openai so this is Tik token bip
[01:11:48 -> 01:11:54]  install P to Tik token and then um you
[01:11:51 -> 01:11:55]  can do the tokenization in inference
[01:11:54 -> 01:11:57]  this is again not training code this is
[01:11:55 -> 01:12:00]  only inference code for
[01:11:57 -> 01:12:02]  tokenization um I wanted to show you how
[01:12:00 -> 01:12:04]  you would use it quite simple and
[01:12:02 -> 01:12:06]  running this just gives us the gpt2
[01:12:04 -> 01:12:09]  tokens or the GPT 4 tokens so this is
[01:12:06 -> 01:12:11]  the tokenizer use for GPT 4 and so in
[01:12:09 -> 01:12:14]  particular we see that the Whit space in
[01:12:11 -> 01:12:17]  gpt2 remains unmerged but in GPT 4 uh
[01:12:14 -> 01:12:19]  these Whit spaces merge as we also saw
[01:12:17 -> 01:12:22]  in this one where here they're all
[01:12:19 -> 01:12:25]  unmerged but if we go down to GPT 4 uh
[01:12:22 -> 01:12:27]  they become merged
[01:12:25 -> 01:12:31]  um now in the
[01:12:27 -> 01:12:33]  gp4 uh tokenizer they changed the
[01:12:31 -> 01:12:35]  regular expression that they use to
[01:12:33 -> 01:12:38]  Chunk Up text so the way to see this is
[01:12:35 -> 01:12:41]  that if you come to your the Tik token
[01:12:38 -> 01:12:44]  uh library and then you go to this file
[01:12:41 -> 01:12:45]  Tik token X openi public this is where
[01:12:44 -> 01:12:46]  sort of like the definition of all these
[01:12:45 -> 01:12:50]  different tokenizers that openi
[01:12:46 -> 01:12:51]  maintains is and so uh necessarily to do
[01:12:50 -> 01:12:53]  the inference they had to publish some
[01:12:51 -> 01:12:55]  of the details about the strings
[01:12:53 -> 01:12:58]  so this is the string that we already
[01:12:55 -> 01:13:00]  saw for gpt2 it is slightly different
[01:12:58 -> 01:13:02]  but it is actually equivalent uh to what
[01:13:00 -> 01:13:04]  we discussed here so this pattern that
[01:13:02 -> 01:13:07]  we discussed is equivalent to this
[01:13:04 -> 01:13:09]  pattern this one just executes a little
[01:13:07 -> 01:13:10]  bit faster so here you see a little bit
[01:13:09 -> 01:13:12]  of a slightly different definition but
[01:13:10 -> 01:13:15]  otherwise it's the same we're going to
[01:13:12 -> 01:13:18]  go into special tokens in a bit and then
[01:13:15 -> 01:13:20]  if you scroll down to CL 100k this is
[01:13:18 -> 01:13:23]  the GPT 4 tokenizer you see that the
[01:13:20 -> 01:13:26]  pattern has changed um and this is kind
[01:13:23 -> 01:13:27]  of like the main the major change in
[01:13:26 -> 01:13:30]  addition to a bunch of other special
[01:13:27 -> 01:13:31]  tokens which I'll go into in a bit again
[01:13:30 -> 01:13:33]  now some I'm not going to actually go
[01:13:31 -> 01:13:35]  into the full detail of the pattern
[01:13:33 -> 01:13:37]  change because honestly this is my
[01:13:35 -> 01:13:39]  numbing uh I would just advise that you
[01:13:37 -> 01:13:42]  pull out chat GPT and the regex
[01:13:39 -> 01:13:44]  documentation and just step through it
[01:13:42 -> 01:13:48]  but really the major changes are number
[01:13:44 -> 01:13:51]  one you see this eye here that means
[01:13:48 -> 01:13:53]  that the um case sensitivity this is
[01:13:51 -> 01:13:56]  case insensitive match and so the
[01:13:53 -> 01:13:58]  comment that we saw earlier on oh we
[01:13:56 -> 01:14:01]  should have used re. uppercase uh
[01:13:58 -> 01:14:04]  basically we're now going to be matching
[01:14:01 -> 01:14:06]  these apostrophe s apostrophe D
[01:14:04 -> 01:14:08]  apostrophe M Etc uh we're going to be
[01:14:06 -> 01:14:11]  matching them both in lowercase and in
[01:14:08 -> 01:14:12]  uppercase so that's fixed there's a
[01:14:11 -> 01:14:14]  bunch of different like handling of the
[01:14:12 -> 01:14:16]  whites space that I'm not going to go
[01:14:14 -> 01:14:18]  into the full details of and then one
[01:14:16 -> 01:14:20]  more thing here is you will notice that
[01:14:18 -> 01:14:23]  when they match the numbers they only
[01:14:20 -> 01:14:26]  match one to three numbers so so they
[01:14:23 -> 01:14:28]  will never merge
[01:14:26 -> 01:14:31]  numbers that are in low in more than
[01:14:28 -> 01:14:34]  three digits only up to three digits of
[01:14:31 -> 01:14:36]  numbers will ever be merged and uh
[01:14:34 -> 01:14:38]  that's one change that they made as well
[01:14:36 -> 01:14:40]  to prevent uh tokens that are very very
[01:14:38 -> 01:14:42]  long number
[01:14:40 -> 01:14:44]  sequences uh but again we don't really
[01:14:42 -> 01:14:46]  know why they do any of this stuff uh
[01:14:44 -> 01:14:49]  because none of this is documented and
[01:14:46 -> 01:14:51]  uh it's just we just get the pattern so
[01:14:49 -> 01:14:54]  um yeah it is what it is but those are
[01:14:51 -> 01:14:56]  some of the changes that gp4 has made
[01:14:54 -> 01:14:58]  and of course the vocabulary size went
[01:14:56 -> 01:15:00]  from roughly 50k to roughly
[01:14:58 -> 01:15:02]  100K the next thing I would like to do
[01:15:00 -> 01:15:05]  very briefly is to take you through the
[01:15:02 -> 01:15:07]  gpt2 encoder dopy that openi has
[01:15:05 -> 01:15:09]  released uh this is the file that I
[01:15:07 -> 01:15:12]  already mentioned to you briefly now
[01:15:09 -> 01:15:14]  this file is uh fairly short and should
[01:15:12 -> 01:15:17]  be relatively understandable to you at
[01:15:14 -> 01:15:21]  this point um starting at the bottom
[01:15:17 -> 01:15:24]  here they are loading two files encoder
[01:15:21 -> 01:15:25]  Json and vocab bpe and they do some
[01:15:24 -> 01:15:27]  light processing on it and then they
[01:15:25 -> 01:15:30]  call this encoder object which is the
[01:15:27 -> 01:15:31]  tokenizer now if you'd like to inspect
[01:15:30 -> 01:15:34]  these two files which together
[01:15:31 -> 01:15:36]  constitute their saved tokenizer then
[01:15:34 -> 01:15:36]  you can do that with a piece of code
[01:15:36 -> 01:15:39]  like
[01:15:36 -> 01:15:40]  this um this is where you can download
[01:15:39 -> 01:15:42]  these two files and you can inspect them
[01:15:40 -> 01:15:45]  if you'd like and what you will find is
[01:15:42 -> 01:15:47]  that this encoder as they call it in
[01:15:45 -> 01:15:51]  their code is exactly equivalent to our
[01:15:47 -> 01:15:53]  vocab so remember here where we have
[01:15:51 -> 01:15:56]  this vocab object which allowed us us to
[01:15:53 -> 01:16:00]  decode very efficiently and basically it
[01:15:56 -> 01:16:03]  took us from the integer to the byes uh
[01:16:00 -> 01:16:07]  for that integer so our vocab is exactly
[01:16:03 -> 01:16:11]  their encoder and then their vocab bpe
[01:16:07 -> 01:16:14]  confusingly is actually are merges so
[01:16:11 -> 01:16:16]  their BP merges which is based on the
[01:16:14 -> 01:16:20]  data inside vocab bpe ends up being
[01:16:16 -> 01:16:24]  equivalent to our merges so uh basically
[01:16:20 -> 01:16:26]  they are saving and loading the two uh
[01:16:24 -> 01:16:28]  variables that for us are also critical
[01:16:26 -> 01:16:31]  the merges variable and the vocab
[01:16:28 -> 01:16:32]  variable using just these two variables
[01:16:31 -> 01:16:34]  you can represent a tokenizer and you
[01:16:32 -> 01:16:36]  can both do encoding and decoding once
[01:16:34 -> 01:16:40]  you've trained this
[01:16:36 -> 01:16:42]  tokenizer now the only thing that um is
[01:16:40 -> 01:16:44]  actually slightly confusing inside what
[01:16:42 -> 01:16:46]  opening ey does here is that in addition
[01:16:44 -> 01:16:48]  to this encoder and a decoder they also
[01:16:46 -> 01:16:51]  have something called a bite encoder and
[01:16:48 -> 01:16:53]  a bite decoder and this is actually
[01:16:51 -> 01:16:55]  unfortunately just
[01:16:53 -> 01:16:57]  kind of a spirous implementation detail
[01:16:55 -> 01:16:59]  and isn't actually deep or interesting
[01:16:57 -> 01:17:01]  in any way so I'm going to skip the
[01:16:59 -> 01:17:02]  discussion of it but what opening ey
[01:17:01 -> 01:17:05]  does here for reasons that I don't fully
[01:17:02 -> 01:17:06]  understand is that not only have they
[01:17:05 -> 01:17:08]  this tokenizer which can encode and
[01:17:06 -> 01:17:10]  decode but they have a whole separate
[01:17:08 -> 01:17:12]  layer here in addition that is used
[01:17:10 -> 01:17:16]  serially with the tokenizer and so you
[01:17:12 -> 01:17:17]  first do um bite encode and then encode
[01:17:16 -> 01:17:20]  and then you do decode and then bite
[01:17:17 -> 01:17:22]  decode so that's the loop and they are
[01:17:20 -> 01:17:24]  just stacked serial on top of each other
[01:17:22 -> 01:17:25]  and and it's not that interesting so I
[01:17:24 -> 01:17:28]  won't cover it and you can step through
[01:17:25 -> 01:17:30]  it if you'd like otherwise this file if
[01:17:28 -> 01:17:31]  you ignore the bite encoder and the bite
[01:17:30 -> 01:17:33]  decoder will be algorithmically very
[01:17:31 -> 01:17:37]  familiar with you and the meat of it
[01:17:33 -> 01:17:39]  here is the what they call bpe function
[01:17:37 -> 01:17:41]  and you should recognize this Loop here
[01:17:39 -> 01:17:43]  which is very similar to our own y Loop
[01:17:41 -> 01:17:46]  where they're trying to identify the
[01:17:43 -> 01:17:49]  Byram uh a pair that they should be
[01:17:46 -> 01:17:50]  merging next and then here just like we
[01:17:49 -> 01:17:53]  had they have a for Loop trying to merge
[01:17:50 -> 01:17:55]  this pair uh so they will go over all of
[01:17:53 -> 01:17:57]  the sequence and they will merge the
[01:17:55 -> 01:17:59]  pair whenever they find it and they keep
[01:17:57 -> 01:18:02]  repeating that until they run out of
[01:17:59 -> 01:18:04]  possible merges in the in the text so
[01:18:02 -> 01:18:06]  that's the meat of this file and uh
[01:18:04 -> 01:18:08]  there's an encode and a decode function
[01:18:06 -> 01:18:09]  just like we have implemented it so long
[01:18:08 -> 01:18:11]  story short what I want you to take away
[01:18:09 -> 01:18:13]  at this point is that unfortunately it's
[01:18:11 -> 01:18:15]  a little bit of a messy code that they
[01:18:13 -> 01:18:17]  have but algorithmically it is identical
[01:18:15 -> 01:18:19]  to what we've built up above and what
[01:18:17 -> 01:18:21]  we've built up above if you understand
[01:18:19 -> 01:18:23]  it is algorithmically what is necessary
[01:18:21 -> 01:18:26]  to actually build a BP to organizer
[01:18:23 -> 01:18:28]  train it and then both encode and decode
[01:18:26 -> 01:18:30]  the next topic I would like to turn to
[01:18:28 -> 01:18:32]  is that of special tokens so in addition
[01:18:30 -> 01:18:35]  to tokens that are coming from you know
[01:18:32 -> 01:18:36]  raw bytes and the BP merges we can
[01:18:35 -> 01:18:38]  insert all kinds of tokens that we are
[01:18:36 -> 01:18:41]  going to use to delimit different parts
[01:18:38 -> 01:18:44]  of the data or introduced to create a
[01:18:41 -> 01:18:47]  special structure of the token streams
[01:18:44 -> 01:18:50]  so in uh if you look at this encoder
[01:18:47 -> 01:18:52]  object from open AIS gpd2 right here we
[01:18:50 -> 01:18:54]  mentioned this is very similar to our
[01:18:52 -> 01:18:57]  vocab you'll notice that the length of
[01:18:54 -> 01:18:57]  this is
[01:18:58 -> 01:19:03]  50257 and as I mentioned it's mapping uh
[01:19:01 -> 01:19:06]  and it's inverted from the mapping of
[01:19:03 -> 01:19:08]  our vocab our vocab goes from integer to
[01:19:06 -> 01:19:11]  string and they go the other way around
[01:19:08 -> 01:19:13]  for no amazing reason um but the thing
[01:19:11 -> 01:19:15]  to note here is that this the mapping
[01:19:13 -> 01:19:18]  table here is
[01:19:15 -> 01:19:20]  50257 where does that number come from
[01:19:18 -> 01:19:24]  where what are the tokens as I mentioned
[01:19:20 -> 01:19:27]  there are 256 raw bite token
[01:19:24 -> 01:19:28]  tokens and then opena actually did
[01:19:27 -> 01:19:32]  50,000
[01:19:28 -> 01:19:34]  merges so those become the other tokens
[01:19:32 -> 01:19:37]  but this would have been
[01:19:34 -> 01:19:40]  50256 so what is the 57th token and
[01:19:37 -> 01:19:43]  there is basically one special
[01:19:40 -> 01:19:47]  token and that one special token you can
[01:19:43 -> 01:19:49]  see is called end of text so this is a
[01:19:47 -> 01:19:52]  special token and it's the very last
[01:19:49 -> 01:19:55]  token and this token is used to delimit
[01:19:52 -> 01:19:57]  documents ments in the training set so
[01:19:55 -> 01:19:59]  when we're creating the training data we
[01:19:57 -> 01:20:01]  have all these documents and we tokenize
[01:19:59 -> 01:20:05]  them and we get a stream of tokens those
[01:20:01 -> 01:20:07]  tokens only range from Z to
[01:20:05 -> 01:20:10]  50256 and then in between those
[01:20:07 -> 01:20:12]  documents we put special end of text
[01:20:10 -> 01:20:15]  token and we insert that token in
[01:20:12 -> 01:20:18]  between documents and we are using this
[01:20:15 -> 01:20:20]  as a signal to the language model that
[01:20:18 -> 01:20:23]  the document has ended and what follows
[01:20:20 -> 01:20:25]  is going to be unrelated to the document
[01:20:23 -> 01:20:27]  previously that said the language model
[01:20:25 -> 01:20:29]  has to learn this from data it it needs
[01:20:27 -> 01:20:31]  to learn that this token usually means
[01:20:29 -> 01:20:34]  that it should wipe its sort of memory
[01:20:31 -> 01:20:35]  of what came before and what came before
[01:20:34 -> 01:20:37]  this token is not actually informative
[01:20:35 -> 01:20:39]  to what comes next but we are expecting
[01:20:37 -> 01:20:40]  the language model to just like learn
[01:20:39 -> 01:20:44]  this but we're giving it the Special
[01:20:40 -> 01:20:46]  sort of the limiter of these documents
[01:20:44 -> 01:20:49]  we can go here to Tech tokenizer and um
[01:20:46 -> 01:20:51]  this the gpt2 tokenizer uh our code that
[01:20:49 -> 01:20:53]  we've been playing with before so we can
[01:20:51 -> 01:20:55]  add here right hello world world how are
[01:20:53 -> 01:20:58]  you and we're getting different tokens
[01:20:55 -> 01:21:02]  but now you can see what if what happens
[01:20:58 -> 01:21:03]  if I put end of text you see how until I
[01:21:02 -> 01:21:06]  finished it these are all different
[01:21:03 -> 01:21:08]  tokens end of
[01:21:06 -> 01:21:13]  text still set different tokens and now
[01:21:08 -> 01:21:15]  when I finish it suddenly we get token
[01:21:13 -> 01:21:18]  50256 and the reason this works is
[01:21:15 -> 01:21:21]  because this didn't actually go through
[01:21:18 -> 01:21:25]  the bpe merges instead the code that
[01:21:21 -> 01:21:28]  actually outposted tokens has special
[01:21:25 -> 01:21:30]  case instructions for handling special
[01:21:28 -> 01:21:32]  tokens um we did not see these special
[01:21:30 -> 01:21:36]  instructions for handling special tokens
[01:21:32 -> 01:21:38]  in the encoder dopy it's absent there
[01:21:36 -> 01:21:40]  but if you go to Tech token Library
[01:21:38 -> 01:21:42]  which is uh implemented in Rust you will
[01:21:40 -> 01:21:44]  find all kinds of special case handling
[01:21:42 -> 01:21:47]  for these special tokens that you can
[01:21:44 -> 01:21:49]  register uh create adds to the
[01:21:47 -> 01:21:50]  vocabulary and then it looks for them
[01:21:49 -> 01:21:53]  and it uh whenever it sees these special
[01:21:50 -> 01:21:56]  tokens like this it will actually come
[01:21:53 -> 01:21:58]  in and swap in that special token so
[01:21:56 -> 01:22:00]  these things are outside of the typical
[01:21:58 -> 01:22:02]  algorithm of uh B PA en
[01:22:00 -> 01:22:05]  coding so these special tokens are used
[01:22:02 -> 01:22:07]  pervasively uh not just in uh basically
[01:22:05 -> 01:22:09]  base language modeling of predicting the
[01:22:07 -> 01:22:10]  next token in the sequence but
[01:22:09 -> 01:22:13]  especially when it gets to later to the
[01:22:10 -> 01:22:15]  fine tuning stage and all of the chat uh
[01:22:13 -> 01:22:16]  gbt sort of aspects of it uh because we
[01:22:15 -> 01:22:18]  don't just want to Del limit documents
[01:22:16 -> 01:22:21]  we want to delimit entire conversations
[01:22:18 -> 01:22:24]  between an assistant and a user so if I
[01:22:21 -> 01:22:26]  refresh this sck tokenizer page the
[01:22:24 -> 01:22:30]  default example that they have here is
[01:22:26 -> 01:22:33]  using not sort of base model encoders
[01:22:30 -> 01:22:35]  but ftuned model uh sort of tokenizers
[01:22:33 -> 01:22:38]  um so for example using the GPT 3.5
[01:22:35 -> 01:22:43]  turbo scheme these here are all special
[01:22:38 -> 01:22:46]  tokens I am start I end Etc uh this is
[01:22:43 -> 01:22:49]  short for Imaginary mcore start by the
[01:22:46 -> 01:22:51]  way but you can see here that there's a
[01:22:49 -> 01:22:52]  sort of start and end of every single
[01:22:51 -> 01:22:56]  message and there can be many other
[01:22:52 -> 01:22:58]  other tokens lots of tokens um in use to
[01:22:56 -> 01:23:00]  delimit these conversations and kind of
[01:22:58 -> 01:23:03]  keep track of the flow of the messages
[01:23:00 -> 01:23:06]  here now we can go back to the Tik token
[01:23:03 -> 01:23:08]  library and here when you scroll to the
[01:23:06 -> 01:23:10]  bottom they talk about how you can
[01:23:08 -> 01:23:13]  extend tick token and I can you can
[01:23:10 -> 01:23:17]  create basically you can Fork uh the um
[01:23:13 -> 01:23:18]  CL 100K base tokenizers in gp4 and for
[01:23:17 -> 01:23:20]  example you can extend it by adding more
[01:23:18 -> 01:23:21]  special tokens and these are totally up
[01:23:20 -> 01:23:23]  to you you can come up with any
[01:23:21 -> 01:23:26]  arbitrary tokens and add them with the
[01:23:23 -> 01:23:29]  new ID afterwards and the tikken library
[01:23:26 -> 01:23:31]  will uh correctly swap them out uh when
[01:23:29 -> 01:23:34]  it sees this in the
[01:23:31 -> 01:23:37]  strings now we can also go back to this
[01:23:34 -> 01:23:39]  file which we've looked at previously
[01:23:37 -> 01:23:41]  and I mentioned that the gpt2 in Tik
[01:23:39 -> 01:23:44]  toen open
[01:23:41 -> 01:23:46]  I.P we have the vocabulary we have the
[01:23:44 -> 01:23:48]  pattern for splitting and then here we
[01:23:46 -> 01:23:50]  are registering the single special token
[01:23:48 -> 01:23:53]  in gpd2 which was the end of text token
[01:23:50 -> 01:23:56]  and we saw that it has this ID
[01:23:53 -> 01:23:57]  in GPT 4 when they defy this here you
[01:23:56 -> 01:23:59]  see that the pattern has changed as
[01:23:57 -> 01:24:01]  we've discussed but also the special
[01:23:59 -> 01:24:03]  tokens have changed in this tokenizer so
[01:24:01 -> 01:24:06]  we of course have the end of text just
[01:24:03 -> 01:24:09]  like in gpd2 but we also see three sorry
[01:24:06 -> 01:24:12]  four additional tokens here Thim prefix
[01:24:09 -> 01:24:14]  middle and suffix what is fim fim is
[01:24:12 -> 01:24:17]  short for fill in the middle and if
[01:24:14 -> 01:24:20]  you'd like to learn more about this idea
[01:24:17 -> 01:24:21]  it comes from this paper um and I'm not
[01:24:20 -> 01:24:23]  going to go into detail in this video
[01:24:21 -> 01:24:27]  it's beyond this video and then there's
[01:24:23 -> 01:24:29]  one additional uh serve token here so
[01:24:27 -> 01:24:31]  that's that encoding as well so it's
[01:24:29 -> 01:24:34]  very common basically to train a
[01:24:31 -> 01:24:37]  language model and then if you'd like uh
[01:24:34 -> 01:24:39]  you can add special tokens now when you
[01:24:37 -> 01:24:41]  add special tokens you of course have to
[01:24:39 -> 01:24:43]  um do some model surgery to the
[01:24:41 -> 01:24:45]  Transformer and all the parameters
[01:24:43 -> 01:24:47]  involved in that Transformer because you
[01:24:45 -> 01:24:48]  are basically adding an integer and you
[01:24:47 -> 01:24:50]  want to make sure that for example your
[01:24:48 -> 01:24:53]  embedding Matrix for the vocabulary
[01:24:50 -> 01:24:54]  tokens has to be extended by adding a
[01:24:53 -> 01:24:56]  row and typically this row would be
[01:24:54 -> 01:24:58]  initialized uh with small random numbers
[01:24:56 -> 01:25:01]  or something like that because we need
[01:24:58 -> 01:25:03]  to have a vector that now stands for
[01:25:01 -> 01:25:04]  that token in addition to that you have
[01:25:03 -> 01:25:05]  to go to the final layer of the
[01:25:04 -> 01:25:07]  Transformer and you have to make sure
[01:25:05 -> 01:25:09]  that that projection at the very end
[01:25:07 -> 01:25:11]  into the classifier uh is extended by
[01:25:09 -> 01:25:13]  one as well so basically there's some
[01:25:11 -> 01:25:16]  model surgery involved that you have to
[01:25:13 -> 01:25:18]  couple with the tokenization changes if
[01:25:16 -> 01:25:20]  you are going to add special tokens but
[01:25:18 -> 01:25:21]  this is a very common operation that
[01:25:20 -> 01:25:23]  people do especially if they'd like to
[01:25:21 -> 01:25:26]  fine tune the model for example taking
[01:25:23 -> 01:25:27]  it from a base model to a chat model
[01:25:26 -> 01:25:29]  like chat
[01:25:27 -> 01:25:31]  GPT okay so at this point you should
[01:25:29 -> 01:25:33]  have everything you need in order to
[01:25:31 -> 01:25:35]  build your own gp4 tokenizer now in the
[01:25:33 -> 01:25:37]  process of developing this lecture I've
[01:25:35 -> 01:25:38]  done that and I published the code under
[01:25:37 -> 01:25:42]  this repository
[01:25:38 -> 01:25:45]  MBP so MBP looks like this right now as
[01:25:42 -> 01:25:46]  I'm recording but uh the MBP repository
[01:25:45 -> 01:25:49]  will probably change quite a bit because
[01:25:46 -> 01:25:51]  I intend to continue working on it um in
[01:25:49 -> 01:25:53]  addition to the MBP repository I've
[01:25:51 -> 01:25:55]  published the this uh exercise
[01:25:53 -> 01:25:58]  progression that you can follow so if
[01:25:55 -> 01:26:01]  you go to exercise. MD here uh this is
[01:25:58 -> 01:26:03]  sort of me breaking up the task ahead of
[01:26:01 -> 01:26:06]  you into four steps that sort of uh
[01:26:03 -> 01:26:08]  build up to what can be a gp4 tokenizer
[01:26:06 -> 01:26:10]  and so feel free to follow these steps
[01:26:08 -> 01:26:12]  exactly and follow a little bit of the
[01:26:10 -> 01:26:14]  guidance that I've laid out here and
[01:26:12 -> 01:26:17]  anytime you feel stuck just reference
[01:26:14 -> 01:26:20]  the MBP repository here so either the
[01:26:17 -> 01:26:22]  tests could be useful or the MBP
[01:26:20 -> 01:26:26]  repository itself I try to keep the code
[01:26:22 -> 01:26:28]  fairly clean and understandable and so
[01:26:26 -> 01:26:30]  um feel free to reference it whenever um
[01:26:28 -> 01:26:32]  you get
[01:26:30 -> 01:26:34]  stuck uh in addition to that basically
[01:26:32 -> 01:26:36]  once you write it you should be able to
[01:26:34 -> 01:26:39]  reproduce this behavior from Tech token
[01:26:36 -> 01:26:41]  so getting the gb4 tokenizer you can
[01:26:39 -> 01:26:43]  take uh you can encode the string and
[01:26:41 -> 01:26:44]  you should get these tokens and then you
[01:26:43 -> 01:26:47]  can encode and decode the exact same
[01:26:44 -> 01:26:48]  string to recover it and in addition to
[01:26:47 -> 01:26:50]  all that you should be able to implement
[01:26:48 -> 01:26:52]  your own train function uh which Tik
[01:26:50 -> 01:26:54]  token Library does not provide it's it's
[01:26:52 -> 01:26:57]  again only inference code but you could
[01:26:54 -> 01:26:59]  write your own train MBP does it as well
[01:26:57 -> 01:27:00]  and that will allow you to train your
[01:26:59 -> 01:27:02]  own token
[01:27:00 -> 01:27:06]  vocabularies so here are some of the
[01:27:02 -> 01:27:08]  code inside M be mean bpe uh shows the
[01:27:06 -> 01:27:12]  token vocabularies that you might obtain
[01:27:08 -> 01:27:15]  so on the left uh here we have the GPT 4
[01:27:12 -> 01:27:17]  merges uh so the first 256 are raw
[01:27:15 -> 01:27:19]  individual bytes and then here I am
[01:27:17 -> 01:27:21]  visualizing the merges that gp4
[01:27:19 -> 01:27:24]  performed during its training so the
[01:27:21 -> 01:27:27]  very first merge that gp4 did was merge
[01:27:24 -> 01:27:30]  two spaces into a single token for you
[01:27:27 -> 01:27:32]  know two spaces and that is a token 256
[01:27:30 -> 01:27:34]  and so this is the order in which things
[01:27:32 -> 01:27:39]  merged during gb4 training and this is
[01:27:34 -> 01:27:41]  the merge order that um we obtain in MBP
[01:27:39 -> 01:27:43]  by training a tokenizer and in this case
[01:27:41 -> 01:27:45]  I trained it on a Wikipedia page of
[01:27:43 -> 01:27:47]  Taylor Swift uh not because I'm a Swifty
[01:27:45 -> 01:27:49]  but because that is one of the longest
[01:27:47 -> 01:27:54]  um Wikipedia Pages apparently that's
[01:27:49 -> 01:27:56]  available but she is pretty cool and
[01:27:54 -> 01:27:59]  um what was I going to say yeah so you
[01:27:56 -> 01:28:04]  can compare these two uh vocabularies
[01:27:59 -> 01:28:06]  and so as an example um here GPT for
[01:28:04 -> 01:28:10]  merged I in to become in and we've done
[01:28:06 -> 01:28:13]  the exact same thing on this token 259
[01:28:10 -> 01:28:14]  here space t becomes space t and that
[01:28:13 -> 01:28:16]  happened for us a little bit later as
[01:28:14 -> 01:28:18]  well so the difference here is again to
[01:28:16 -> 01:28:20]  my understanding only a difference of
[01:28:18 -> 01:28:22]  the training set so as an example
[01:28:20 -> 01:28:23]  because I see a lot of white space I
[01:28:22 -> 01:28:25]  supect that gp4 probably had a lot of
[01:28:23 -> 01:28:27]  python code in its training set I'm not
[01:28:25 -> 01:28:30]  sure uh for the
[01:28:27 -> 01:28:32]  tokenizer and uh here we see much less
[01:28:30 -> 01:28:34]  of that of course in the Wikipedia page
[01:28:32 -> 01:28:35]  so roughly speaking they look the same
[01:28:34 -> 01:28:38]  and they look the same because they're
[01:28:35 -> 01:28:39]  running the same algorithm and when you
[01:28:38 -> 01:28:41]  train your own you're probably going to
[01:28:39 -> 01:28:43]  get something similar depending on what
[01:28:41 -> 01:28:45]  you train it on okay so we are now going
[01:28:43 -> 01:28:47]  to move on from tick token and the way
[01:28:45 -> 01:28:49]  that open AI tokenizes its strings and
[01:28:47 -> 01:28:51]  we're going to discuss one more very
[01:28:49 -> 01:28:52]  commonly used library for working with
[01:28:51 -> 01:28:55]  tokenization inlm
[01:28:52 -> 01:28:58]  and that is sentence piece so sentence
[01:28:55 -> 01:29:00]  piece is very commonly used in language
[01:28:58 -> 01:29:02]  models because unlike Tik token it can
[01:29:00 -> 01:29:04]  do both training and inference and is
[01:29:02 -> 01:29:06]  quite efficient at both it supports a
[01:29:04 -> 01:29:09]  number of algorithms for training uh
[01:29:06 -> 01:29:10]  vocabularies but one of them is the B
[01:29:09 -> 01:29:13]  pair en coding algorithm that we've been
[01:29:10 -> 01:29:15]  looking at so it supports it now
[01:29:13 -> 01:29:18]  sentence piece is used both by llama and
[01:29:15 -> 01:29:20]  mistal series and many other models as
[01:29:18 -> 01:29:22]  well it is on GitHub under Google
[01:29:20 -> 01:29:24]  sentence piece
[01:29:22 -> 01:29:26]  and the big difference with sentence
[01:29:24 -> 01:29:27]  piece and we're going to look at example
[01:29:26 -> 01:29:31]  because this is kind of hard and subtle
[01:29:27 -> 01:29:35]  to explain is that they think different
[01:29:31 -> 01:29:38]  about the order of operations here so in
[01:29:35 -> 01:29:41]  the case of Tik token we first take our
[01:29:38 -> 01:29:42]  code points in the string we encode them
[01:29:41 -> 01:29:44]  using mutf to bytes and then we're
[01:29:42 -> 01:29:48]  merging bytes it's fairly
[01:29:44 -> 01:29:50]  straightforward for sentence piece um it
[01:29:48 -> 01:29:52]  works directly on the level of the code
[01:29:50 -> 01:29:53]  points themselves so so it looks at
[01:29:52 -> 01:29:55]  whatever code points are available in
[01:29:53 -> 01:29:59]  your training set and then it starts
[01:29:55 -> 01:30:01]  merging those code points and um the bpe
[01:29:59 -> 01:30:04]  is running on the level of code
[01:30:01 -> 01:30:06]  points and if you happen to run out of
[01:30:04 -> 01:30:08]  code points so there are maybe some rare
[01:30:06 -> 01:30:09]  uh code points that just don't come up
[01:30:08 -> 01:30:11]  too often and the Rarity is determined
[01:30:09 -> 01:30:14]  by this character coverage hyper
[01:30:11 -> 01:30:16]  parameter then these uh code points will
[01:30:14 -> 01:30:19]  either get mapped to a special unknown
[01:30:16 -> 01:30:22]  token like ank or if you have the bite
[01:30:19 -> 01:30:23]  foldback option turned on then that will
[01:30:22 -> 01:30:26]  take those rare Cod points it will
[01:30:23 -> 01:30:27]  encode them using utf8 and then the
[01:30:26 -> 01:30:30]  individual bytes of that encoding will
[01:30:27 -> 01:30:32]  be translated into tokens and there are
[01:30:30 -> 01:30:35]  these special bite tokens that basically
[01:30:32 -> 01:30:38]  get added to the vocabulary so it uses
[01:30:35 -> 01:30:41]  BP on on the code points and then it
[01:30:38 -> 01:30:44]  falls back to bytes for rare Cod points
[01:30:41 -> 01:30:45]  um and so that's kind of like difference
[01:30:44 -> 01:30:47]  personally I find the Tik token we
[01:30:45 -> 01:30:48]  significantly cleaner uh but it's kind
[01:30:47 -> 01:30:50]  of like a subtle but pretty major
[01:30:48 -> 01:30:52]  difference between the way they approach
[01:30:50 -> 01:30:54]  tokenization let's work with with a
[01:30:52 -> 01:30:56]  concrete example because otherwise this
[01:30:54 -> 01:30:59]  is kind of hard to um to get your head
[01:30:56 -> 01:31:01]  around so let's work with a concrete
[01:30:59 -> 01:31:03]  example this is how we can import
[01:31:01 -> 01:31:05]  sentence piece and then here we're going
[01:31:03 -> 01:31:06]  to take I think I took like the
[01:31:05 -> 01:31:08]  description of sentence piece and I just
[01:31:06 -> 01:31:10]  created like a little toy data set it
[01:31:08 -> 01:31:13]  really likes to have a file so I created
[01:31:10 -> 01:31:15]  a toy. txt file with this
[01:31:13 -> 01:31:16]  content now what's kind of a little bit
[01:31:15 -> 01:31:18]  crazy about sentence piece is that
[01:31:16 -> 01:31:20]  there's a ton of options and
[01:31:18 -> 01:31:22]  configurations and the reason this is so
[01:31:20 -> 01:31:23]  is because sentence piece has been
[01:31:22 -> 01:31:25]  around I think for a while and it really
[01:31:23 -> 01:31:28]  tries to handle a large diversity of
[01:31:25 -> 01:31:30]  things and um because it's been around I
[01:31:28 -> 01:31:33]  think it has quite a bit of accumulated
[01:31:30 -> 01:31:35]  historical baggage uh as well and so in
[01:31:33 -> 01:31:36]  particular there's like a ton of
[01:31:35 -> 01:31:39]  configuration arguments this is not even
[01:31:36 -> 01:31:40]  all of it you can go to here to see all
[01:31:39 -> 01:31:44]  the training
[01:31:40 -> 01:31:45]  options um and uh there's also quite
[01:31:44 -> 01:31:48]  useful documentation when you look at
[01:31:45 -> 01:31:52]  the raw Proto buff uh that is used to
[01:31:48 -> 01:31:54]  represent the trainer spec and so on um
[01:31:52 -> 01:31:56]  many of these options are irrelevant to
[01:31:54 -> 01:31:59]  us so maybe to point out one example Das
[01:31:56 -> 01:32:01]  Das shrinking Factor uh this shrinking
[01:31:59 -> 01:32:03]  factor is not used in the B pair en
[01:32:01 -> 01:32:05]  coding algorithm so this is just an
[01:32:03 -> 01:32:09]  argument that is irrelevant to us um it
[01:32:05 -> 01:32:09]  applies to a different training
[01:32:09 -> 01:32:13]  algorithm now what I tried to do here is
[01:32:11 -> 01:32:15]  I tried to set up sentence piece in a
[01:32:13 -> 01:32:18]  way that is very very similar as far as
[01:32:15 -> 01:32:22]  I can tell to maybe identical hopefully
[01:32:18 -> 01:32:25]  to the way that llama 2 was strained so
[01:32:22 -> 01:32:27]  the way they trained their own um their
[01:32:25 -> 01:32:28]  own tokenizer and the way I did this was
[01:32:27 -> 01:32:31]  basically you can take the tokenizer
[01:32:28 -> 01:32:35]  model file that meta released and you
[01:32:31 -> 01:32:38]  can um open it using the Proto protuff
[01:32:35 -> 01:32:39]  uh sort of file that you can generate
[01:32:38 -> 01:32:41]  and then you can inspect all the options
[01:32:39 -> 01:32:43]  and I tried to copy over all the options
[01:32:41 -> 01:32:46]  that looked relevant so here we set up
[01:32:43 -> 01:32:48]  the input it's raw text in this file
[01:32:46 -> 01:32:50]  here's going to be the output so it's
[01:32:48 -> 01:32:52]  going to be for talk 400. model and
[01:32:50 -> 01:32:53]  vocab
[01:32:52 -> 01:32:56]  we're saying that we're going to use the
[01:32:53 -> 01:32:58]  BP algorithm and we want to Bap size of
[01:32:56 -> 01:33:00]  400 then there's a ton of configurations
[01:32:58 -> 01:33:00]  here
[01:33:01 -> 01:33:07]  for um for basically pre-processing and
[01:33:05 -> 01:33:09]  normalization rules as they're called
[01:33:07 -> 01:33:11]  normalization used to be very prevalent
[01:33:09 -> 01:33:12]  I would say before llms in natural
[01:33:11 -> 01:33:14]  language processing so in machine
[01:33:12 -> 01:33:16]  translation and uh text classification
[01:33:14 -> 01:33:18]  and so on you want to normalize and
[01:33:16 -> 01:33:19]  simplify the text and you want to turn
[01:33:18 -> 01:33:22]  it all lowercase and you want to remove
[01:33:19 -> 01:33:23]  all double whites space Etc
[01:33:22 -> 01:33:25]  and in language models we prefer not to
[01:33:23 -> 01:33:26]  do any of it or at least that is my
[01:33:25 -> 01:33:28]  preference as a deep learning person you
[01:33:26 -> 01:33:31]  want to not touch your data you want to
[01:33:28 -> 01:33:33]  keep the raw data as much as possible um
[01:33:31 -> 01:33:35]  in a raw
[01:33:33 -> 01:33:38]  form so you're basically trying to turn
[01:33:35 -> 01:33:39]  off a lot of this if you can the other
[01:33:38 -> 01:33:43]  thing that sentence piece does is that
[01:33:39 -> 01:33:45]  it has this concept of sentences so
[01:33:43 -> 01:33:46]  sentence piece it's back it's kind of
[01:33:45 -> 01:33:50]  like was developed I think early in the
[01:33:46 -> 01:33:51]  days where there was um an idea that
[01:33:50 -> 01:33:54]  they you're training a tokenizer on a
[01:33:51 -> 01:33:56]  bunch of independent sentences so it has
[01:33:54 -> 01:33:58]  a lot of like how many sentences you're
[01:33:56 -> 01:34:00]  going to train on what is the maximum
[01:33:58 -> 01:34:03]  sentence length
[01:34:00 -> 01:34:04]  um shuffling sentences and so for it
[01:34:03 -> 01:34:06]  sentences are kind of like the
[01:34:04 -> 01:34:08]  individual training examples but again
[01:34:06 -> 01:34:10]  in the context of llms I find that this
[01:34:08 -> 01:34:13]  is like a very spous and weird
[01:34:10 -> 01:34:15]  distinction like sentences are just like
[01:34:13 -> 01:34:18]  don't touch the raw data sentences
[01:34:15 -> 01:34:20]  happen to exist but in raw data sets
[01:34:18 -> 01:34:22]  there are a lot of like inet like what
[01:34:20 -> 01:34:25]  exactly is a sentence what isn't a
[01:34:22 -> 01:34:26]  sentence um and so I think like it's
[01:34:25 -> 01:34:28]  really hard to Define what an actual
[01:34:26 -> 01:34:30]  sentence is if you really like dig into
[01:34:28 -> 01:34:32]  it and there could be different concepts
[01:34:30 -> 01:34:33]  of it in different languages or
[01:34:32 -> 01:34:35]  something like that so why even
[01:34:33 -> 01:34:36]  introduce the concept it it doesn't
[01:34:35 -> 01:34:39]  honestly make sense to me I would just
[01:34:36 -> 01:34:40]  prefer to treat a file as a giant uh
[01:34:39 -> 01:34:42]  stream of
[01:34:40 -> 01:34:45]  bytes it has a lot of treatment around
[01:34:42 -> 01:34:46]  rare word characters and when I say word
[01:34:45 -> 01:34:48]  I mean code points we're going to come
[01:34:46 -> 01:34:51]  back to this in a second and it has a
[01:34:48 -> 01:34:54]  lot of other rules for um basically
[01:34:51 -> 01:34:56]  splitting digits splitting white space
[01:34:54 -> 01:34:58]  and numbers and how you deal with that
[01:34:56 -> 01:35:00]  so these are some kind of like merge
[01:34:58 -> 01:35:02]  rules so I think this is a little bit
[01:35:00 -> 01:35:04]  equivalent to tick token using the
[01:35:02 -> 01:35:07]  regular expression to split up
[01:35:04 -> 01:35:09]  categories there's like kind of
[01:35:07 -> 01:35:10]  equivalence of it if you squint T it in
[01:35:09 -> 01:35:14]  sentence piece where you can also for
[01:35:10 -> 01:35:15]  example split up split up the digits uh
[01:35:14 -> 01:35:18]  and uh so
[01:35:15 -> 01:35:19]  on there's a few more things here that
[01:35:18 -> 01:35:20]  I'll come back to in a bit and then
[01:35:19 -> 01:35:23]  there are some special tokens that you
[01:35:20 -> 01:35:25]  can indicate and it hardcodes the UN
[01:35:23 -> 01:35:29]  token the beginning of sentence end of
[01:35:25 -> 01:35:32]  sentence and a pad token um and the UN
[01:35:29 -> 01:35:34]  token must exist for my understanding
[01:35:32 -> 01:35:37]  and then some some things so we can
[01:35:34 -> 01:35:40]  train and when when I press train it's
[01:35:37 -> 01:35:43]  going to create this file talk 400.
[01:35:40 -> 01:35:45]  model and talk 400. wab I can then load
[01:35:43 -> 01:35:48]  the model file and I can inspect the
[01:35:45 -> 01:35:53]  vocabulary off it and so we trained
[01:35:48 -> 01:35:55]  vocab size 400 on this text here and
[01:35:53 -> 01:35:56]  these are the individual pieces the
[01:35:55 -> 01:35:58]  individual tokens that sentence piece
[01:35:56 -> 01:36:02]  will create so in the beginning we see
[01:35:58 -> 01:36:04]  that we have the an token uh with the ID
[01:36:02 -> 01:36:07]  zero then we have the beginning of
[01:36:04 -> 01:36:09]  sequence end of sequence one and two and
[01:36:07 -> 01:36:12]  then we said that the pad ID is negative
[01:36:09 -> 01:36:13]  1 so we chose not to use it so there's
[01:36:12 -> 01:36:16]  no pad ID
[01:36:13 -> 01:36:20]  here then these are individual bite
[01:36:16 -> 01:36:23]  tokens so here we saw that bite fallback
[01:36:20 -> 01:36:26]  in llama was turned on so it's true so
[01:36:23 -> 01:36:27]  what follows are going to be the 256
[01:36:26 -> 01:36:30]  bite
[01:36:27 -> 01:36:30]  tokens and these are their
[01:36:31 -> 01:36:37]  IDs and then at the bottom after the
[01:36:35 -> 01:36:40]  bite tokens come the
[01:36:37 -> 01:36:42]  merges and these are the parent nodes in
[01:36:40 -> 01:36:43]  the merges so we're not seeing the
[01:36:42 -> 01:36:44]  children we're just seeing the parents
[01:36:43 -> 01:36:47]  and their
[01:36:44 -> 01:36:50]  ID and then after the
[01:36:47 -> 01:36:53]  merges comes eventually the individual
[01:36:50 -> 01:36:55]  tokens and their IDs and so these are
[01:36:53 -> 01:36:58]  the individual tokens so these are the
[01:36:55 -> 01:37:00]  individual code Point tokens if you will
[01:36:58 -> 01:37:01]  and they come at the end so that is the
[01:37:00 -> 01:37:03]  ordering with which sentence piece sort
[01:37:01 -> 01:37:06]  of like represents its vocabularies it
[01:37:03 -> 01:37:08]  starts with special tokens then the bike
[01:37:06 -> 01:37:11]  tokens then the merge tokens and then
[01:37:08 -> 01:37:14]  the individual codo tokens and all these
[01:37:11 -> 01:37:16]  raw codepoint to tokens are the ones
[01:37:14 -> 01:37:19]  that it encountered in the training
[01:37:16 -> 01:37:22]  set so those individual code points are
[01:37:19 -> 01:37:24]  all the the entire set of code points
[01:37:22 -> 01:37:27]  that occurred
[01:37:24 -> 01:37:29]  here so those all get put in there and
[01:37:27 -> 01:37:31]  then those that are extremely rare as
[01:37:29 -> 01:37:32]  determined by character coverage so if a
[01:37:31 -> 01:37:35]  code Point occurred only a single time
[01:37:32 -> 01:37:37]  out of like a million um sentences or
[01:37:35 -> 01:37:40]  something like that then it would be
[01:37:37 -> 01:37:41]  ignored and it would not be added to our
[01:37:40 -> 01:37:43]  uh
[01:37:41 -> 01:37:46]  vocabulary once we have a vocabulary we
[01:37:43 -> 01:37:47]  can encode into IDs and we can um sort
[01:37:46 -> 01:37:50]  of get a
[01:37:47 -> 01:37:54]  list and then here I am also decoding
[01:37:50 -> 01:37:56]  the indiv idual tokens back into little
[01:37:54 -> 01:38:01]  pieces as they call it so let's take a
[01:37:56 -> 01:38:04]  look at what happened here hello space
[01:38:01 -> 01:38:07]  on so these are the token IDs we got
[01:38:04 -> 01:38:11]  back and when we look here uh a few
[01:38:07 -> 01:38:14]  things sort of uh jump to mind number
[01:38:11 -> 01:38:15]  one take a look at these characters the
[01:38:14 -> 01:38:18]  Korean characters of course were not
[01:38:15 -> 01:38:19]  part of the training set so sentence
[01:38:18 -> 01:38:22]  piece is encountering code points that
[01:38:19 -> 01:38:24]  it has not seen during training time and
[01:38:22 -> 01:38:26]  those code points do not have a token
[01:38:24 -> 01:38:30]  associated with them so suddenly these
[01:38:26 -> 01:38:33]  are un tokens unknown tokens but because
[01:38:30 -> 01:38:36]  bite fall back as true instead sentence
[01:38:33 -> 01:38:39]  piece falls back to bytes and so it
[01:38:36 -> 01:38:43]  takes this it encodes it with utf8 and
[01:38:39 -> 01:38:45]  then it uses these tokens to represent
[01:38:43 -> 01:38:49]  uh those bytes and that's what we are
[01:38:45 -> 01:38:52]  getting sort of here this is the utf8 uh
[01:38:49 -> 01:38:56]  encoding and in this shifted by three uh
[01:38:52 -> 01:38:58]  because of these um special tokens here
[01:38:56 -> 01:39:02]  that have IDs earlier on so that's what
[01:38:58 -> 01:39:05]  happened here now one more thing that um
[01:39:02 -> 01:39:08]  well first before I go on with respect
[01:39:05 -> 01:39:10]  to the bitef back let me remove bite
[01:39:08 -> 01:39:12]  foldback if this is false what's going
[01:39:10 -> 01:39:14]  to happen let's
[01:39:12 -> 01:39:17]  retrain so the first thing that happened
[01:39:14 -> 01:39:19]  is all the bite tokens disappeared right
[01:39:17 -> 01:39:20]  and now we just have the merges and we
[01:39:19 -> 01:39:21]  have a lot more merges now because we
[01:39:20 -> 01:39:25]  have a lot more space because we're not
[01:39:21 -> 01:39:25]  taking up space in the wab size uh with
[01:39:25 -> 01:39:29]  all the
[01:39:25 -> 01:39:33]  bytes and now if we encode
[01:39:29 -> 01:39:35]  this we get a zero so this entire string
[01:39:33 -> 01:39:39]  here suddenly there's no bitef back so
[01:39:35 -> 01:39:42]  this is unknown and unknown is an and so
[01:39:39 -> 01:39:44]  this is zero because the an token is
[01:39:42 -> 01:39:46]  token zero and you have to keep in mind
[01:39:44 -> 01:39:48]  that this would feed into your uh
[01:39:46 -> 01:39:49]  language model so what is a language
[01:39:48 -> 01:39:52]  model supposed to do when all kinds of
[01:39:49 -> 01:39:54]  different things that are unrecognized
[01:39:52 -> 01:39:56]  because they're rare just end up mapping
[01:39:54 -> 01:39:57]  into Unk it's not exactly the property
[01:39:56 -> 01:40:02]  that you want so that's why I think
[01:39:57 -> 01:40:03]  llama correctly uh used by fallback true
[01:40:02 -> 01:40:06]  uh because we definitely want to feed
[01:40:03 -> 01:40:08]  these um unknown or rare code points
[01:40:06 -> 01:40:10]  into the model and some uh some manner
[01:40:08 -> 01:40:12]  the next thing I want to show you is the
[01:40:10 -> 01:40:14]  following notice here when we are
[01:40:12 -> 01:40:18]  decoding all the individual tokens you
[01:40:14 -> 01:40:21]  see how spaces uh space here ends up
[01:40:18 -> 01:40:23]  being this um bold underline I'm not
[01:40:21 -> 01:40:25]  100% sure by the way why sentence piece
[01:40:23 -> 01:40:27]  switches whites space into these bold
[01:40:25 -> 01:40:29]  underscore characters maybe it's for
[01:40:27 -> 01:40:32]  visualization I'm not 100% sure why that
[01:40:29 -> 01:40:37]  happens uh but notice this why do we
[01:40:32 -> 01:40:40]  have an extra space in the front of
[01:40:37 -> 01:40:43]  hello um what where is this coming from
[01:40:40 -> 01:40:45]  well it's coming from this option
[01:40:43 -> 01:40:48]  here
[01:40:45 -> 01:40:49]  um add dummy prefix is true and when you
[01:40:48 -> 01:40:51]  go to the
[01:40:49 -> 01:40:53]  documentation add D whites space at the
[01:40:51 -> 01:40:55]  beginning of text in order to treat
[01:40:53 -> 01:40:57]  World in world and hello world in the
[01:40:55 -> 01:40:59]  exact same way so what this is trying to
[01:40:57 -> 01:41:02]  do is the
[01:40:59 -> 01:41:06]  following if we go back to our tick
[01:41:02 -> 01:41:10]  tokenizer world as uh token by itself
[01:41:06 -> 01:41:14]  has a different ID than space world so
[01:41:10 -> 01:41:16]  we have this is 1917 but this is 14 Etc
[01:41:14 -> 01:41:17]  so these are two different tokens for
[01:41:16 -> 01:41:18]  the language model and the language
[01:41:17 -> 01:41:20]  model has to learn from data that they
[01:41:18 -> 01:41:23]  are actually kind of like a very similar
[01:41:20 -> 01:41:26]  concept so to the language model in the
[01:41:23 -> 01:41:27]  Tik token World um basically words in
[01:41:26 -> 01:41:29]  the beginning of sentences and words in
[01:41:27 -> 01:41:32]  the middle of sentences actually look
[01:41:29 -> 01:41:34]  completely different um and it has to
[01:41:32 -> 01:41:36]  learned that they are roughly the same
[01:41:34 -> 01:41:38]  so this add dami prefix is trying to
[01:41:36 -> 01:41:41]  fight that a little bit and the way that
[01:41:38 -> 01:41:46]  works is that it basically
[01:41:41 -> 01:41:49]  uh adds a dummy prefix so for as a as a
[01:41:46 -> 01:41:51]  part of pre-processing it will take the
[01:41:49 -> 01:41:54]  string and it will add a space it will
[01:41:51 -> 01:41:57]  do this and that's done in an effort to
[01:41:54 -> 01:42:00]  make this world and that world the same
[01:41:57 -> 01:42:02]  they will both be space world so that's
[01:42:00 -> 01:42:05]  one other kind of pre-processing option
[01:42:02 -> 01:42:07]  that is turned on and llama 2 also uh
[01:42:05 -> 01:42:08]  uses this option and that's I think
[01:42:07 -> 01:42:10]  everything that I want to say for my
[01:42:08 -> 01:42:13]  preview of sentence piece and how it is
[01:42:10 -> 01:42:16]  different um maybe here what I've done
[01:42:13 -> 01:42:19]  is I just uh put in the Raw protocol
[01:42:16 -> 01:42:22]  buffer representation basically of the
[01:42:19 -> 01:42:24]  tokenizer the too trained so feel free
[01:42:22 -> 01:42:27]  to sort of Step through this and if you
[01:42:24 -> 01:42:30]  would like uh your tokenization to look
[01:42:27 -> 01:42:31]  identical to that of the meta uh llama 2
[01:42:30 -> 01:42:34]  then you would be copy pasting these
[01:42:31 -> 01:42:36]  settings as I tried to do up above and
[01:42:34 -> 01:42:38]  uh yeah that's I think that's it for
[01:42:36 -> 01:42:40]  this section I think my summary for
[01:42:38 -> 01:42:42]  sentence piece from all of this is
[01:42:40 -> 01:42:44]  number one I think that there's a lot of
[01:42:42 -> 01:42:45]  historical baggage in sentence piece a
[01:42:44 -> 01:42:47]  lot of Concepts that I think are
[01:42:45 -> 01:42:49]  slightly confusing and I think
[01:42:47 -> 01:42:50]  potentially um contain foot guns like
[01:42:49 -> 01:42:53]  this concept of a sentence and it's
[01:42:50 -> 01:42:55]  maximum length and stuff like that um
[01:42:53 -> 01:42:58]  otherwise it is fairly commonly used in
[01:42:55 -> 01:43:01]  the industry um because it is efficient
[01:42:58 -> 01:43:02]  and can do both training and inference
[01:43:01 -> 01:43:05]  uh it has a few quirks like for example
[01:43:02 -> 01:43:06]  un token must exist and the way the bite
[01:43:05 -> 01:43:08]  fallbacks are done and so on I don't
[01:43:06 -> 01:43:09]  find particularly elegant and
[01:43:08 -> 01:43:11]  unfortunately I have to say it's not
[01:43:09 -> 01:43:14]  very well documented so it took me a lot
[01:43:11 -> 01:43:16]  of time working with this myself um and
[01:43:14 -> 01:43:17]  just visualizing things and trying to
[01:43:16 -> 01:43:19]  really understand what is happening here
[01:43:17 -> 01:43:21]  because uh the documentation
[01:43:19 -> 01:43:24]  unfortunately is in my opion not not
[01:43:21 -> 01:43:26]  super amazing but it is a very nice repo
[01:43:24 -> 01:43:28]  that is available to you if you'd like
[01:43:26 -> 01:43:29]  to train your own tokenizer right now
[01:43:28 -> 01:43:31]  okay let me now switch gears again as
[01:43:29 -> 01:43:33]  we're starting to slowly wrap up here I
[01:43:31 -> 01:43:35]  want to revisit this issue in a bit more
[01:43:33 -> 01:43:36]  detail of how we should set the vocap
[01:43:35 -> 01:43:39]  size and what are some of the
[01:43:36 -> 01:43:40]  considerations around it so for this I'd
[01:43:39 -> 01:43:42]  like to go back to the model
[01:43:40 -> 01:43:44]  architecture that we developed in the
[01:43:42 -> 01:43:47]  last video when we built the GPT from
[01:43:44 -> 01:43:49]  scratch so this here was uh the file
[01:43:47 -> 01:43:51]  that we built in the previous video and
[01:43:49 -> 01:43:52]  we defined the Transformer model and and
[01:43:51 -> 01:43:55]  let's specifically look at Bap size and
[01:43:52 -> 01:43:58]  where it appears in this file so here we
[01:43:55 -> 01:43:59]  Define the voap size uh at this time it
[01:43:58 -> 01:44:02]  was 65 or something like that extremely
[01:43:59 -> 01:44:04]  small number so this will grow much
[01:44:02 -> 01:44:06]  larger you'll see that Bap size doesn't
[01:44:04 -> 01:44:08]  come up too much in most of these layers
[01:44:06 -> 01:44:11]  the only place that it comes up to is in
[01:44:08 -> 01:44:13]  exactly these two places here so when we
[01:44:11 -> 01:44:15]  Define the language model there's the
[01:44:13 -> 01:44:18]  token embedding table which is this
[01:44:15 -> 01:44:21]  two-dimensional array where the vocap
[01:44:18 -> 01:44:23]  size is basically the number of rows and
[01:44:21 -> 01:44:25]  uh each vocabulary element each token
[01:44:23 -> 01:44:27]  has a vector that we're going to train
[01:44:25 -> 01:44:29]  using back propagation that Vector is of
[01:44:27 -> 01:44:31]  size and embed which is number of
[01:44:29 -> 01:44:33]  channels in the Transformer and
[01:44:31 -> 01:44:35]  basically as voap size increases this
[01:44:33 -> 01:44:37]  embedding table as I mentioned earlier
[01:44:35 -> 01:44:39]  is going to also grow we're going to be
[01:44:37 -> 01:44:41]  adding rows in addition to that at the
[01:44:39 -> 01:44:44]  end of the Transformer there's this LM
[01:44:41 -> 01:44:46]  head layer which is a linear layer and
[01:44:44 -> 01:44:48]  you'll notice that that layer is used at
[01:44:46 -> 01:44:49]  the very end to produce the logits uh
[01:44:48 -> 01:44:51]  which become the probabilities for the
[01:44:49 -> 01:44:53]  next token in sequence and so
[01:44:51 -> 01:44:56]  intuitively we're trying to produce a
[01:44:53 -> 01:44:58]  probability for every single token that
[01:44:56 -> 01:45:01]  might come next at every point in time
[01:44:58 -> 01:45:02]  of that Transformer and if we have more
[01:45:01 -> 01:45:04]  and more tokens we need to produce more
[01:45:02 -> 01:45:06]  and more probabilities so every single
[01:45:04 -> 01:45:08]  token is going to introduce an
[01:45:06 -> 01:45:10]  additional dot product that we have to
[01:45:08 -> 01:45:11]  do here in this linear layer for this
[01:45:10 -> 01:45:14]  final layer in a
[01:45:11 -> 01:45:16]  Transformer so why can't vocap size be
[01:45:14 -> 01:45:18]  infinite why can't we grow to Infinity
[01:45:16 -> 01:45:21]  well number one your token embedding
[01:45:18 -> 01:45:23]  table is going to grow uh your linear
[01:45:21 -> 01:45:25]  layer is going to grow so we're going to
[01:45:23 -> 01:45:26]  be doing a lot more computation here
[01:45:25 -> 01:45:29]  because this LM head layer will become
[01:45:26 -> 01:45:30]  more computational expensive number two
[01:45:29 -> 01:45:33]  because we have more parameters we could
[01:45:30 -> 01:45:35]  be worried that we are going to be under
[01:45:33 -> 01:45:37]  trining some of these
[01:45:35 -> 01:45:38]  parameters so intuitively if you have a
[01:45:37 -> 01:45:41]  very large vocabulary size say we have a
[01:45:38 -> 01:45:42]  million uh tokens then every one of
[01:45:41 -> 01:45:45]  these tokens is going to come up more
[01:45:42 -> 01:45:46]  and more rarely in the training data
[01:45:45 -> 01:45:48]  because there's a lot more other tokens
[01:45:46 -> 01:45:51]  all over the place and so we're going to
[01:45:48 -> 01:45:53]  be seeing fewer and fewer examples uh
[01:45:51 -> 01:45:55]  for each individual token and you might
[01:45:53 -> 01:45:56]  be worried that basically the vectors
[01:45:55 -> 01:45:58]  associated with every token will be
[01:45:56 -> 01:45:59]  undertrained as a result because they
[01:45:58 -> 01:46:00]  just don't come up too often and they
[01:45:59 -> 01:46:03]  don't participate in the forward
[01:46:00 -> 01:46:04]  backward pass in addition to that as
[01:46:03 -> 01:46:07]  your vocab size grows you're going to
[01:46:04 -> 01:46:09]  start shrinking your sequences a lot
[01:46:07 -> 01:46:10]  right and that's really nice because
[01:46:09 -> 01:46:12]  that means that we're going to be
[01:46:10 -> 01:46:13]  attending to more and more text so
[01:46:12 -> 01:46:15]  that's nice but also you might be
[01:46:13 -> 01:46:18]  worrying that two large of chunks are
[01:46:15 -> 01:46:20]  being squished into single tokens and so
[01:46:18 -> 01:46:25]  the model just doesn't have as much of
[01:46:20 -> 01:46:26]  time to think per sort of um some number
[01:46:25 -> 01:46:28]  of characters in the text or you can
[01:46:26 -> 01:46:29]  think about it that way right so
[01:46:28 -> 01:46:31]  basically we're squishing too much
[01:46:29 -> 01:46:33]  information into a single token and then
[01:46:31 -> 01:46:34]  the forward pass of the Transformer is
[01:46:33 -> 01:46:36]  not enough to actually process that
[01:46:34 -> 01:46:37]  information appropriately and so these
[01:46:36 -> 01:46:38]  are some of the considerations you're
[01:46:37 -> 01:46:40]  thinking about when you're designing the
[01:46:38 -> 01:46:42]  vocab size as I mentioned this is mostly
[01:46:40 -> 01:46:44]  an empirical hyperparameter and it seems
[01:46:42 -> 01:46:46]  like in state-of-the-art architectures
[01:46:44 -> 01:46:49]  today this is usually in the high 10,000
[01:46:46 -> 01:46:50]  or somewhere around 100,000 today and
[01:46:49 -> 01:46:53]  the next consideration I want to briefly
[01:46:50 -> 01:46:55]  talk about is what if we want to take a
[01:46:53 -> 01:46:57]  pre-trained model and we want to extend
[01:46:55 -> 01:46:58]  the vocap size and this is done fairly
[01:46:57 -> 01:47:02]  commonly actually so for example when
[01:46:58 -> 01:47:03]  you're doing fine-tuning for cha GPT um
[01:47:02 -> 01:47:05]  a lot more new special tokens get
[01:47:03 -> 01:47:08]  introduced on top of the base model to
[01:47:05 -> 01:47:09]  maintain the metadata and all the
[01:47:08 -> 01:47:11]  structure of conversation objects
[01:47:09 -> 01:47:14]  between a user and an assistant so that
[01:47:11 -> 01:47:15]  takes a lot of special tokens you might
[01:47:14 -> 01:47:17]  also try to throw in more special tokens
[01:47:15 -> 01:47:20]  for example for using the browser or any
[01:47:17 -> 01:47:22]  other tool and so it's very tempting to
[01:47:20 -> 01:47:24]  add a lot of tokens for all kinds of
[01:47:22 -> 01:47:25]  special functionality so if you want to
[01:47:24 -> 01:47:27]  be adding a token that's totally
[01:47:25 -> 01:47:29]  possible Right all we have to do is we
[01:47:27 -> 01:47:32]  have to resize this embedding so we have
[01:47:29 -> 01:47:34]  to add rows we would initialize these uh
[01:47:32 -> 01:47:36]  parameters from scratch to be small
[01:47:34 -> 01:47:39]  random numbers and then we have to
[01:47:36 -> 01:47:41]  extend the weight inside this linear uh
[01:47:39 -> 01:47:43]  so we have to start making dot products
[01:47:41 -> 01:47:44]  um with the associated parameters as
[01:47:43 -> 01:47:46]  well to basically calculate the
[01:47:44 -> 01:47:48]  probabilities for these new tokens so
[01:47:46 -> 01:47:50]  both of these are just a resizing
[01:47:48 -> 01:47:52]  operation it's a very mild
[01:47:50 -> 01:47:54]  model surgery and can be done fairly
[01:47:52 -> 01:47:55]  easily and it's quite common that
[01:47:54 -> 01:47:57]  basically you would freeze the base
[01:47:55 -> 01:47:58]  model you introduce these new parameters
[01:47:57 -> 01:48:00]  and then you only train these new
[01:47:58 -> 01:48:03]  parameters to introduce new tokens into
[01:48:00 -> 01:48:04]  the architecture um and so you can
[01:48:03 -> 01:48:06]  freeze arbitrary parts of it or you can
[01:48:04 -> 01:48:08]  train arbitrary parts of it and that's
[01:48:06 -> 01:48:10]  totally up to you but basically minor
[01:48:08 -> 01:48:11]  surgery required if you'd like to
[01:48:10 -> 01:48:13]  introduce new tokens and finally I'd
[01:48:11 -> 01:48:15]  like to mention that actually there's an
[01:48:13 -> 01:48:17]  entire design space of applications in
[01:48:15 -> 01:48:19]  terms of introducing new tokens into a
[01:48:17 -> 01:48:21]  vocabulary that go Way Beyond just
[01:48:19 -> 01:48:23]  adding special tokens and special new
[01:48:21 -> 01:48:24]  functionality so just to give you a
[01:48:23 -> 01:48:26]  sense of the design space but this could
[01:48:24 -> 01:48:28]  be an entire video just by itself uh
[01:48:26 -> 01:48:31]  this is a paper on learning to compress
[01:48:28 -> 01:48:33]  prompts with what they called uh gist
[01:48:31 -> 01:48:34]  tokens and the rough idea is suppose
[01:48:33 -> 01:48:37]  that you're using language models in a
[01:48:34 -> 01:48:38]  setting that requires very long prompts
[01:48:37 -> 01:48:39]  while these long prompts just slow
[01:48:38 -> 01:48:41]  everything down because you have to
[01:48:39 -> 01:48:43]  encode them and then you have to use
[01:48:41 -> 01:48:45]  them and then you're tending over them
[01:48:43 -> 01:48:47]  and it's just um you know heavy to have
[01:48:45 -> 01:48:50]  very large prompts so instead what they
[01:48:47 -> 01:48:54]  do here in this paper is they introduce
[01:48:50 -> 01:48:56]  new tokens and um imagine basically
[01:48:54 -> 01:48:59]  having a few new tokens you put them in
[01:48:56 -> 01:49:01]  a sequence and then you train the model
[01:48:59 -> 01:49:03]  by distillation so you are keeping the
[01:49:01 -> 01:49:05]  entire model Frozen and you're only
[01:49:03 -> 01:49:06]  training the representations of the new
[01:49:05 -> 01:49:09]  tokens their embeddings and you're
[01:49:06 -> 01:49:11]  optimizing over the new tokens such that
[01:49:09 -> 01:49:15]  the behavior of the language model is
[01:49:11 -> 01:49:17]  identical uh to the model that has a
[01:49:15 -> 01:49:19]  very long prompt that works for you and
[01:49:17 -> 01:49:20]  so it's a compression technique of
[01:49:19 -> 01:49:23]  compressing that very long prompt into
[01:49:20 -> 01:49:25]  those few new gist tokens and so you can
[01:49:23 -> 01:49:26]  train this and then at test time you can
[01:49:25 -> 01:49:28]  discard your old prompt and just swap in
[01:49:26 -> 01:49:31]  those tokens and they sort of like uh
[01:49:28 -> 01:49:33]  stand in for that very long prompt and
[01:49:31 -> 01:49:36]  have an almost identical performance and
[01:49:33 -> 01:49:38]  so this is one um technique and a class
[01:49:36 -> 01:49:39]  of parameter efficient fine-tuning
[01:49:38 -> 01:49:41]  techniques where most of the model is
[01:49:39 -> 01:49:43]  basically fixed and there's no training
[01:49:41 -> 01:49:45]  of the model weights there's no training
[01:49:43 -> 01:49:47]  of Laura or anything like that of new
[01:49:45 -> 01:49:49]  parameters the the parameters that
[01:49:47 -> 01:49:51]  you're training are now just the uh
[01:49:49 -> 01:49:52]  token embeddings so that's just one
[01:49:51 -> 01:49:54]  example but this could again be like an
[01:49:52 -> 01:49:55]  entire video but just to give you a
[01:49:54 -> 01:49:57]  sense that there's a whole design space
[01:49:55 -> 01:49:59]  here that is potentially worth exploring
[01:49:57 -> 01:50:01]  in the future the next thing I want to
[01:49:59 -> 01:50:03]  briefly address is that I think recently
[01:50:01 -> 01:50:05]  there's a lot of momentum in how you
[01:50:03 -> 01:50:06]  actually could construct Transformers
[01:50:05 -> 01:50:08]  that can simultaneously process not just
[01:50:06 -> 01:50:11]  text as the input modality but a lot of
[01:50:08 -> 01:50:14]  other modalities so be it images videos
[01:50:11 -> 01:50:16]  audio Etc and how do you feed in all
[01:50:14 -> 01:50:18]  these modalities and potentially predict
[01:50:16 -> 01:50:19]  these modalities from a Transformer uh
[01:50:18 -> 01:50:21]  do you have to change the architecture
[01:50:19 -> 01:50:23]  in some fundamental way and I think what
[01:50:21 -> 01:50:24]  a lot of people are starting to converge
[01:50:23 -> 01:50:25]  towards is that you're not changing the
[01:50:24 -> 01:50:27]  architecture you stick with the
[01:50:25 -> 01:50:29]  Transformer you just kind of tokenize
[01:50:27 -> 01:50:31]  your input domains and then call the day
[01:50:29 -> 01:50:33]  and pretend it's just text tokens and
[01:50:31 -> 01:50:36]  just do everything else identical in an
[01:50:33 -> 01:50:37]  identical manner so here for example
[01:50:36 -> 01:50:39]  there was a early paper that has nice
[01:50:37 -> 01:50:42]  graphic for how you can take an image
[01:50:39 -> 01:50:45]  and you can chunc at it into
[01:50:42 -> 01:50:46]  integers um and these sometimes uh so
[01:50:45 -> 01:50:49]  these will basically become the tokens
[01:50:46 -> 01:50:52]  of images as an example and uh these
[01:50:49 -> 01:50:53]  tokens can be uh hard tokens where you
[01:50:52 -> 01:50:57]  force them to be integers they can also
[01:50:53 -> 01:51:00]  be soft tokens where you uh sort of
[01:50:57 -> 01:51:02]  don't require uh these to be discrete
[01:51:00 -> 01:51:04]  but you do Force these representations
[01:51:02 -> 01:51:06]  to go through bottlenecks like in Auto
[01:51:04 -> 01:51:08]  encoders uh also in this paper that came
[01:51:06 -> 01:51:11]  out from open a SORA which I think
[01:51:08 -> 01:51:13]  really um uh blew the mind of many
[01:51:11 -> 01:51:15]  people and inspired a lot of people in
[01:51:13 -> 01:51:16]  terms of what's possible they have a
[01:51:15 -> 01:51:20]  Graphic here and they talk briefly about
[01:51:16 -> 01:51:22]  how llms have text tokens Sora has
[01:51:20 -> 01:51:24]  visual patches so again they came up
[01:51:22 -> 01:51:26]  with a way to chunc a videos into
[01:51:24 -> 01:51:28]  basically tokens when they own
[01:51:26 -> 01:51:30]  vocabularies and then you can either
[01:51:28 -> 01:51:32]  process discrete tokens say with autog
[01:51:30 -> 01:51:35]  regressive models or even soft tokens
[01:51:32 -> 01:51:38]  with diffusion models and uh all of that
[01:51:35 -> 01:51:39]  is sort of uh being actively worked on
[01:51:38 -> 01:51:40]  designed on and is beyond the scope of
[01:51:39 -> 01:51:42]  this video but just something I wanted
[01:51:40 -> 01:51:45]  to mention briefly okay now that we have
[01:51:42 -> 01:51:46]  come quite deep into the tokenization
[01:51:45 -> 01:51:48]  algorithm and we understand a lot more
[01:51:46 -> 01:51:50]  about how it works let's loop back
[01:51:48 -> 01:51:51]  around to the beginning of this video
[01:51:50 -> 01:51:54]  and go through some of these bullet
[01:51:51 -> 01:51:56]  points and really see why they happen so
[01:51:54 -> 01:51:58]  first of all why can't my llm spell
[01:51:56 -> 01:52:00]  words very well or do other spell
[01:51:58 -> 01:52:02]  related
[01:52:00 -> 01:52:05]  tasks so fundamentally this is because
[01:52:02 -> 01:52:07]  as we saw these characters are chunked
[01:52:05 -> 01:52:10]  up into tokens and some of these tokens
[01:52:07 -> 01:52:12]  are actually fairly long so as an
[01:52:10 -> 01:52:15]  example I went to the gp4 vocabulary and
[01:52:12 -> 01:52:17]  I looked at uh one of the longer tokens
[01:52:15 -> 01:52:19]  so that default style turns out to be a
[01:52:17 -> 01:52:22]  single individual token so that's a lot
[01:52:19 -> 01:52:23]  of characters for a single token so my
[01:52:22 -> 01:52:26]  suspicion is that there's just too much
[01:52:23 -> 01:52:27]  crammed into this single token and my
[01:52:26 -> 01:52:30]  suspicion was that the model should not
[01:52:27 -> 01:52:34]  be very good at tasks related to
[01:52:30 -> 01:52:37]  spelling of this uh single token so I
[01:52:34 -> 01:52:41]  asked how many letters L are there in
[01:52:37 -> 01:52:44]  the word default style and of course my
[01:52:41 -> 01:52:45]  prompt is intentionally done that way
[01:52:44 -> 01:52:47]  and you see how default style will be a
[01:52:45 -> 01:52:49]  single token so this is what the model
[01:52:47 -> 01:52:51]  sees so my suspicion is that it wouldn't
[01:52:49 -> 01:52:53]  be very good at this and indeed it is
[01:52:51 -> 01:52:54]  not it doesn't actually know how many
[01:52:53 -> 01:52:57]  L's are in there it thinks there are
[01:52:54 -> 01:52:59]  three and actually there are four if I'm
[01:52:57 -> 01:53:02]  not getting this wrong myself so that
[01:52:59 -> 01:53:04]  didn't go extremely well let's look look
[01:53:02 -> 01:53:08]  at another kind of uh character level
[01:53:04 -> 01:53:11]  task so for example here I asked uh gp4
[01:53:08 -> 01:53:13]  to reverse the string default style and
[01:53:11 -> 01:53:15]  they tried to use a code interpreter and
[01:53:13 -> 01:53:19]  I stopped it and I said just do it just
[01:53:15 -> 01:53:21]  try it and uh it gave me jumble so it
[01:53:19 -> 01:53:23]  doesn't actually really know how to
[01:53:21 -> 01:53:26]  reverse this string going from right to
[01:53:23 -> 01:53:28]  left uh so it gave a wrong result so
[01:53:26 -> 01:53:30]  again like working with this working
[01:53:28 -> 01:53:31]  hypothesis that maybe this is due to the
[01:53:30 -> 01:53:34]  tokenization I tried a different
[01:53:31 -> 01:53:36]  approach I said okay let's reverse the
[01:53:34 -> 01:53:38]  exact same string but take the following
[01:53:36 -> 01:53:40]  approach step one just print out every
[01:53:38 -> 01:53:43]  single character separated by spaces and
[01:53:40 -> 01:53:44]  then as a step two reverse that list and
[01:53:43 -> 01:53:47]  it again Tred to use a tool but when I
[01:53:44 -> 01:53:48]  stopped it it uh first uh produced all
[01:53:47 -> 01:53:50]  the characters and that was actually
[01:53:48 -> 01:53:53]  correct and then It reversed them and
[01:53:50 -> 01:53:54]  that was correct once it had this so
[01:53:53 -> 01:53:57]  somehow it can't reverse it directly but
[01:53:54 -> 01:53:59]  when you go just first uh you know
[01:53:57 -> 01:54:01]  listing it out in order it can do that
[01:53:59 -> 01:54:03]  somehow and then it can once it's uh
[01:54:01 -> 01:54:06]  broken up this way this becomes all
[01:54:03 -> 01:54:07]  these individual characters and so now
[01:54:06 -> 01:54:10]  this is much easier for it to see these
[01:54:07 -> 01:54:13]  individual tokens and reverse them and
[01:54:10 -> 01:54:16]  print them out so that is kind of
[01:54:13 -> 01:54:20]  interesting so let's continue now why
[01:54:16 -> 01:54:22]  are llms worse at uh non-english langu
[01:54:20 -> 01:54:24]  and I briefly covered this already but
[01:54:22 -> 01:54:27]  basically um it's not only that the
[01:54:24 -> 01:54:28]  language model sees less non-english
[01:54:27 -> 01:54:31]  data during training of the model
[01:54:28 -> 01:54:34]  parameters but also the tokenizer is not
[01:54:31 -> 01:54:37]  um is not sufficiently trained on
[01:54:34 -> 01:54:40]  non-english data and so here for example
[01:54:37 -> 01:54:42]  hello how are you is five tokens and its
[01:54:40 -> 01:54:45]  translation is 15 tokens so this is a
[01:54:42 -> 01:54:48]  three times blow up and so for example
[01:54:45 -> 01:54:50]  anang is uh just hello basically in
[01:54:48 -> 01:54:51]  Korean and that end up being three
[01:54:50 -> 01:54:53]  tokens I'm actually kind of surprised by
[01:54:51 -> 01:54:55]  that because that is a very common
[01:54:53 -> 01:54:57]  phrase there just the typical greeting
[01:54:55 -> 01:54:58]  of like hello and that ends up being
[01:54:57 -> 01:55:00]  three tokens whereas our hello is a
[01:54:58 -> 01:55:02]  single token and so basically everything
[01:55:00 -> 01:55:04]  is a lot more bloated and diffuse and
[01:55:02 -> 01:55:07]  this is I think partly the reason that
[01:55:04 -> 01:55:10]  the model Works worse on other
[01:55:07 -> 01:55:13]  languages uh coming back why is LM bad
[01:55:10 -> 01:55:17]  at simple arithmetic um that has to do
[01:55:13 -> 01:55:19]  with the tokenization of numbers and so
[01:55:17 -> 01:55:20]  um you'll notice that for example
[01:55:19 -> 01:55:23]  addition is very sort of
[01:55:20 -> 01:55:25]  like uh there's an algorithm that is
[01:55:23 -> 01:55:27]  like character level for doing addition
[01:55:25 -> 01:55:29]  so for example here we would first add
[01:55:27 -> 01:55:31]  the ones and then the tens and then the
[01:55:29 -> 01:55:34]  hundreds you have to refer to specific
[01:55:31 -> 01:55:36]  parts of these digits but uh these
[01:55:34 -> 01:55:37]  numbers are represented completely
[01:55:36 -> 01:55:39]  arbitrarily based on whatever happened
[01:55:37 -> 01:55:41]  to merge or not merge during the
[01:55:39 -> 01:55:42]  tokenization process there's an entire
[01:55:41 -> 01:55:44]  blog post about this that I think is
[01:55:42 -> 01:55:46]  quite good integer tokenization is
[01:55:44 -> 01:55:48]  insane and this person basically
[01:55:46 -> 01:55:52]  systematically explores the tokenization
[01:55:48 -> 01:55:53]  of numbers in I believe this is gpt2 and
[01:55:52 -> 01:55:57]  so they notice that for example for the
[01:55:53 -> 01:56:00]  for um four-digit numbers you can take a
[01:55:57 -> 01:56:02]  look at whether it is uh a single token
[01:56:00 -> 01:56:04]  or whether it is two tokens that is a 1
[01:56:02 -> 01:56:06]  three or a 2 two or a 31 combination and
[01:56:04 -> 01:56:08]  so all the different numbers are all the
[01:56:06 -> 01:56:09]  different combinations and you can
[01:56:08 -> 01:56:11]  imagine this is all completely
[01:56:09 -> 01:56:14]  arbitrarily so and the model
[01:56:11 -> 01:56:16]  unfortunately sometimes sees uh four um
[01:56:14 -> 01:56:18]  a token for for all four digits
[01:56:16 -> 01:56:20]  sometimes for three sometimes for two
[01:56:18 -> 01:56:22]  sometimes for one and it's in an
[01:56:20 -> 01:56:25]  arbitrary uh Manner and so this is
[01:56:22 -> 01:56:26]  definitely a headwind if you will for
[01:56:25 -> 01:56:27]  the language model and it's kind of
[01:56:26 -> 01:56:30]  incredible that it can kind of do it and
[01:56:27 -> 01:56:32]  deal with it but it's also kind of not
[01:56:30 -> 01:56:34]  ideal and so that's why for example we
[01:56:32 -> 01:56:36]  saw that meta when they train the Llama
[01:56:34 -> 01:56:39]  2 algorithm and they use sentence piece
[01:56:36 -> 01:56:42]  they make sure to split up all the um
[01:56:39 -> 01:56:44]  all the digits as an example for uh
[01:56:42 -> 01:56:46]  llama 2 and this is partly to improve a
[01:56:44 -> 01:56:50]  simple arithmetic kind of
[01:56:46 -> 01:56:52]  performance and finally why is gpt2 not
[01:56:50 -> 01:56:54]  as good in Python again this is partly a
[01:56:52 -> 01:56:56]  modeling issue on in the architecture
[01:56:54 -> 01:56:58]  and the data set and the strength of the
[01:56:56 -> 01:57:00]  model but it's also partially
[01:56:58 -> 01:57:03]  tokenization because as we saw here with
[01:57:00 -> 01:57:05]  the simple python example the encoding
[01:57:03 -> 01:57:07]  efficiency of the tokenizer for handling
[01:57:05 -> 01:57:09]  spaces in Python is terrible and every
[01:57:07 -> 01:57:11]  single space is an individual token and
[01:57:09 -> 01:57:12]  this dramatically reduces the context
[01:57:11 -> 01:57:14]  length that the model can attend to
[01:57:12 -> 01:57:16]  cross so that's almost like a
[01:57:14 -> 01:57:20]  tokenization bug for gpd2 and that was
[01:57:16 -> 01:57:22]  later fixed with gp4 okay so here's
[01:57:20 -> 01:57:25]  another fun one my llm abruptly halts
[01:57:22 -> 01:57:28]  when it sees the string end of text so
[01:57:25 -> 01:57:30]  here's um here's a very strange Behavior
[01:57:28 -> 01:57:32]  print a string end of text is what I
[01:57:30 -> 01:57:35]  told jt4 and it says could you please
[01:57:32 -> 01:57:37]  specify the string and I'm I'm telling
[01:57:35 -> 01:57:39]  it give me end of text and it seems like
[01:57:37 -> 01:57:41]  there's an issue it's not seeing end of
[01:57:39 -> 01:57:44]  text and then I give it end of text is
[01:57:41 -> 01:57:45]  the string and then here's a string and
[01:57:44 -> 01:57:47]  then it just doesn't print it so
[01:57:45 -> 01:57:48]  obviously something is breaking here
[01:57:47 -> 01:57:50]  with respect to the handling of the
[01:57:48 -> 01:57:52]  special token and I don't actually know
[01:57:50 -> 01:57:54]  what open ey is doing under the hood
[01:57:52 -> 01:57:58]  here and whether they are potentially
[01:57:54 -> 01:58:01]  parsing this as an um as an actual token
[01:57:58 -> 01:58:04]  instead of this just being uh end of
[01:58:01 -> 01:58:06]  text um as like individual sort of
[01:58:04 -> 01:58:09]  pieces of it without the special token
[01:58:06 -> 01:58:11]  handling logic and so it might be that
[01:58:09 -> 01:58:13]  someone when they're calling do encode
[01:58:11 -> 01:58:16]  uh they are passing in the allowed
[01:58:13 -> 01:58:18]  special and they are allowing end of
[01:58:16 -> 01:58:20]  text as a special character in the user
[01:58:18 -> 01:58:23]  prompt but the user prompt of course is
[01:58:20 -> 01:58:25]  is a sort of um attacker controlled text
[01:58:23 -> 01:58:28]  so you would hope that they don't really
[01:58:25 -> 01:58:30]  parse or use special tokens or you know
[01:58:28 -> 01:58:31]  from that kind of input but it appears
[01:58:30 -> 01:58:34]  that there's something definitely going
[01:58:31 -> 01:58:36]  wrong here and um so your knowledge of
[01:58:34 -> 01:58:38]  these special tokens ends up being in a
[01:58:36 -> 01:58:43]  tax surface potentially and so if you'd
[01:58:38 -> 01:58:44]  like to confuse llms then just um try to
[01:58:43 -> 01:58:46]  give them some special tokens and see if
[01:58:44 -> 01:58:49]  you're breaking something by chance okay
[01:58:46 -> 01:58:52]  so this next one is a really fun one uh
[01:58:49 -> 01:58:56]  the trailing whites space issue so if
[01:58:52 -> 01:58:58]  you come to playground and uh we come
[01:58:56 -> 01:59:00]  here to GPT 3.5 turbo instruct so this
[01:58:58 -> 01:59:02]  is not a chat model this is a completion
[01:59:00 -> 01:59:05]  model so think of it more like it's a
[01:59:02 -> 01:59:07]  lot more closer to a base model it does
[01:59:05 -> 01:59:09]  completion it will continue the token
[01:59:07 -> 01:59:11]  sequence so here's a tagline for ice
[01:59:09 -> 01:59:14]  cream shop and we want to continue the
[01:59:11 -> 01:59:18]  sequence and so we can submit and get a
[01:59:14 -> 01:59:20]  bunch of tokens okay no problem but now
[01:59:18 -> 01:59:23]  suppose I do this but instead of
[01:59:20 -> 01:59:26]  pressing submit here I do here's a
[01:59:23 -> 01:59:28]  tagline for ice cream shop space so I
[01:59:26 -> 01:59:31]  have a space here before I click
[01:59:28 -> 01:59:33]  submit we get a warning your text ends
[01:59:31 -> 01:59:35]  in a trail Ling space which causes worse
[01:59:33 -> 01:59:38]  performance due to how API splits text
[01:59:35 -> 01:59:40]  into tokens so what's happening here it
[01:59:38 -> 01:59:42]  still gave us a uh sort of completion
[01:59:40 -> 01:59:44]  here but let's take a look at what's
[01:59:42 -> 01:59:48]  happening so here's a tagline for an ice
[01:59:44 -> 01:59:50]  cream shop and then what does this look
[01:59:48 -> 01:59:52]  like in the actual actual training data
[01:59:50 -> 01:59:53]  suppose you found the completion in the
[01:59:52 -> 01:59:55]  training document somewhere on the
[01:59:53 -> 01:59:58]  internet and the llm trained on this
[01:59:55 -> 02:00:00]  data so maybe it's something like oh
[01:59:58 -> 02:00:02]  yeah maybe that's the tagline that's a
[02:00:00 -> 02:00:05]  terrible tagline but notice here that
[02:00:02 -> 02:00:07]  when I create o you see that because
[02:00:05 -> 02:00:11]  there's the the space character is
[02:00:07 -> 02:00:13]  always a prefix to these tokens in GPT
[02:00:11 -> 02:00:16]  so it's not an O token it's a space o
[02:00:13 -> 02:00:19]  token the space is part of the O and
[02:00:16 -> 02:00:21]  together they are token 8840 that's
[02:00:19 -> 02:00:24]  that's space o so what's What's
[02:00:21 -> 02:00:27]  Happening Here is that when I just have
[02:00:24 -> 02:00:30]  it like this and I let it complete the
[02:00:27 -> 02:00:32]  next token it can sample the space o
[02:00:30 -> 02:00:34]  token but instead if I have this and I
[02:00:32 -> 02:00:37]  add my space then what I'm doing here
[02:00:34 -> 02:00:39]  when I incode this string is I have
[02:00:37 -> 02:00:42]  basically here's a t line for an ice
[02:00:39 -> 02:00:44]  cream uh shop and this space at the very
[02:00:42 -> 02:00:47]  end becomes a token
[02:00:44 -> 02:00:49]  220 and so we've added token 220 and
[02:00:47 -> 02:00:51]  this token otherwise would be part of
[02:00:49 -> 02:00:55]  the tagline because if there actually is
[02:00:51 -> 02:00:57]  a tagline here so space o is the token
[02:00:55 -> 02:00:59]  and so this is suddenly a of
[02:00:57 -> 02:01:01]  distribution for the model because this
[02:00:59 -> 02:01:04]  space is part of the next token but
[02:01:01 -> 02:01:07]  we're putting it here like this and the
[02:01:04 -> 02:01:10]  model has seen very very little data of
[02:01:07 -> 02:01:11]  actual Space by itself and we're asking
[02:01:10 -> 02:01:13]  it to complete the sequence like add in
[02:01:11 -> 02:01:16]  more tokens but the problem is that
[02:01:13 -> 02:01:18]  we've sort of begun the first token and
[02:01:16 -> 02:01:20]  now it's been split up and now we're out
[02:01:18 -> 02:01:23]  of this distribution and now arbitrary
[02:01:20 -> 02:01:24]  bad things happen and it's just a very
[02:01:23 -> 02:01:26]  rare example for it to see something
[02:01:24 -> 02:01:29]  like that and uh that's why we get the
[02:01:26 -> 02:01:32]  warning so the fundamental issue here is
[02:01:29 -> 02:01:34]  of course that um the llm is on top of
[02:01:32 -> 02:01:36]  these tokens and these tokens are text
[02:01:34 -> 02:01:38]  chunks they're not characters in a way
[02:01:36 -> 02:01:40]  you and I would think of them they are
[02:01:38 -> 02:01:41]  these are the atoms of what the LM is
[02:01:40 -> 02:01:43]  seeing and there's a bunch of weird
[02:01:41 -> 02:01:48]  stuff that comes out of it let's go back
[02:01:43 -> 02:01:49]  to our default cell style I bet you that
[02:01:48 -> 02:01:54]  the model has never in its training set
[02:01:49 -> 02:01:56]  seen default cell sta without Le in
[02:01:54 -> 02:01:59]  there it's always seen this as a single
[02:01:56 -> 02:02:02]  group because uh this is some kind of a
[02:01:59 -> 02:02:03]  function in um I'm guess I don't
[02:02:02 -> 02:02:05]  actually know what this is part of this
[02:02:03 -> 02:02:07]  is some kind of API but I bet you that
[02:02:05 -> 02:02:10]  it's never seen this combination of
[02:02:07 -> 02:02:12]  tokens uh in its training data because
[02:02:10 -> 02:02:14]  or I think it would be extremely rare so
[02:02:12 -> 02:02:17]  I took this and I copy pasted it here
[02:02:14 -> 02:02:19]  and I had I tried to complete from it
[02:02:17 -> 02:02:21]  and the it immediately gave me a big
[02:02:19 -> 02:02:22]  error and it said the model predicted to
[02:02:21 -> 02:02:24]  completion that begins with a stop
[02:02:22 -> 02:02:26]  sequence resulting in no output consider
[02:02:24 -> 02:02:27]  adjusting your prompt or stop sequences
[02:02:26 -> 02:02:30]  so what happened here when I clicked
[02:02:27 -> 02:02:32]  submit is that immediately the model
[02:02:30 -> 02:02:34]  emitted and sort of like end of text
[02:02:32 -> 02:02:36]  token I think or something like that it
[02:02:34 -> 02:02:38]  basically predicted the stop sequence
[02:02:36 -> 02:02:40]  immediately so it had no completion and
[02:02:38 -> 02:02:42]  so this is why I'm getting a warning
[02:02:40 -> 02:02:45]  again because we're off the data
[02:02:42 -> 02:02:47]  distribution and the model is just uh
[02:02:45 -> 02:02:49]  predicting just totally arbitrary things
[02:02:47 -> 02:02:50]  it's just really confused basically this
[02:02:49 -> 02:02:53]  is uh this is giving it brain damage
[02:02:50 -> 02:02:54]  it's never seen this before it's shocked
[02:02:53 -> 02:02:57]  and it's predicting end of text or
[02:02:54 -> 02:02:59]  something I tried it again here and it
[02:02:57 -> 02:03:01]  in this case it completed it but then
[02:02:59 -> 02:03:03]  for some reason this request May violate
[02:03:01 -> 02:03:06]  our usage policies this was
[02:03:03 -> 02:03:07]  flagged um basically something just like
[02:03:06 -> 02:03:09]  goes wrong and there's something like
[02:03:07 -> 02:03:11]  Jank you can just feel the Jank because
[02:03:09 -> 02:03:12]  the model is like extremely unhappy with
[02:03:11 -> 02:03:14]  just this and it doesn't know how to
[02:03:12 -> 02:03:16]  complete it because it's never occurred
[02:03:14 -> 02:03:18]  in training set in a training set it
[02:03:16 -> 02:03:20]  always appears like this and becomes a
[02:03:18 -> 02:03:21]  single token
[02:03:20 -> 02:03:24]  so these kinds of issues where tokens
[02:03:21 -> 02:03:26]  are either you sort of like complete the
[02:03:24 -> 02:03:28]  first character of the next token or you
[02:03:26 -> 02:03:29]  are sort of you have long tokens that
[02:03:28 -> 02:03:32]  you then have just some of the
[02:03:29 -> 02:03:35]  characters off all of these are kind of
[02:03:32 -> 02:03:37]  like issues with partial tokens is how I
[02:03:35 -> 02:03:39]  would describe it and if you actually
[02:03:37 -> 02:03:41]  dig into the T token
[02:03:39 -> 02:03:44]  repository go to the rust code and
[02:03:41 -> 02:03:47]  search for
[02:03:44 -> 02:03:49]  unstable and you'll see um en code
[02:03:47 -> 02:03:51]  unstable native unstable token tokens
[02:03:49 -> 02:03:53]  and a lot of like special case handling
[02:03:51 -> 02:03:55]  none of this stuff about unstable tokens
[02:03:53 -> 02:03:58]  is documented anywhere but there's a ton
[02:03:55 -> 02:04:00]  of code dealing with unstable tokens and
[02:03:58 -> 02:04:02]  unstable tokens is exactly kind of like
[02:04:00 -> 02:04:05]  what I'm describing here what you would
[02:04:02 -> 02:04:06]  like out of a completion API is
[02:04:05 -> 02:04:08]  something a lot more fancy like if we're
[02:04:06 -> 02:04:10]  putting in default cell sta if we're
[02:04:08 -> 02:04:12]  asking for the next token sequence we're
[02:04:10 -> 02:04:14]  not actually trying to append the next
[02:04:12 -> 02:04:16]  token exactly after this list we're
[02:04:14 -> 02:04:19]  actually trying to append we're trying
[02:04:16 -> 02:04:22]  to consider lots of tokens um
[02:04:19 -> 02:04:25]  that if we were or I guess like we're
[02:04:22 -> 02:04:28]  trying to search over characters that if
[02:04:25 -> 02:04:30]  we retened would be of high probability
[02:04:28 -> 02:04:32]  if that makes sense um so that we can
[02:04:30 -> 02:04:34]  actually add a single individual
[02:04:32 -> 02:04:36]  character uh instead of just like adding
[02:04:34 -> 02:04:39]  the next full token that comes after
[02:04:36 -> 02:04:41]  this partial token list so I this is
[02:04:39 -> 02:04:43]  very tricky to describe and I invite you
[02:04:41 -> 02:04:44]  to maybe like look through this it ends
[02:04:43 -> 02:04:46]  up being extremely gnarly and hairy kind
[02:04:44 -> 02:04:49]  of topic it and it comes from
[02:04:46 -> 02:04:50]  tokenization fundamentally so um maybe I
[02:04:49 -> 02:04:52]  can even spend an entire video talking
[02:04:50 -> 02:04:54]  about unstable tokens sometime in the
[02:04:52 -> 02:04:56]  future okay and I'm really saving the
[02:04:54 -> 02:04:59]  best for last my favorite one by far is
[02:04:56 -> 02:05:01]  the solid gold
[02:04:59 -> 02:05:03]  Magikarp and it just okay so this comes
[02:05:01 -> 02:05:07]  from this blog post uh solid gold
[02:05:03 -> 02:05:10]  Magikarp and uh this is um internet
[02:05:07 -> 02:05:11]  famous now for those of us in llms and
[02:05:10 -> 02:05:13]  basically I I would advise you to uh
[02:05:11 -> 02:05:16]  read this block Post in full but
[02:05:13 -> 02:05:19]  basically what this person was doing is
[02:05:16 -> 02:05:22]  this person went to the um
[02:05:19 -> 02:05:24]  token embedding stable and clustered the
[02:05:22 -> 02:05:27]  tokens based on their embedding
[02:05:24 -> 02:05:29]  representation and this person noticed
[02:05:27 -> 02:05:31]  that there's a cluster of tokens that
[02:05:29 -> 02:05:34]  look really strange so there's a cluster
[02:05:31 -> 02:05:36]  here at rot e stream Fame solid gold
[02:05:34 -> 02:05:39]  Magikarp Signet message like really
[02:05:36 -> 02:05:42]  weird tokens in uh basically in this
[02:05:39 -> 02:05:43]  embedding cluster and so what are these
[02:05:42 -> 02:05:45]  tokens and where do they even come from
[02:05:43 -> 02:05:48]  like what is solid gold magikarpet makes
[02:05:45 -> 02:05:50]  no sense and then they found bunch of
[02:05:48 -> 02:05:52]  these
[02:05:50 -> 02:05:53]  tokens and then they notice that
[02:05:52 -> 02:05:56]  actually the plot thickens here because
[02:05:53 -> 02:05:58]  if you ask the model about these tokens
[02:05:56 -> 02:06:00]  like you ask it uh some very benign
[02:05:58 -> 02:06:02]  question like please can you repeat back
[02:06:00 -> 02:06:04]  to me the string sold gold Magikarp uh
[02:06:02 -> 02:06:07]  then you get a variety of basically
[02:06:04 -> 02:06:09]  totally broken llm Behavior so either
[02:06:07 -> 02:06:11]  you get evasion so I'm sorry I can't
[02:06:09 -> 02:06:14]  hear you or you get a bunch of
[02:06:11 -> 02:06:17]  hallucinations as a response um you can
[02:06:14 -> 02:06:20]  even get back like insults so you ask it
[02:06:17 -> 02:06:22]  uh about streamer bot it uh tells the
[02:06:20 -> 02:06:24]  and the model actually just calls you
[02:06:22 -> 02:06:26]  names uh or it kind of comes up with
[02:06:24 -> 02:06:28]  like weird humor like you're actually
[02:06:26 -> 02:06:30]  breaking the model by asking about these
[02:06:28 -> 02:06:32]  very simple strings like at Roth and
[02:06:30 -> 02:06:34]  sold gold Magikarp so like what the hell
[02:06:32 -> 02:06:37]  is happening and there's a variety of
[02:06:34 -> 02:06:38]  here documented behaviors uh there's a
[02:06:37 -> 02:06:40]  bunch of tokens not just so good
[02:06:38 -> 02:06:42]  Magikarp that have that kind of a
[02:06:40 -> 02:06:44]  behavior and so basically there's a
[02:06:42 -> 02:06:46]  bunch of like trigger words and if you
[02:06:44 -> 02:06:48]  ask the model about these trigger words
[02:06:46 -> 02:06:50]  or you just include them in your prompt
[02:06:48 -> 02:06:52]  the model goes haywire and has all kinds
[02:06:50 -> 02:06:54]  of uh really Strange Behaviors including
[02:06:52 -> 02:06:57]  sort of ones that violate typical safety
[02:06:54 -> 02:06:59]  guidelines uh and the alignment of the
[02:06:57 -> 02:07:01]  model like it's swearing back at you so
[02:06:59 -> 02:07:04]  what is happening here and how can this
[02:07:01 -> 02:07:06]  possibly be true well this again comes
[02:07:04 -> 02:07:08]  down to tokenization so what's happening
[02:07:06 -> 02:07:11]  here is that sold gold Magikarp if you
[02:07:08 -> 02:07:14]  actually dig into it is a Reddit user so
[02:07:11 -> 02:07:16]  there's a u Sol gold
[02:07:14 -> 02:07:18]  Magikarp and probably what happened here
[02:07:16 -> 02:07:20]  even though I I don't know that this has
[02:07:18 -> 02:07:23]  been like really definitively explored
[02:07:20 -> 02:07:25]  but what is thought to have happened is
[02:07:23 -> 02:07:28]  that the tokenization data set was very
[02:07:25 -> 02:07:29]  different from the training data set for
[02:07:28 -> 02:07:31]  the actual language model so in the
[02:07:29 -> 02:07:34]  tokenization data set there was a ton of
[02:07:31 -> 02:07:36]  redded data potentially where the user
[02:07:34 -> 02:07:39]  solid gold Magikarp was mentioned in the
[02:07:36 -> 02:07:41]  text because solid gold Magikarp was a
[02:07:39 -> 02:07:43]  very common um sort of uh person who
[02:07:41 -> 02:07:45]  would post a lot uh this would be a
[02:07:43 -> 02:07:48]  string that occurs many times in a
[02:07:45 -> 02:07:50]  tokenization data set because it occurs
[02:07:48 -> 02:07:51]  many times in a tokenization data set
[02:07:50 -> 02:07:53]  these tokens would end up getting merged
[02:07:51 -> 02:07:56]  to the single individual token for that
[02:07:53 -> 02:07:58]  single Reddit user sold gold Magikarp so
[02:07:56 -> 02:08:00]  they would have a dedicated token in a
[02:07:58 -> 02:08:04]  vocabulary of was it 50,000 tokens in
[02:08:00 -> 02:08:05]  gpd2 that is devoted to that Reddit user
[02:08:04 -> 02:08:08]  and then what happens is the
[02:08:05 -> 02:08:10]  tokenization data set has those strings
[02:08:08 -> 02:08:13]  but then later when you train the model
[02:08:10 -> 02:08:16]  the language model itself um this data
[02:08:13 -> 02:08:18]  from Reddit was not present and so
[02:08:16 -> 02:08:21]  therefore in the entire training set for
[02:08:18 -> 02:08:24]  the language model sold gold Magikarp
[02:08:21 -> 02:08:25]  never occurs that token never appears in
[02:08:24 -> 02:08:28]  the training set for the actual language
[02:08:25 -> 02:08:31]  model later so this token never gets
[02:08:28 -> 02:08:32]  activated it's initialized at random in
[02:08:31 -> 02:08:34]  the beginning of optimization then you
[02:08:32 -> 02:08:36]  have forward backward passes and updates
[02:08:34 -> 02:08:37]  to the model and this token is just
[02:08:36 -> 02:08:40]  never updated in the embedding table
[02:08:37 -> 02:08:42]  that row Vector never gets sampled it
[02:08:40 -> 02:08:43]  never gets used so it never gets trained
[02:08:42 -> 02:08:46]  and it's completely untrained it's kind
[02:08:43 -> 02:08:48]  of like unallocated memory in a typical
[02:08:46 -> 02:08:50]  binary program written in C or something
[02:08:48 -> 02:08:51]  like that that so it's unallocated
[02:08:50 -> 02:08:54]  memory and then at test time if you
[02:08:51 -> 02:08:55]  evoke this token then you're basically
[02:08:54 -> 02:08:57]  plucking out a row of the embedding
[02:08:55 -> 02:08:58]  table that is completely untrained and
[02:08:57 -> 02:09:00]  that feeds into a Transformer and
[02:08:58 -> 02:09:02]  creates undefined behavior and that's
[02:09:00 -> 02:09:03]  what we're seeing here this completely
[02:09:02 -> 02:09:06]  undefined never before seen in a
[02:09:03 -> 02:09:08]  training behavior and so any of these
[02:09:06 -> 02:09:09]  kind of like weird tokens would evoke
[02:09:08 -> 02:09:14]  this Behavior because fundamentally the
[02:09:09 -> 02:09:16]  model is um is uh uh out of sample out
[02:09:14 -> 02:09:18]  of distribution okay and the very last
[02:09:16 -> 02:09:19]  thing I wanted to just briefly mention
[02:09:18 -> 02:09:21]  point out although I think a lot of
[02:09:19 -> 02:09:23]  people are quite aware of this is that
[02:09:21 -> 02:09:25]  different kinds of formats and different
[02:09:23 -> 02:09:26]  representations and different languages
[02:09:25 -> 02:09:29]  and so on might be more or less
[02:09:26 -> 02:09:31]  efficient with GPD tokenizers uh or any
[02:09:29 -> 02:09:33]  tokenizers for any other L for that
[02:09:31 -> 02:09:36]  matter so for example Json is actually
[02:09:33 -> 02:09:39]  really dense in tokens and yaml is a lot
[02:09:36 -> 02:09:41]  more efficient in tokens um so for
[02:09:39 -> 02:09:44]  example this are these are the same in
[02:09:41 -> 02:09:48]  Json and in yaml the Json is
[02:09:44 -> 02:09:51]  116 and the yaml is 99 so quite a bit of
[02:09:48 -> 02:09:53]  an Improvement and so in the token
[02:09:51 -> 02:09:55]  economy where we are paying uh per token
[02:09:53 -> 02:09:57]  in many ways and you are paying in the
[02:09:55 -> 02:09:59]  context length and you're paying in um
[02:09:57 -> 02:10:01]  dollar amount for uh the cost of
[02:09:59 -> 02:10:03]  processing all this kind of structured
[02:10:01 -> 02:10:06]  data when you have to um so prefer to
[02:10:03 -> 02:10:07]  use theal over Json and in general kind
[02:10:06 -> 02:10:09]  of like the tokenization density is
[02:10:07 -> 02:10:11]  something that you have to um sort of
[02:10:09 -> 02:10:13]  care about and worry about at all times
[02:10:11 -> 02:10:15]  and try to find efficient encoding
[02:10:13 -> 02:10:16]  schemes and spend a lot of time in tick
[02:10:15 -> 02:10:18]  tokenizer and measure the different
[02:10:16 -> 02:10:21]  token efficiencies of different formats
[02:10:18 -> 02:10:23]  and settings and so on okay so that
[02:10:21 -> 02:10:25]  concludes my fairly long video on
[02:10:23 -> 02:10:28]  tokenization I know it's a try I know
[02:10:25 -> 02:10:30]  it's annoying I know it's irritating I
[02:10:28 -> 02:10:32]  personally really dislike the stage what
[02:10:30 -> 02:10:34]  I do have to say at this point is don't
[02:10:32 -> 02:10:38]  brush it off there's a lot of foot guns
[02:10:34 -> 02:10:39]  sharp edges here security issues uh AI
[02:10:38 -> 02:10:42]  safety issues as we saw plugging in
[02:10:39 -> 02:10:45]  unallocated memory into uh language
[02:10:42 -> 02:10:48]  models so um it's worth understanding
[02:10:45 -> 02:10:50]  this stage um that said I will say that
[02:10:48 -> 02:10:52]  eternal glory goes to anyone who can get
[02:10:50 -> 02:10:54]  rid of it uh I showed you one possible
[02:10:52 -> 02:10:57]  paper that tried to uh do that and I
[02:10:54 -> 02:10:59]  think I hope a lot more can follow over
[02:10:57 -> 02:11:01]  time and my final recommendations for
[02:10:59 -> 02:11:03]  the application right now are if you can
[02:11:01 -> 02:11:05]  reuse the GPT 4 tokens and the
[02:11:03 -> 02:11:06]  vocabulary uh in your application then
[02:11:05 -> 02:11:07]  that's something you should consider and
[02:11:06 -> 02:11:11]  just use Tech token because it is very
[02:11:07 -> 02:11:13]  efficient and nice library for inference
[02:11:11 -> 02:11:17]  for bpe I also really like the bite
[02:11:13 -> 02:11:19]  level BP that uh Tik toen and openi uses
[02:11:17 -> 02:11:22]  uh if you for some reason want to train
[02:11:19 -> 02:11:25]  your own vocabulary from scratch um then
[02:11:22 -> 02:11:28]  I would use uh the bpe with sentence
[02:11:25 -> 02:11:30]  piece um oops as I mentioned I'm not a
[02:11:28 -> 02:11:33]  huge fan of sentence piece I don't like
[02:11:30 -> 02:11:35]  its uh bite fallback and I don't like
[02:11:33 -> 02:11:37]  that it's doing BP on unic code code
[02:11:35 -> 02:11:39]  points I think it's uh it also has like
[02:11:37 -> 02:11:40]  a million settings and I think there's a
[02:11:39 -> 02:11:42]  lot of foot gonss here and I think it's
[02:11:40 -> 02:11:43]  really easy to Mis calibrate them and
[02:11:42 -> 02:11:45]  you end up cropping your sentences or
[02:11:43 -> 02:11:47]  something like that uh because of some
[02:11:45 -> 02:11:49]  type of parameter that you don't fully
[02:11:47 -> 02:11:51]  understand so so be very careful with
[02:11:49 -> 02:11:54]  the settings try to copy paste exactly
[02:11:51 -> 02:11:56]  maybe where what meta did or basically
[02:11:54 -> 02:11:57]  spend a lot of time looking at all the
[02:11:56 -> 02:11:59]  hyper parameters and go through the code
[02:11:57 -> 02:12:02]  of sentence piece and make sure that you
[02:11:59 -> 02:12:03]  have this correct um but even if you
[02:12:02 -> 02:12:04]  have all the settings correct I still
[02:12:03 -> 02:12:07]  think that the algorithm is kind of
[02:12:04 -> 02:12:09]  inferior to what's happening here and
[02:12:07 -> 02:12:11]  maybe the best if you really need to
[02:12:09 -> 02:12:13]  train your vocabulary maybe the best
[02:12:11 -> 02:12:16]  thing is to just wait for M bpe to
[02:12:13 -> 02:12:18]  becomes as efficient as possible and uh
[02:12:16 -> 02:12:20]  that's something that maybe I hope to
[02:12:18 -> 02:12:22]  work on and at some point maybe we can
[02:12:20 -> 02:12:24]  be training basically really what we
[02:12:22 -> 02:12:27]  want is we want tick token but training
[02:12:24 -> 02:12:31]  code and that is the ideal thing that
[02:12:27 -> 02:12:33]  currently does not exist and MBP is um
[02:12:31 -> 02:12:35]  is in implementation of it but currently
[02:12:33 -> 02:12:38]  it's in Python so that's currently what
[02:12:35 -> 02:12:40]  I have to say for uh tokenization there
[02:12:38 -> 02:12:41]  might be an advanced video that has even
[02:12:40 -> 02:12:43]  drier and even more detailed in the
[02:12:41 -> 02:12:46]  future but for now I think we're going
[02:12:43 -> 02:12:50]  to leave things off here and uh I hope
[02:12:46 -> 02:12:50]  that was helpful bye
[02:12:54 -> 02:13:02]  and uh they increase this contact size
[02:12:56 -> 02:13:05]  from gpt1 of 512 uh to 1024 and GPT 4
[02:13:02 -> 02:13:07]  two the
[02:13:05 -> 02:13:09]  next okay next I would like us to
[02:13:07 -> 02:13:13]  briefly walk through the code from open
[02:13:09 -> 02:13:13]  AI on the gpt2 encoded
[02:13:15 -> 02:13:21]  ATP I'm sorry I'm gonna sneeze
[02:13:19 -> 02:13:24]  and then what's Happening Here
[02:13:21 -> 02:13:26]  is this is a spous layer that I will
[02:13:24 -> 02:13:30]  explain in a
[02:13:26 -> 02:13:30]  bit What's Happening Here
[02:13:33 -> 02:13:36]  is
"""
    
    # Split into chunks
    chunks = split_transcript_with_time(sample_transcript, chunk_duration=300)  # 5 minute chunks
    
    # Process in parallel
    final_summary = summarize_all_chunks_parallel(chunks, num_processes=32)
    print(final_summary)
