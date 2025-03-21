from section import split_transcript_with_time

def test_basic_functionality():
    # Test case 1: Basic splitting with short duration
    transcript1 = """[00:00 -> 00:03] Hello, this is the first line
[00:03 -> 00:08] This is the second line
[00:08 -> 00:15] And this is the third line"""

    chunks = split_transcript_with_time(transcript1, chunk_duration=10)
    print("\nTest 1 - Basic splitting (10s chunks):")
    for i, (start, end, text) in enumerate(chunks):
        print(f"\nChunk {i+1} ({start} -> {end}):")
        print(text)

    # Test case 2: Single chunk (duration longer than total time)
    chunks = split_transcript_with_time(transcript1, chunk_duration=30)
    print("\nTest 2 - Single chunk (30s duration):")
    for i, (start, end, text) in enumerate(chunks):
        print(f"\nChunk {i+1} ({start} -> {end}):")
        print(text)

    # Test case 3: Empty lines and error handling
    transcript2 = """[00:00 -> 00:03] First line
    
[invalid timestamp] This should be skipped
[00:03 -> 00:08] Valid line
"""
    chunks = split_transcript_with_time(transcript2, chunk_duration=10)
    print("\nTest 3 - Error handling:")
    for i, (start, end, text) in enumerate(chunks):
        print(f"\nChunk {i+1} ({start} -> {end}):")
        print(text)

    # Test case 4: Longer timestamps
    transcript3 = """[01:00 -> 01:30] First minute
[01:30 -> 02:00] Second part
[02:00 -> 02:30] Third part"""

    chunks = split_transcript_with_time(transcript3, chunk_duration=60)
    print("\nTest 4 - Longer timestamps (60s chunks):")
    for i, (start, end, text) in enumerate(chunks):
        print(f"\nChunk {i+1} ({start} -> {end}):")
        print(text)

if __name__ == "__main__":
    test_basic_functionality()
