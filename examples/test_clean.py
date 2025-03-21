from section_parallel_clean import clean_think_tags, summarize_all_chunks_parallel

def test_clean_think_tags():
    # Test basic cleaning
    test_text = "Hello <think>this should be removed</think> World"
    cleaned = clean_think_tags(test_text)
    print("Basic cleaning test:")
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaned}")
    print()

    # Test nested tags
    test_text = "Start <think>outer <think>inner</think> text</think> End"
    cleaned = clean_think_tags(test_text)
    print("Nested tags test:")
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaned}")
    print()

    # Test multiple tags
    test_text = "<think>First</think> Middle <think>Last</think>"
    cleaned = clean_think_tags(test_text)
    print("Multiple tags test:")
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaned}")

if __name__ == "__main__":
    test_clean_think_tags()
