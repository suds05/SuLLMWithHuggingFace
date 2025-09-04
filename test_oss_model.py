#!/usr/bin/env python3
"""
Test script for loading and testing OSS models using transformers.
This script loads the openai/gpt-oss-20b model and runs basic functionality tests.
"""
from transformers import pipeline
import torch

def main():
    """Main test function."""
    print("Starting OSS Model Tests")

    model_id = "openai/gpt-oss-20b"

    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    messages = [
        {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
    ]

    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1])

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
