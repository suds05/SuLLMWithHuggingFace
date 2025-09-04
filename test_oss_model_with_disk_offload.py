#!/usr/bin/env python3
"""
Test script for loading and testing OSS models using transformers with disk offloading.
This script loads the openai/gpt-oss-20b model with proper memory management using disk_offload.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import disk_offload
import torch
import os
import sys

def main():
    """Main test function."""
    print("Starting OSS Model Tests with Disk Offloading")
    
    model_id = "openai/gpt-oss-20b"
    
    # Create offload directory
    offload_dir = "./model_offload"
    os.makedirs(offload_dir, exist_ok=True)
    print(f"Created offload directory: {offload_dir}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load model with low CPU memory usage
    print("Loading model with low CPU memory usage...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu"  # Start on CPU
    )
    
    # Offload model to disk
    print("Offloading model to disk...")
    disk_offload(model=model, offload_dir=offload_dir)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        model = model.to(device)
    
    # Test the model
    messages = [
        {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
    ]
    
    # Format messages for the model
    formatted_input = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_input, return_tensors="pt").to(device)
    
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print("Generated response:", response)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)




