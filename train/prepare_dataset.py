#!/usr/bin/env python3
import os
import json
import torch
from PIL import Image
from transformers import pipeline
from tqdm import tqdm
import argparse

class ShortCaptionGenerator:
    def __init__(self, model_type="florence", device="cuda"):
        self.device = device
        self.model_type = model_type
        
        if model_type == "florence":
            self.pipe = pipeline(
                "image-to-text",
                model="microsoft/florence-2-base",
                device=device
            )
        else:  # LLaVA
            self.pipe = pipeline(
                "image-to-text",
                model="llava-hf/llava-1.5-7b-hf",
                device=device
            )
    
    def generate_short_caption(self, image_path):
        image = Image.open(image_path)
        
        if self.model_type == "florence":
            result = self.pipe(image, max_new_tokens=15)
            caption = result[0]['generated_text']
        else:  # LLaVA
            prompt = "Briefly describe this image in 5-7 words:"
            result = self.pipe(image, prompt=prompt, max_new_tokens=12)
            caption = result[0]['generated_text'].split(':')[-1].strip()
        
        # Clean up and shorten
        caption = caption.split('.')[0]  # Take first sentence
        caption = ' '.join(caption.split()[:7])  # Max 7 words
        return f"Ghibli style: {caption}"

def process_dataset(input_dir, output_dir, output_jsonl, caption_model="florence"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = ShortCaptionGenerator(model_type=caption_model, device=device)
    
    input_files = sorted(os.listdir(input_dir))
    output_files = sorted(os.listdir(output_dir))
    
    with open(output_jsonl, 'w') as f_out:
        for in_file, out_file in tqdm(zip(input_files, output_files), desc="Processing"):
            source_path = os.path.join(input_dir, in_file)
            target_path = os.path.join(output_dir, out_file)
            
            try:
                caption = generator.generate_short_caption(source_path)
                f_out.write(json.dumps({
                    "source": source_path,
                    "caption": caption,
                    "target": target_path
                }) + '\n')
            except Exception as e:
                print(f"Skipped {in_file}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_jsonl", default="dataset.jsonl")
    parser.add_argument("--caption_model", choices=["florence", "llava"], default="florence")
    
    args = parser.parse_args()
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_jsonl=args.output_jsonl,
        caption_model=args.caption_model
    )