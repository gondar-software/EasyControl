import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, LlavaForConditionalGeneration
from tqdm import tqdm
import argparse

class CaptionGenerator:
    def __init__(self, model_type="florence", device="cuda"):
        self.device = device
        self.model_type = model_type
        
        if model_type == "florence":
            self.model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=torch.float16, trust_remote_code=True).to(self.device)
            self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                "llava-hf/llava-1.5-7b-hf", 
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True, 
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    
    def generate_short_caption(self, image_path, trigger_word):
        image = Image.open(image_path)
        prompt = "Briefly describe this image in 10~20 words:"
        
        if self.model_type == "florence":
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, torch.float16)
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=48,
                do_sample=False,
                num_beams=3,
            )
            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0].replace("</s>", "").replace("<s>", "").strip()
        else:
            conversation = [
                {

                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt}"},
                    {"type": "image"},
                    ],
                },
            ]
            prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(images=image, text=prompt_text, return_tensors='pt').to(0, torch.float16)
            generated_ids = self.model.generate(**inputs, max_new_tokens=48, do_sample=False)
            caption = self.processor.decode(generated_ids[0][2:], skip_special_tokens=True).split("ASSISTANT: ")[-1]
        
        return f"{trigger_word}, A digital illustration of {caption}"

def process_dataset(input_dir, output_dir, output_jsonl, trigger_word, caption_model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = CaptionGenerator(model_type=caption_model, device=device)
    
    input_files = sorted(os.listdir(input_dir))
    output_files = sorted(os.listdir(output_dir))
    
    with open(output_jsonl, 'w') as f_out:
        for in_file, out_file in tqdm(zip(input_files, output_files), desc="Processing"):
            source_path = os.path.join(input_dir, in_file)
            target_path = os.path.join(output_dir, out_file)
            
            try:
                caption = generator.generate_short_caption(source_path, trigger_word)
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
    parser.add_argument("--trigger_word", default="Ghibli Studio style")
    parser.add_argument("--caption_model", choices=["florence", "llava"], default="florence")
    
    args = parser.parse_args()
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_jsonl=args.output_jsonl,
        trigger_word=args.trigger_word,
        caption_model=args.caption_model
    )