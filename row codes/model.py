# file: model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class AIModel:
    def __init__(self):
        print("🚀 Loading Qwen model... (This will take 10-20 seconds)")
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        
        # Load Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,  
            device_map="auto"
        )
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print("✅ Model loaded successfully!")

    def generate_response(self, user_prompt: str):
        # System prompt to define personality
        messages = [
            {"role": "system", "content": "You are Qwen, a helpful API assistant."},
            {"role": "user", "content": user_prompt}
        ]
        
        # Prepare input
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Generate output
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        
        # Decode output
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]