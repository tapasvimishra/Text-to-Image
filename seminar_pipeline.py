from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

def summarize_text(text):
    model_name = 't5-small'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs.to(device)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_image_prompts(summary):
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    input_ids = tokenizer.encode(summary, return_tensors="pt")
    prompt_ids = model.generate(
        input_ids, 
        max_length=50, 
        num_beams=5,  # Use beam search
        num_return_sequences=5, 
        no_repeat_ngram_size=2
    )
    
    prompts = [tokenizer.decode(g, skip_special_tokens=True) for g in prompt_ids]
    return prompts
from diffusers import StableDiffusionPipeline

def generate_images(prompts):
    model_name = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

    images = []
    for prompt in prompts:
        image = pipe(prompt).images[0]
        images.append(image)
        display(image)
    
    return images
from sentence_transformers import SentenceTransformer, util

def compute_embeddings(text, prompts):
    model = SentenceTransformer('all-mpnet-base-v2')

    text_embedding = model.encode(text, convert_to_tensor=True)
    prompt_embeddings = model.encode(prompts, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(text_embedding, prompt_embeddings)
    return similarities
def full_pipeline(input_text):
    summary = summarize_text(input_text)
    print("Summary:", summary)
    
    prompts = generate_image_prompts(summary)
    print("Generated Prompts:", prompts)
    
    images = generate_images(prompts)
    images[0].show()
    
    similarities = compute_embeddings(summary, prompts)
    print("Similarities:", similarities)
    
    return summary, prompts, images, similarities
input_text = "Munich is a modern developed capital city in germany in the free state of Bavaria"
summary, prompts, images, similarities = full_pipeline(input_text)
