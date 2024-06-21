# Text to Image Generation Pipeline

## Overview
This pipeline is designed to transform textual input into images using a combination of state-of-the-art models from natural language processing (NLP) and image generation domains. It leverages various pre-trained models to achieve this goal, including T5 for summarization, GPT-2 for prompt generation, Stable Diffusion for image creation, and SentenceTransformers for computing embeddings and similarities.

## Pipeline Workflow
#### Text Summarization: Uses T5 to condense the input text into a concise summary.
#### Prompt Generation: Employs GPT-2 to create descriptive prompts based on the summarized text.
#### Image Generation: Utilizes Stable Diffusion to generate images from the prompts.
####  Similarity Computation: Calculates the similarity between the original text and generated prompts using SentenceTransformers.
