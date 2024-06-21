<div>
<div align='center'>
<h2 align="center"> Text Summarization through Images </h2>
<h3 align="center"></h3>
</div>
<div>
<div align="center">
<div style="text-align:center">
<img src="image (2).png"  width="50%" height="50%">
</div>
<div align="center">
<div style="text-align:center">
<img src="image (3).png"  width="50%" height="50%">
</div>
<div align="center">
<div style="text-align:center">
<img src="image (4).png"  width="50%" height="50%">
</div>
<div align="center">
<div style="text-align:center">
<img src="image (5).png"  width="50%" height="50%">
</div>

## Overview
This pipeline is designed to transform textual input into images using a combination of state-of-the-art models from natural language processing (NLP) and image generation domains. It leverages various pre-trained models to achieve this goal, including T5 for summarization, GPT-2 for prompt generation, Stable Diffusion for image creation, and SentenceTransformers for computing embeddings and similarities.

## Pipeline Workflow
#### Text Summarization: Uses T5 to condense the input text into a concise summary.
#### Prompt Generation: Employs GPT-2 to create descriptive prompts based on the summarized text.
#### Image Generation: Utilizes Stable Diffusion to generate images from the prompts.
####  Similarity Computation: Calculates the similarity between the original text and generated prompts using SentenceTransformers.
