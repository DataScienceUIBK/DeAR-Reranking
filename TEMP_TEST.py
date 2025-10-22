import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

os.environ["CUDA_VISIBLE_DEVICES"]='1'
# Load model
model_path = "abdoelsayed/dear-8b-reranker-ce-v1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, torch_dtype=torch.bfloat16)
model.eval().cuda()

# Score a query-document pair
query = "What is llama?"
doc = "The llama is a domesticated South American camelid..."
inputs = tokenizer(f"query: {query}", f"document: {doc}", return_tensors="pt", truncation=True, max_length=228)
inputs = {k: v.cuda() for k, v in inputs.items()}

with torch.no_grad():
    score = model(**inputs).logits.squeeze().item()
print(f"Relevance score: {score}")