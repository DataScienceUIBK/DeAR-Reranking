import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM  # pip install peft

adapter_repo = "abdoelsayed/dear-8b-reranker-listwise-lora-v1"  # adapter repo
# If the adapter has "base_model_name_or_path" in its config, PEFT will auto-fetch the base model.

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

tokenizer = AutoTokenizer.from_pretrained(adapter_repo, use_fast=True, trust_remote_code=True)

# Important: do NOT also call model.to("cuda") when using device_map
model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_repo,
    torch_dtype=dtype,
    device_map={"": "cuda:1"},           # or just device_map="cuda"
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# If you still hit OOM, try 4-bit:
# model = AutoPeftModelForCausalLM.from_pretrained(
#     adapter_repo,
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=dtype,
#     device_map="auto",
#     trust_remote_code=True
# )

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Build a plain prompt if no chat_template is available:
def post_prompt(query: str, num: int) -> str:
    return (
        f"Search Query: {query}.\n"
        f"Rank the {num} passages above based on their relevance to the search query.\n"
        "The passages should be listed in descending order using identifiers.\n"
        "Please follow the steps below:\n"
        "Step 1. List the information requirements to answer the query.\n"
        "Step 2. For each requirement, find the passages containing that information.\n"
        "Step 3. Rank passages that best cover clear and diverse information. Include all passages.\n"
        "Output format strictly: [2] > [1] > [3]"
    )

SYSTEM_PROMPT = "You are RankLLM, an assistant that ranks passages by relevance to the query."
query = "You are RankLLM, an assistant that ranks passages by relevance to the query."

messages = [
    ("system", SYSTEM_PROMPT),
    ("user", f"I will provide you with 5 passages, each indicated by number identifier [].\n"
             f"Rank the passages based on their relevance to query: {query}."),
    ("assistant", "Okay, please provide the passages."),
    ("user", "[1] Lightning strike at Seoul National University."),
    ("assistant", "Received passage [1]."),
    ("user", "[2] Thomas Edison tried to invent a device for car but failed"),
    ("assistant", "Received passage [2]."),
    ("user", "[3] Coffee is good for diet"),
    ("assistant", "Received passage [3]."),
    ("user", "[4] KEPCO fixes light problems"),
    ("assistant", "Received passage [4]."),
    ("user", "[5] Thomas Edison invented the light bulb in 1879."),
    ("assistant", "Received passage [5]."),
    ("user", post_prompt(query, 5))
]

# If the tokenizer has a chat_template you can use it; otherwise, just concatenate:
if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
    prompt = tokenizer.apply_chat_template(
        [{"role": r, "content": c} for r, c in messages],
        add_generation_prompt=True, tokenize=False
    )
else:
    prompt = "\n".join([f"<|{r}|>\n{c}" for r, c in messages]) + "\n<|assistant|>\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

out = model.generate(
    **inputs,
    max_new_tokens=256,       # keep modest to save VRAM via shorter KV cache
    do_sample=True,
    temperature=0.5,
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
gen_ids = out[0][inputs.input_ids.shape[1]:]
print(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
