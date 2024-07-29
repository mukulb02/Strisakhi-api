from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load the Tiny LLaMA model and tokenizer
model_name = "PY007/TinyLlama-1.1B-Chat-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.get("/predict")
async def predict(query: str):
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=50)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return JSONResponse(content={"response": response_text})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
