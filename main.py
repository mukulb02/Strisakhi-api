from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load the DistilGPT-2 model and tokenizer
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model with appropriate device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

@app.get("/")
async def root():
    return {"message": "Welcome to the DistilGPT-2 API! Use the /predict endpoint to get responses."}

@app.get("/predict")
async def predict(query: str):
    inputs = tokenizer(query, return_tensors="pt").to(device)
    outputs = model.generate(inputs["input_ids"], max_new_tokens=50)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return JSONResponse(content={"response": response_text})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
