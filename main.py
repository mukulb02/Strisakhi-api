from fastapi import FastAPI
from fastapi.responses import JSONResponse
from transformers import pipeline

app = FastAPI()

# Initialize the pipeline
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v0.1")

@app.get("/")
async def root():
    return {"message": "Welcome to the Text Generation API! Use the /predict endpoint to get responses."}

@app.get("/predict")
async def predict(query: str):
    # Generate text using the pipeline
    results = pipe(query, max_new_tokens=50)
    response_text = results[0]['generated_text']
    return JSONResponse(content={"response": response_text})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
