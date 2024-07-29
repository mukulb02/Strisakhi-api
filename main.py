from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/predict")
async def predict(query: str):
    if query.lower() == 'hello':
        response = {"response": "Hi there!"}
    elif query.lower() == 'how are you':
        response = {"response": "I'm doing great, thank you!"}
        elif query.lower() == 'hii':
        response = {"response": "Namaste"}
    elif query.lower() == 'bye':
        response = {"response": "Goodbye!"}
    else:
        response = {"response": "I don't understand the query."}
    return JSONResponse(content=response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
