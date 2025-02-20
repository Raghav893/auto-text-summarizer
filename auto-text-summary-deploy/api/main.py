from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import pipeline
import torch
from fastapi import Request

# Initialize FastAPI app
app = FastAPI(title="Text Summarizer")

# Mount templates
templates = Jinja2Templates(directory="templates")

# Initialize the summarization pipeline
try:
    print("Loading model...")
    summarizer = pipeline(
        "summarization",
        model="t5-small",
        device=0 if torch.cuda.is_available() else -1
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    summarizer = None

class SummaryRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/summarize")
async def summarize(request: SummaryRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Please enter some text.")
        
    if len(request.text) > 10000:
        raise HTTPException(status_code=400, detail="Text is too long. Please limit to 10,000 characters.")
        
    if summarizer is None:
        raise HTTPException(status_code=500, detail="Model not properly loaded.")
        
    try:
        summary = summarizer(
            request.text.strip(), 
            max_length=130, 
            min_length=30, 
            do_sample=False
        )
        
        return {
            "summary_text": summary[0]['summary_text'],
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)