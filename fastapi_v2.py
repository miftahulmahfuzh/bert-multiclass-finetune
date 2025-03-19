from fastapi import FastAPI, HTTPException, Depends, Header, Query
from tuntun_chatbot_v2 import rag_chain
import uvicorn

app = FastAPI()

API_KEY = "ac7c07ad4851146d36ba0af67ad8bfb5f945c694f122a0babb14ff2632b60196"

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.get("/chat", dependencies=[Depends(verify_api_key)])
def chat(query: str = Query(..., title="User Query")):
    final_answer = rag_chain(query, stream=False)
    return final_answer

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
