from fastapi import FastAPI, HTTPException, Depends, Header, Query
from tuntun_chatbot_v2 import rag_chain
import uvicorn
from config import settings

app = FastAPI()

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.API_KEY.get_secret_value():
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.get("/chat", dependencies=[Depends(verify_api_key)])
def chat(query: str = Query(..., title="User Query")):
    final_answer = rag_chain(query, stream=False)
    return final_answer

if __name__ == "__main__":
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
