# Postman Documentation
# https://documenter.getpostman.com/view/4281680/2sB2cbcKXd

from fastapi import FastAPI, HTTPException, Depends, Header, Query
from tuntun_chatbot_v2 import rag_chain, db
from config import settings
import uvicorn

app = FastAPI()

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.API_KEY.get_secret_value():
        raise HTTPException(status_code=401, detail="Invalid API Key")

@app.get("/chat", dependencies=[Depends(verify_api_key)])
def chat(query: str = Query(..., title="User Query"), user_id: str = 101):
    final_answer = rag_chain(query, user_id, stream=False)
    return final_answer

@app.get("/update_channel", dependencies=[Depends(verify_api_key)])
def update_channel(user_id: str, channel: str):
    result = {"status":"failed", "message":"connection to database not established"}
    if db.connect():
        result = db.update_channel(user_id, channel)
    return result

@app.get("/update_reaction", dependencies=[Depends(verify_api_key)])
def update_reaction(query_id: str, reaction: str):
    result = {"status":"failed", "message":"connection to database not established"}
    if db.connect():
        result = db.update_reaction(query_id, reaction)
    return result

@app.get("/get_chat_history", dependencies=[Depends(verify_api_key)])
def get_chat_history(user_id: str):
    result = {"status":"failed", "message":"connection to database not established"}
    if db.connect():
        result = db.get_chat_history(user_id)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
