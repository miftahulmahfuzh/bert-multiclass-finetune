from fastapi import FastAPI, HTTPException, Depends, Header, Query
from datetime import datetime
from core.model import llm_ollama
from core.prompt import prompt, doc_prompt
from core.rag import vectorstore_none
from tool.tool_caller import process_tools
from langchain.chains.retrieval_qa.base import RetrievalQA

app = FastAPI()

API_KEY = "ac7c07ad4851146d36ba0af67ad8bfb5f945c694f122a0babb14ff2632b60196"

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

def combine_docs(docs):
    return "\n\n".join(doc.metadata['Answer'] for doc in docs)

def rag_chain(question: str) -> str:
    processed_prompt = prompt.partial(tools_output=process_tools(question))
    qa = RetrievalQA.from_chain_type(
        llm=llm_ollama,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": processed_prompt,
            "document_prompt": doc_prompt,
            "verbose": True
        },
        retriever=vectorstore_none.as_retriever(),
        verbose=True
    )
    print("Timestamp:", datetime.today())
    return qa.invoke(question, return_only_outputs='result').get("result", "No response")

@app.get("/chat", dependencies=[Depends(verify_api_key)])
def chat(query: str = Query(..., title="User Query")):
    final_answer = rag_chain(query)
    return final_answer


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
