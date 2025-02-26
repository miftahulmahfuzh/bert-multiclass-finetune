from langchain_core.prompts import PromptTemplate

# template = """<|begin_of_text|> <|start_header_id|>system<|end_header_id|> You are a virtual assistant for the Tuntun investment app. Based on the given JSON context or tools output, provide a comprehensive response that accurately answers the query. If the required information is not available, respond with 'I don't know' without referencing any sources or tools used. Ensure your response is complete, clear, and helpful.
# <|eot_id|> <|start_header_id|>context<|end_header_id|> {context} <|eot_id|>
# <|start_header_id|>tools_output<|end_header_id|> {tools_output} <|eot_id|>
# <|start_header_id|>user<|end_header_id|> {question} <|eot_id|> <|start_header_id|>assistant<|end_header_id|>"""
template = """<|begin_of_text|> <|start_header_id|>system<|end_header_id|>You are a virtual assistant for the Tuntun investment app. Based on the given JSON context or tools output, provide a comprehensive response that accurately answers the query. If the required information is not available, respond with 'I don't know' without referencing any sources or tools used. Ensure your response is complete, clear, and helpful.
<|eot_id|> <|start_header_id|>context<|end_header_id|> {tools_output} 
{context} <|eot_id|>
<|start_header_id|>user<|end_header_id|> {question} <|eot_id|> <|start_header_id|>assistant<|end_header_id|>"""
# template = """<|begin_of_text|> <|start_header_id|>system<|end_header_id|> You are an assistant for the investment app Tuntun. You should select related context or tool and give a concise and easy-to-understand answer response with full data. Your response should in the same language with user’s question. If you don't know the answer, simply say 'I don't know.' Avoid mentioning sources or tools used.
# <|eot_id|> <|start_header_id|>context<|end_header_id|> {context} <|eot_id|>
# <|start_header_id|>tools<|end_header_id|> {tools_output} <|eot_id|>
# <|start_header_id|>user<|end_header_id|> {question} <|eot_id|> <|start_header_id|>assistant<|end_header_id|>"""
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)
doc_template = "{Answer}"
doc_prompt = PromptTemplate(
    input_variables=["Answer"],
    template=doc_template
)
