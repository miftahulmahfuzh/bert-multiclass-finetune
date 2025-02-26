from core.rag import compression_retriever_reordered


def combine_docs(docs):
    return "\n\n".join(doc.metadata['Answer'] for doc in docs)


def faq_rag(question):
    return combine_docs(compression_retriever_reordered.invoke(question))
