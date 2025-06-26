from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

def build_qa_chain():
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an intelligent assistant. Use the following context to answer the question at the end.

        Context:
        {context}

        Question:
        {question}

        Answer:"""
    )
    llm = ChatOpenAI(model_name="gpt-4o-mini")
    return LLMChain(llm=llm, prompt=prompt_template)

def rag(query: str, vecdb, qa_chain, k: int = 5):
    retrieved_docs = vecdb.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    result = qa_chain.run(context=context, question=query)
    return {
        "answer": result,
        "source_documents": retrieved_docs
    }
