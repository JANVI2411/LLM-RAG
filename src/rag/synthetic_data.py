from typing import List
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from datasets import Dataset 
hf_token=""

class QAPair(BaseModel):
    question: str = Field(..., description="A question based on the context")
    answer: str = Field(..., description="The answer to the question")

class QAPairList(BaseModel):
    context: str = Field(..., description="The source context")
    context_id: str = Field(..., description="Unique ID for context")
    qa_pairs: List[QAPair] = Field(..., description="Generated question-answer pairs")

def get_qa_generator():
    prompt = PromptTemplate.from_template("""
        You are a QA generator.

        Context:
        {context}

        Generate 3 diverse and relevant question-answer pairs from the above context.
        Format:
        {format_instructions}
    """)

    parser = PydanticOutputParser(pydantic_object=QAPairList)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    qa_chain = prompt | llm | parser
    return qa_chain, parser

def generate_data(table_points):
    synthetic_data = []
    qa_chain,parser = get_qa_generator()
    for point in table_points:
        context_text = point.payload["text"]
        context_id = str(point.id)

        result = qa_chain.invoke({
            "context": context_text,
            "format_instructions": parser.get_format_instructions()
        })

        result.context = context_text
        result.context_id = context_id
        synthetic_data.append(result)

    all_qa = []
    for item in synthetic_data:
        for qa in item.qa_pairs:
            all_qa.append({
                "question": qa.question,
                "answer": qa.answer,
                "context": item.context,
                "context_id": item.context_id
            })


    hf_dataset = Dataset.from_list(all_qa)

    hf_dataset.push_to_hub("JC-24/meta-record-rag-synthetic-dataset", token=hf_token)

