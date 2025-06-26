from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai

from openai import OpenAI

client = OpenAI()


def chunk_text(text, chunk_size=500, chunk_overlap=50, docs = False) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    if docs:
        chunks = splitter.split_documents(text)
    else:
        chunks = splitter.split_text(text)
    return chunks

def table_chunking(table_content, document_context):
    prompt = f"""
      You are a table restoration and summarization assistant.

      You will be given:
      - An HTML table chunk and
      - Its context from the original document.

      Your task:
      1. Restore missing column headers using the document context.
      2. Correct any corrupted values from the context.
      3. Return the table in markdown format.
      4. Describe the table briefly and assign it a semantic title.
      5. If the column headers are split across multiple rows, flatten them into single-line column names
        For example, if the table has headers like:

      |                     | Three Months Ended June 30 | Six Months Ended June 30 |
      |---------------------|----------------------------|--------------------------|
      |                     | 2024 |   2023              | 2024 | 2023              |

      Then flatten it into:

      | Three Months Ended June 30 2024 | Three Months Ended June 30 2023 | Six Months Ended June 30 2024 | Six Months Ended June 30 2023 |


      Original Document Context:
      {document_context}

      HTML Table Chunk:
      {table_content}

      Return your answer strictly in the following format:

      TableDescription: <Brief natural language description of what the table contains>
      TableHeader: <Semantic title for the table, e.g., "Quarterly Financial Summary">
      TableMarkdown: <Markdown formatted table>

      Answer:
      """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that describes tables and formats them in markdown."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

