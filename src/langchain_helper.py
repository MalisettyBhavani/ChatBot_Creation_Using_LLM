from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import os
from dotenv import load_dotenv

load_dotenv()


llm = GooglePalm(google_api_key = os.environ["GOOGLE_API_KEY"], temperature = 0)

## load csv file


## create Word Embeddings:
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path="codebasics_faqs.csv", source_column="prompt", encoding="latin1")
    data = loader.load()
    vector_db = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vector_db.save_local(vectordb_file_path)

# Load the vector database from the local folder
def get_qa_chain():

    vector_db = FAISS.load_local(vectordb_file_path,instructor_embeddings)
    # Create a retriever object for querying the vector database
    retriever = vector_db.as_retriever(score_threshold=0.7)
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer, try to provide as much text as possible from the "response" section in the source document context without making things up.
    If the answer is not found in the context, kindly state "I don't know". Don't try to make up an answer.

    CONTEXT: {context}
    QUESTION: {question}"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return chain

if __name__=="__main__":
    create_vector_db()
    # Load the QA chain
    chain = get_qa_chain()
    print(chain("Do you provide internships? Do you have an EMI option?"))



