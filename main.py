from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from constants import *




embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
document_vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX, embedding=embeddings)
retreiver = document_vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0.7)
template = PromptTemplate(template="{query} Context: {context}", input_variables=["query", "context"])


prompt = ""

while(prompt != "exit"):
    prompt = input("PROMPT (type 'exit' to close the app)> ")
    if prompt == "exit":
        print("Thank you for using the Rag App!\n\n")
        break
    context = retreiver.get_relevant_documents(prompt)
    prompt_with_context = template.invoke({"query": prompt, "context": context})
    results = llm.invoke(prompt_with_context)
    print(results.content)