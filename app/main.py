from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader,PyPDFLoader
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

try:
 loader=PyPDFLoader("docs/Freelancing.pdf")

except Exception as e:  
 print("Error while loading file=", e)


# embeddings
embedding= GoogleGenerativeAIEmbeddings(model="models/embedding-001")     

# splitting docs
text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)



store=VectorstoreIndexCreator(text_splitter=text_splitter,embedding=embedding)
index=store.from_loaders([loader])


while True:
    human_message = input("How i can help you today? ")
    response = index.query(human_message, llm=llm)
    print(response)

