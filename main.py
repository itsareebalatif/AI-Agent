from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(model="llama3-70b-8192") 
response=model.invoke("who is founder of pakistan?")
print(response)