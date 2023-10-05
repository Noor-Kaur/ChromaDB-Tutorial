import chromadb
from chromadb.config import Settings
#settings is used to configure the database 

#chromadb client is created by calling the Client constructor 
#client constructor is passed a Settings object as an argument
#'chroma_db_impl' is used to specify the type of database to be used
#'persist_directory' specifies the directory where the database will be stored
#client = chromadb.Client(Settings(chroma_db_impl='duckdb+parquet', persist_directory='/db'))
client = chromadb.PersistentClient(path="/db")

#Reseting the database so that all the consecutive commands can run smoothly
client.delete_collection(name='Students')
client.delete_collection(name='Students2')

#A collection object is crated using the client, it is similar to creating a table in a traditional database
collection = client.create_collection(name='Students')


#initializing text about student, club, and university
student_info = """
Alexandra Thompson, a 19-year-old computer science sophomore with a 3.7 GPA,
is a member of the programming and chess clubs who enjoys pizza, swimming, and hiking
in her free time in hopes of working at a tech company after graduating from the University of Washington.
"""

club_info = """
The university chess club provides an outlet for students to come together and enjoy playing
the classic strategy game of chess. Members of all skill levels are welcome, from beginners learning
the rules to experienced tournament players. The club typically meets a few times per week to play casual games,
participate in tournaments, analyze famous chess matches, and improve members' skills.
"""

university_info = """
The University of Washington, founded in 1861 in Seattle, is a public research university
with over 45,000 students across three campuses in Seattle, Tacoma, and Bothell.
As the flagship institution of the six public universities in Washington state,
UW encompasses over 500 buildings and 20 million square feet of space,
including one of the largest library systems in the world.
"""


#The data is added with metadata and unique IDs
#chromadb will automatically convert the text into embeddings and store it in the Studnets collection
#It uses the  'all-MiniLM-L6-v2' model to convert text into embeddings
collection.add(
    documents = [student_info, club_info, university_info],
    metadatas= [{'source': 'student_info'}, {'source': 'club_info'}, {'source': 'university_info'}],
    ids= ['id1', 'id2', 'id3']
)


#the query function is used to used to ask questions in natural language for a similarity search
#It will convert the query into embedding and use similarity search to come up with similar results
results = collection.query(
    query_texts=['What is the student name?'],
    n_results=2
)


# results variable consists of the 2 files that I closest to our expected output
print(results)


#we can choose the embedding function or even create our oen embedding function
#text documents are then added to create embeddings
from chromadb.utils import embedding_functions
import openai
openai.api_key = ('add_openai_api_key')
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                model_name="text-embedding-ada-002"
            )
students_embeddings = openai_ef([student_info, club_info, university_info])
print(students_embeddings)

#Now we will use the above mentioned embedding model in the new database
#We will now use the 'get_or_create_collection', as the name suggests it will either create a collection or get it if it already exists
#we used the embedding function to create the embeddings and then feed it to the collection
collection2 = client.get_or_create_collection(name='Students2')

collection2.add(
    embeddings= students_embeddings,
    documents= [student_info, club_info, university_info],
    metadatas= [{'source': 'student info'}, {'source': 'club info'}, {'source': 'university info'}],
    ids= ['id1', 'id2', 'id3']
)

#We can use an easier way wherein instead of feeding the embeddings we will specify the embedding function while creating of the collection
collection2 = client.get_or_create_collection(name='Students2', embedding_function=openai_ef)

collection2.add(
    documents = [student_info, club_info, university_info],
    metadatas= [{'source': 'student info'}, {'source': 'club info'}, {'source': 'university info'}],
    ids = ['id1', 'id2', 'id3']
)

results = collection2.query(
    query_texts=['What is the student name?'],
    n_results=2
)

print(results)

#Updating data
#To alter the text or the meta data we provide the id
collection2.update(
    ids=["id1"],
    documents=["Kristiane Carina, a 19-year-old computer science sophomore with a 3.7 GPA"],
    metadatas=[{"source": "student info"}],
)

results = collection2.query(
    query_texts=["What is the student name?"],
    n_results=2
)

print(results)

#Removing data
collection2.delete(ids = ['id1'])

results = collection2.query(
    query_texts=['What is the student name?'],
    n_results=2
)

print(results)

#Collection Management
vector_collections = client.create_collection("vectordb")


vector_collections.add(
    documents=["This is Chroma DB CheatSheet",
               "This is Chroma DB Documentation",
               "This document Chroma JS API Docs"],
    metadatas=[{"source": "Chroma Cheatsheet"},
    {"source": "Chroma Doc"},
    {'source':'JS API Doc'}],
    ids=["id1", "id2", "id3"]
)

#Count the records in the collection
vector_collections.count()

#View all the records in the collection
vector_collections.get()

#change the name of the collection
vector_collections.modify(name='chroma_info')

#list all the collections in the client
client.list_collections()

#Access a new collection
vector_collection_new = client.get_collection(name='chroma_info')

#Delete a collection
client.delete_collection(name='chroma_info')
client.list_collections()

#Delete the entire database/client
client.reset()
client.list_collections()



