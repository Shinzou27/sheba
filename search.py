import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="historias")

def search_dialogue_in_db(consulta, top_k=5):
    resultados = collection.query(
        query_texts=[consulta],
        n_results=top_k
    )
    return resultados
def search_char_dialogues(personagem):
    results = collection.query(
        query_texts=[""],
        n_results=30,
        where={"personagem": personagem}
    )
    return results['documents']

def search_context(arrow_id):
    contexto = collection.query(
        query_texts=[""],
        n_results=5,
        where={"arrow": arrow_id}
    )
    return contexto

target_personagem = "Mash"
target_arrow = 200020510

# print("Falas do personagem:", search_char_dialogues(target_personagem))
# print("Contexto da fala:", search_context(target_arrow))

def get_near_dialogues(id):
  ids = [str(int(id) + i) for i in range(15)]
  results = collection.get(ids=ids)
  documents = results['documents']
  sorted_documents = [doc for _, doc in sorted(zip(results['ids'], documents), key=lambda x: ids.index(x[0]))]
  return sorted_documents

for i in range(20):
  print(get_near_dialogues(i+4584))