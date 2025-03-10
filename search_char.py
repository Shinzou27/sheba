import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")
char_collection = chroma_client.get_collection(name="personagens_orleans")

def inspect_metadata():
    char_data = char_collection.get()
    if "metadatas" in char_data and char_data["metadatas"]:
        print("Estrutura dos metadados:", char_data["metadatas"][0])
    else:
        print("Nenhum metadado encontrado.")

inspect_metadata()


def get_character_by_name(name):
    char_data = char_collection.get(where={"name": name})
    if char_data["documents"]:
        return char_data["metadatas"][0]
    return None
  
items = char_collection.count()
print(items)