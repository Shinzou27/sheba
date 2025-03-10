import chromadb
import time
import random
import json
import ollama
import concurrent.futures

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="historias_orleans")

model_name = "llama3.2:3b" 
print(f"Modelo carregado: {model_name}")

def get_near_dialogues(id):
    ids = [str(int(id) + i - 4) for i in range(9)]
    results = collection.get(ids=ids)
    documents = results['documents']
    metadatas = results['metadatas']
    arrow_from_id = results["metadatas"][results["ids"].index(str(id))]['arrow']
    
    sorted_documents = []
    for i, doc in enumerate(documents):
        if metadatas[i]['arrow'] == arrow_from_id:
            char_name = f"{metadatas[i]['personagem']} |" if 'personagem' in metadatas[i] else ' '
            to_write = f"{char_name} {doc}"
            sorted_documents.append(to_write)
    
    sorted_documents = [doc for _, doc in sorted(zip(results['ids'], sorted_documents), key=lambda x: ids.index(x[0]))]
    return sorted_documents

def search_dialogues():
    results = collection.get()
    return results["ids"], results["documents"], results["metadatas"]

def generate_near_dialogues_prompt(near_dialogues: list):
    return "\n".join(near_dialogues)

def extract_context(text, near_dialogues):
    retries = 3
    for attempt in range(retries):
        try:
            prompt = f"""
Analise a emoção expressa pelo personagem neste texto e o contexto atual da história. O texto está no seguinte formato: "Nome do personagem | Texto".
Retorne, de forma clara, APENAS uma palavra para a emoção identificada e uma frase de até 100 caracteres que resume o que está acontecendo no texto alvo, NO FORMATO JSON, assim: 
{{"emotion": "felicidade", "summary": "aconteceu X e Y"}}

Alguns exemplos de emoções: 'felicidade', 'tristeza', 'raiva', etc.
Informações relevantes acerca dos dados:
Os textos cujo personagem é "1" ou "2" são opções de diálogo do próprio personagem do jogador.

Texto alvo: {text}

Contexto relevante com interações anteriores e posteriores:
{generate_near_dialogues_prompt(near_dialogues)}
                """
            
            response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
            content = response.message.content.strip()
            print(f"Resposta do modelo para documento: {content}")

            result = json.loads(content
                                .replace('"emoção"', '"emotion"')
                                .replace('"emotions"', '"emotion"')
                                .replace('"resumo"', '"summary"')
                                .replace('"resume"', '"summary"')
                                )
            
            return result["emotion"].strip(), result["summary"].strip()

        except Exception as e:
            print(f"Erro ao processar a requisição, tentativa {attempt + 1} de {retries}. Erro: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt + random.uniform(0, 1))
            else:
                raise

    return None, None

updated_count = 0
total_count = 0
def process_document(id, doc, metadata):
    global updated_count
    print(f"Processando documento {id}... ({updated_count} de {total_count})")
    emotion, summary = extract_context(doc, get_near_dialogues(id))
    if emotion and summary:
        collection.update(
            ids=[id],
            metadatas=[{**metadata, "emotion": emotion, "summary": summary}]
        )
        updated_count += 1
        print(f"Metadados atualizados para o documento {id}")

def update_metadata():
    global updated_count
    global total_count
    updated_count = 0
    print("Iniciando atualização de metadados...")
    ids, documentos, metadatas = search_dialogues()
    print(f"Encontrados {len(ids)} documentos para atualizar.")
    total_count = len(ids)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_document, ids, documentos, metadatas)

    print(f"Metadados de emoção atualizados para {updated_count} documentos!")

print("Iniciando processo...")
update_metadata()
print("Processo finalizado.")
