import chromadb
import time
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

chroma_client = chromadb.PersistentClient(path="./chroma_db") 

collection = chroma_client.get_or_create_collection(name="historias")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
print(model.device)

def get_near_dialogues(id):
    ids = [str(int(id) + i - 4) for i in range(9)]
    results = collection.get(ids=ids)
    documents = results['documents']
    sorted_documents = [doc for _, doc in sorted(zip(results['ids'], documents), key=lambda x: ids.index(x[0]))]
    return sorted_documents

def search_dialogues():
    results = collection.get()
    return results["ids"], results["documents"], results["metadatas"]

def extract_context(text, near_dialogues):
    retries = 3
    for attempt in range(retries):
        try:
            prompt = (
                "Analise a emoção expressa neste texto e o contexto atual da história. "
                "Retorne de forma clara a emoção identificada e uma frase resumindo o que está acontecendo, "
                "no formato exato: 'Emoção: [emoção] | Resumo: [resumo]'.\n\n"
                "Exemplo de emoções: 'felicidade', 'tristeza', 'raiva', 'medo', 'confusão', etc.\n\n"
                f"Texto: {text}\n"
                "Contexto relevante e interações anteriores e posteriores: \n" + "\n".join(near_dialogues))
            print(f"Prompt gerado: {prompt}...")
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                inputs["input_ids"], 
                max_new_tokens=150, 
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            content = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            print(f"Resposta do modelo: {content[:100]}...")

            emotion, summary = content.split(" | ")
            emotion = emotion.replace("Emoção: ", "").strip()
            summary = summary.replace("Resumo: ", "").strip()

            return emotion, summary

        except Exception as e:
            print(f"Erro ao processar a requisição, tentativa {attempt + 1} de {retries}. Erro: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt + random.uniform(0, 1))
            else:
                raise

    return None, None

def update_metadata():
    print("Iniciando atualização de metadados...")
    ids, documentos, metadatas = search_dialogues()
    print(f"Encontrados {len(ids)} documentos para atualizar.")
    for i, doc in enumerate(documentos):
        if i < 30:
            print(f"Processando documento {ids[i]}...")
            emotion, summary = extract_context(doc, get_near_dialogues(ids[i]))
            if emotion and summary:
                collection.update(
                    ids=[ids[i]],
                    metadatas=[{**metadatas[i], "contexto": {"emoção": emotion, "resumo": summary}}]
                )
                print(f"Metadados de emoção atualizados para o documento {ids[i]}")
            time.sleep(1)

    print("Metadados de emoção atualizados para todos os documentos!")

print("E")
# update_metadata()
print("F")
