import pandas as pd
import chromadb
from concurrent.futures import ThreadPoolExecutor

def load_data(tsv_path):
  return pd.read_csv(tsv_path, sep='\t')

def insert_batch(collection, batch):
  for index, row in batch.iterrows():
    print(f"Linha: {index}")
    collection.add(
      ids=[str(index)],
      documents=[row["Texto (Português)"]],
      metadatas=[{
        "personagem": row['Nome (Inglês)'],
        "arrow": row['Arrow'],
        "texto_ingles": row['Texto (Inglês)'],
        "texto_japones": row['Texto (Japonês)']
      }]
    )

def main(tsv_path):
  client = chromadb.PersistentClient(path="./chroma_db")
  collection = client.get_or_create_collection(name="historias")
  
  df = load_data(tsv_path)
  batch_size = len(df) // 4
  
  batches = [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]
  
  with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(lambda batch: insert_batch(collection, batch), batches)

if __name__ == "__main__":
    main("D:/Docs (HD)/Codiguins/python/chaldeas/revised/translations/agartha.tsv")
