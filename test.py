# import torch
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n achou gpu")
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")
client.delete_collection("personagens_orleans")
client.create_collection("personagens_orleans")