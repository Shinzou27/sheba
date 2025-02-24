import asyncio
import chromadb

async def query_test():
  client = await chromadb.AsyncHttpClient(host='localhost', port=8000)
  collection = await client.get_or_create_collection(name="my_collection")
  results = await collection.query(
      query_texts=["This is a query document about florida"],
      n_results=2
  )
  print(results)

asyncio.run(query_test())
