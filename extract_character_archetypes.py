import chromadb
import ollama
import json
import time


model_name = "llama3.2:3b" 
print(f"Modelo carregado: {model_name}")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="historias_orleans")
char_collection = chroma_client.get_collection(name="personagens_orleans")
def timer(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    elapsed_time = end - start

    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    formatted_time = f"{minutes} minutos e {seconds:.2f} segundos"

    return result, formatted_time
def filter_dict(d: dict, chaves_permitidas: list) -> dict:
    return {k: v for k, v in d.items() if k in chaves_permitidas}

def parse_content(content):
    data = json.loads(content)
    for character in data:
        if character.get('persona_updates'):
            for update in character['persona_updates']:
                updated = {}
                for aspecto, valor in update.items():
                    updated[aspecto.lower()] = valor 
                character['persona_updates'] = [updated]
    return data
def print_personality_analysis(data):
    if not isinstance(data, list):
        print("Erro: Os dados devem ser uma lista.")
        return
    
    for character in data:
        if not isinstance(character, dict):
            print("Erro: Esperado um dicionário para cada personagem.")
            return
        
        print(f"Personagem: {character.get('character_name', 'Nome não encontrado')}")
        
        if character.get('persona_updates'):
            for update in character['persona_updates']:
                for aspecto, valor in update.items():
                    print(f"  - Atualização: {aspecto} = {valor}")
        else:
            print("  - Nenhuma atualização de personalidade.")
        
        if character.get('summary'):
            print(f"  - Resumo: {character['summary']}")
        else:
            print("  - Nenhum resumo relevante.")
        
        print("-" * 40)

def normalize_char_name(name):
    return name.lower().replace(" ", "_")

def initialize_character_data():
    dialogs_data = collection.get()
    metadatas = dialogs_data["metadatas"]

    unique_characters = {}
    
    for metadata in metadatas:
        character_name = metadata["personagem"]
        
        if character_name in ["1", "2", "???"]:
            continue
        
        if character_name not in unique_characters:
            unique_characters[character_name] = (
                {
                "id": normalize_char_name(character_name),
                "name": character_name
                },
                {
                "extroversion": 0,
                "emotional_control": 0,
                "creativity": 0,
                "responsibility": 0,
                "kindness": 0,
                "courage": 0,
                "diligence": 0,
                "autonomy": 0,
                "summary": ""
                })

    char_collection.add(
        ids=[char[0]["id"] for char in unique_characters.values()],
        documents=[char[0]["name"] for char in unique_characters.values()],
        metadatas=[char[1] for char in list(unique_characters.values())]
    )

def get_char_metadata(character_name):
    return char_collection.get(ids=[normalize_char_name(character_name)])["metadatas"]

def get_char_info(name, character_metadata):
    if character_metadata and character_metadata[0]:
        character_metadata = character_metadata[0]
        toReturn = f"Nome: {name}\n\nCaracterísticas: "
        toReturn += ", ".join([f"{trait.capitalize()}: {value}" for trait, value in character_metadata.items() if trait != 'summary'])
        toReturn += f"\n\nResumo: {character_metadata.get('summary', 'Sem resumo disponível.')}"
        
        return toReturn
def generate_prompt(dialogs, metadatas):
    prompt = f"""
Você irá analisar um excerto sequencial de diálogos extraídos de uma história de jogo. Os diálogos estarão no formato 'Nome do personagem | texto'. Quando o personagem for "1" ou "2", os diálogos representam opções de resposta do personagem do jogador, e para diálogos com "???" (personagens não revelados), você deve ignorá-los completamente.

Sua tarefa é analisar o comportamento de cada personagem presente e compará-lo com as informações de personalidade fornecidas a seguir. Se você perceber que o comportamento de um personagem durante o diálogo indica uma mudança significativa em alguma característica de personalidade (por exemplo, se um personagem demonstra mais coragem ou criatividade), faça a alteração correspondente. Somente mudanças claras e significativas no comportamento ou nas falas do personagem devem gerar ajustes nas características. Não altere características sem uma razão sólida, e evite ajustes triviais ou baseados em falas menores.

Os ÚNICOS aspectos de personalidade a serem avaliados são: extroversion, emotional_control, creativity, responsibility, kindness, courage, diligence e autonomy.

Por exemplo, se um personagem mostra coragem ou resolve um problema sem hesitar, você pode aumentar o valor do aspecto "Courage" (Coragem). Caso o personagem seja excessivamente agressivo ou egoísta, você poderia ajustar a característica "Kindness" (Bondade) para um valor mais baixo. Um valor 10 é considerado mediano para determinado aspecto de personalidade. Valores mais altos indicam maior presença desse aspecto, enquanto valores mais baixos indicam ausência ou pouca presença desse aspecto na personalidade do personagem.

Nota importante: Se o comportamento de um personagem não for significativamente diferente do que já foi registrado nas características, não faça nenhum ajuste. Somente ajustes substanciais devem ser feitos, com base em atitudes ou falas que indicam mudanças reais no caráter do personagem.

Se houver necessidade de ajustar os valores de qualquer aspecto de personalidade, gere uma chave e um valor no formato especificado mais adiante, com a chave sendo o aspecto de personalidade em questão e o valor numérico a atualização estipulada. O número DEVE estar entre 1 e 20.

No campo 'summary', você deve adicionar apenas informações relevantes e novas sobre o personagem, com frases de no máximo 80 caracteres. Se não houver novas informações importantes, deixe o campo 'summary' vazio (não omita o campo).

Não adicione quaisquer explicações das mudanças. O output deve ser exclusivamente no formato abaixo, adicionando itens na lista caso haja mais de um personagem presente:
[
    {{
        "character_name": "Nome do personagem",
        "persona_updates": [
            {{"Chave": Número}}
        ],
        "summary": "Informação relevante sobre o personagem"
    }}
]

Os seguintes personagens aparecem nesse diálogo:
        """

    characters_in_batch = {meta["personagem"] for meta in metadatas}
    char_info = "\n".join([get_char_info(char, get_char_metadata(char)) for char in characters_in_batch if get_char_info(char, get_char_metadata(char)) is not None])
    prompt += char_info + "\n\nOs diálogos são os seguintes:\n"

    for dialog, metadata in zip(dialogs, metadatas):
        prompt += f"{metadata['personagem']} | {dialog}\n"
    return prompt

def process_dialogs_and_update_characters():
    dialogs_data = collection.get()
    dialogs = dialogs_data["documents"]
    metadatas = dialogs_data["metadatas"]
    
    batch_size = 30
    max_attempts = 3
    for i in range(0, len(dialogs), batch_size):
        print(f"Processo {1+int(i/batch_size)} de {len(dialogs)//batch_size}...")
        for attempt in range(max_attempts):
            try:
                dialogs_batch = dialogs[i:i + batch_size]
                metadatas_batch = metadatas[i:i + batch_size]
                prompt = generate_prompt(dialogs_batch, metadatas_batch)
                response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
                content = response.message.content.strip()
                content = parse_content(content)
                print_personality_analysis(content)
                for char in content:
                    update_character_in_db(char, normalize_char_name(char['character_name']))
                    pass
                break
            except Exception as e:
                print(f"Tentativa {attempt+1} de {max_attempts} sem sucesso.\nErro:\n{e}")
                if attempt <= max_attempts:
                    continue

def update_character_in_db(character_data, character_id):
    updates = {}
    parse = lambda original_list : {list(item.keys())[0]: list(item.values())[0] for item in original_list}
    if 'persona_updates' in character_data:
        updates = filter_dict(parse(character_data['persona_updates']), ['extroversion', 'emotional_control', 'creativity', 'responsibility', 'kindness', 'courage', 'diligence', 'autonomy', 'summary'])
    if 'summary' in character_data and character_data['summary'] != '':
        updates['summary'] = character_data['summary']
    if len(updates.keys()) > 0:
        char_collection.update(
            ids=[character_id],
            metadatas=[updates]
        )


print(timer(
    # initialize_character_data
    process_dialogs_and_update_characters
            )[1])