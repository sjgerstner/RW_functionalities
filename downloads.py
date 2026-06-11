from huggingface_hub import snapshot_download
from lists import MODEL_LIST, VANILLA_MODELS

for model_name in MODEL_LIST+VANILLA_MODELS:
    print(f"downloading {model_name}...")
    try:
        snapshot_download(model_name)
    except Exception as e:
        print(f'downloading {model_name} failed')
