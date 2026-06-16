# Dataset

- Link: https://www.kaggle.com/datasets/tunguz/bank-marketing-data-set
- Versão: Version 1
- Licença: CC0: Public Domain

## Limitações

N/A.

## Instruções de Download

Após configurar corretamente o arquivo `.env`, execute o script abaixo para baixar os arquivos disponíveis no link acima.

Os arquivos serão salvos no caminho `data/kaggle/raw`.

```python
import os
import dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

dotenv.load_dotenv()

os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

api = KaggleApi()
api.authenticate()

dataset = os.getenv("DATASET")
path = f"../data/kaggle/raw/{dataset.split('/')[1]}/"
api.dataset_download_files(dataset, path=path, unzip=True)
```
