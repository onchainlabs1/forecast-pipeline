# Download Manual dos Dados

Como estamos tendo problemas com as credenciais da API do Kaggle, vamos baixar os dados manualmente:

1. Acesse a página da competição no Kaggle: [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

2. Faça login na sua conta do Kaggle

3. Clique no botão "Download All" para baixar todos os arquivos

4. Extraia o arquivo ZIP baixado

5. Mova os seguintes arquivos para o diretório `/Users/fabio/Desktop/mlproject/data/raw/`:
   - train.csv
   - test.csv
   - holidays_events.csv
   - oil.csv
   - stores.csv
   - transactions.csv

## Após o download manual

Uma vez que os arquivos estejam no diretório correto, podemos continuar com o pipeline:

```bash
# Verifique se os arquivos estão no local correto
ls -la data/raw/

# Execute o pré-processamento dos dados
python3 src/data/preprocess.py

# Treine o modelo
python3 src/train_model.py

# Ou execute todo o pipeline (pule a etapa de download)
bash run_pipeline.sh
``` 