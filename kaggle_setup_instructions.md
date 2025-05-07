# Instruções para Configurar Credenciais do Kaggle

Para baixar os dados do Kaggle através da API, siga estas etapas:

## 1. Crie uma conta no Kaggle
Se você ainda não tem uma conta, crie uma em [kaggle.com](https://www.kaggle.com/).

## 2. Gere sua API Token
1. Faça login na sua conta do Kaggle
2. Vá para sua conta: clique na sua foto de perfil no canto superior direito e selecione "Account"
3. Role para baixo até a seção "API"
4. Clique em "Create New API Token"
5. Isso baixará um arquivo chamado `kaggle.json` com suas credenciais

## 3. Configure as credenciais
1. Se o diretório `~/.kaggle` não existir, crie-o:
   ```bash
   mkdir -p ~/.kaggle
   ```

2. Mova o arquivo baixado para este diretório:
   ```bash
   mv ~/Downloads/kaggle.json ~/.kaggle/
   ```

3. Defina as permissões corretas (para garantir que apenas você possa ler o arquivo com suas credenciais):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

## 4. Teste as credenciais
Você pode verificar se as credenciais estão configuradas corretamente executando:
```bash
kaggle competitions list
```

## 5. Continue com o pipeline
Uma vez configuradas as credenciais, você pode continuar com o download dos dados:
```bash
python3 src/data/load_data.py
```

Ou executar todo o pipeline:
```bash
bash run_pipeline.sh
```

## Estrutura do arquivo kaggle.json
O arquivo `kaggle.json` deve ter a seguinte estrutura:
```json
{
  "username": "seu_username_do_kaggle",
  "key": "sua_key_de_api"
}
```

## Nota importante
Nunca compartilhe suas credenciais do Kaggle publicamente ou adicione-as ao controle de versão! 