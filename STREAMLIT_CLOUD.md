# Instruções de Implantação no Streamlit Cloud

## Passo a Passo

1. Acesse o [Streamlit Cloud](https://streamlit.io/cloud)
2. Clique em "New app"
3. Configure o aplicativo com as seguintes informações:
   - **Repository:** seu repositório GitHub
   - **Branch:** main
   - **Main file path:** `streamlit_cloud_app.py` (IMPORTANTE: use este arquivo específico!)
   - **App URL:** escolha um nome para a URL
   
4. Em "Advanced Settings":
   - **Python version:** Use 3.9 (recomendado) ou 3.10
   - **Requirements file:** `streamlit_requirements.txt` (IMPORTANTE: use este arquivo específico!)
   - **Packages:** o arquivo packages.txt já está configurado

## Resolução de Problemas

Se ocorrer erro "ResolutionImpossible":

1. Verifique se está usando `streamlit_cloud_app.py` como arquivo principal
2. Verifique se está usando `streamlit_requirements.txt` como arquivo de requisitos
3. Reinicie o aplicativo após fazer alterações

## Arquivos Importantes

- `streamlit_cloud_app.py`: Ponto de entrada simplificado para o Streamlit Cloud
- `streamlit_requirements.txt`: Apenas as dependências estritamente necessárias
- `.streamlit/secrets.toml`: Configurações do ambiente
- `packages.txt`: Pacotes do sistema necessários

## API e Conexão

O aplicativo se conecta à API em:
```
https://forecast-pipeline-2.onrender.com
```

Se a API não estiver disponível, o dashboard mostrará informações de diagnóstico.

## Nota Importante

O arquivo de requisitos completo (`requirements.txt`) contém dependências conflitantes que o Streamlit Cloud não consegue resolver. Por isso, utilizamos um arquivo de requisitos simplificado (`streamlit_requirements.txt`) com apenas as dependências necessárias para o dashboard. 