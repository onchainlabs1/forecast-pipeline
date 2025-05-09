# Sistema de Previsão de Vendas para Lojas

Este projeto implementa um sistema completo de previsão de vendas para uma rede de lojas, oferecendo um dashboard interativo para visualização de dados históricos, geração de previsões futuras e análise de métricas de desempenho do modelo.

## Funcionalidades

- **Dashboard Interativo**: Interface amigável para explorar dados de vendas
- **Previsão de Vendas**: Previsões por loja e família de produtos
- **Análise de Desempenho**: Verificação transparente da precisão das previsões
- **Insights do Modelo**: Visualização da importância de features e explicabilidade
- **Monitoramento**: Detecção de drift de modelo e alertas de qualidade de dados

## Métricas de Desempenho

- **Precisão da Previsão**: 80.17% (calculada como 100 - MAPE)
- **MAPE**: 19.83% (Mean Absolute Percentage Error)
- **MAE**: 46.16 (Mean Absolute Error)
- **RMSE**: 50.64 (Root Mean Square Error)

## Estrutura do Projeto

```
mlproject/
├── src/
│   ├── api/              # API REST para servir previsões
│   ├── dashboard/        # Interface Streamlit
│   ├── database/         # Modelos e conexão com o banco de dados
│   ├── features/         # Geração de features para o modelo
│   ├── models/           # Modelos de machine learning
│   ├── security/         # Autenticação e segurança
│   └── utils/            # Funções utilitárias
├── models/               # Modelos treinados
├── tests/                # Testes automatizados
├── run_dashboard.sh      # Script para executar o dashboard
└── run_app_with_db.sh    # Script para executar API e banco de dados
```

## Instalação

```bash
# Clonar o repositório
git clone https://github.com/seu-usuario/mlproject.git
cd mlproject

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

## Executando o Sistema

1. Inicie a API:
```bash
python -m src.api.main
```

2. Em outro terminal, inicie o dashboard:
```bash
streamlit run src/dashboard/app.py
```

3. Acesse o dashboard em seu navegador: http://localhost:8501

## Credenciais de Demonstração

- **Usuário**: johndoe
- **Senha**: secret

## Problemas Conhecidos

- Erro na verificação de tipos no método `isinstance(date, datetime.date)`
- Incompatibilidade no número de features (esperado 81, gerado 119)
- Erro ao criar explainer SHAP devido à incompatibilidade de dimensões
- Erro ao salvar previsões no banco de dados (parâmetro 'date' inválido)

## Próximos Passos

- Corrigir bugs de verificação de tipos
- Resolver incompatibilidade no número de features
- Melhorar a explicabilidade do modelo
- Implementar mais testes automatizados