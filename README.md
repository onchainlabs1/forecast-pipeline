# RetailPro AI: Sales Forecasting Platform

<div align="center">
  
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.21.2-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20.0-FF4B4B)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95.0-009688)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

<p align="center">
  <img src="reports/dashboard_screenshot.png" alt="RetailPro AI Dashboard" width="800"/>
</p>

## 🎯 Business Problem

Retail chains face significant challenges in inventory management, with both overstocking and understocking leading to substantial financial losses:

- **Overstocking** ties up capital and leads to markdowns and waste
- **Understocking** results in lost sales and reduced customer satisfaction
- **Seasonal variations** complicate manual forecasting approaches
- **Promotion planning** requires accurate sales predictions

RetailPro AI addresses these challenges by providing precise, store-level forecasts across all product families, enabling retailers to optimize inventory levels, plan promotions effectively, and maximize profitability.

## 💡 Solution

Our platform delivers an end-to-end forecasting solution that:

- **Predicts sales** with 80%+ accuracy for 54 stores across 34 product families
- **Visualizes trends** through an intuitive dashboard with customizable filters
- **Explains predictions** using explainable AI techniques to build trust
- **Monitors performance** with real-time drift detection and model metrics
- **Secures data** with robust JWT authentication and role-based access

## 🔍 Key Features

<table>
  <tr>
    <td width="33%">
      <h3 align="center">📊 Interactive Dashboard</h3>
      <p align="center">Real-time visualization of sales trends with powerful filtering and drill-down capabilities</p>
    </td>
    <td width="33%">
      <h3 align="center">🔮 ML Predictions</h3>
      <p align="center">Generate accurate forecasts using advanced machine learning models with proven accuracy</p>
    </td>
    <td width="33%">
      <h3 align="center">📈 Performance Analysis</h3>
      <p align="center">Track forecast accuracy and model drift with automated monitoring and alerts</p>
    </td>
  </tr>
  <tr>
    <td width="33%">
      <h3 align="center">🔍 Model Insights</h3>
      <p align="center">Understand predictions with explainable AI and feature importance visualization</p>
    </td>
    <td width="33%">
      <h3 align="center">🔒 Enterprise Security</h3>
      <p align="center">JWT-based authentication with role-based access control and data encryption</p>
    </td>
    <td width="33%">
      <h3 align="center">⚙️ MLOps Integration</h3>
      <p align="center">Complete integration with MLflow for experiment tracking and model versioning</p>
    </td>
  </tr>
</table>

## 📊 Performance Metrics

<div align="center">
  
| Metric | Value | Description |
|--------|-------|-------------|
| **Forecast Accuracy** | 80.17% | Overall accuracy of predictions |
| **MAPE** | 19.83% | Mean Absolute Percentage Error |
| **MAE** | 46.16 | Mean Absolute Error |
| **RMSE** | 50.64 | Root Mean Square Error |

</div>

## 🛠️ Technical Architecture

Our platform follows a modern microservices architecture with three main components:

<p align="center">
  <img src="reports/architecture_diagram.png" alt="System Architecture" width="700"/>
</p>

```
┌───────────────────┐     ┌──────────────────┐     ┌────────────────┐
│                   │     │                  │     │                │
│ Streamlit         │─────▶ FastAPI          │─────▶ ML Models      │
│ Dashboard         │     │ Backend          │     │ & Predictions  │
│                   │     │                  │     │                │
└───────────────────┘     └──────────────────┘     └────────────────┘
                                   │                        │
                                   ▼                        ▼
                          ┌──────────────────┐     ┌────────────────┐
                          │                  │     │                │
                          │ Authentication   │     │ Feature        │
                          │ & Security       │     │ Engineering    │
                          │                  │     │                │
                          └──────────────────┘     └────────────────┘
                                                            │
                                                            ▼
                                                   ┌────────────────┐
                                                   │                │
                                                   │ Monitoring     │
                                                   │ & Metrics      │
                                                   │                │
                                                   └────────────────┘
```

## 🧰 Tech Stack

<div align="center">
  
### Frontend
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Altair](https://img.shields.io/badge/Altair-00A4EF?style=for-the-badge&logoColor=white)

### Backend
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=for-the-badge&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)

### ML & Data Science
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-38cc77?style=for-the-badge&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-00B7EB?style=for-the-badge&logoColor=white)

### DevOps & Monitoring
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-945DD6?style=for-the-badge&logo=dvc&logoColor=white)

</div>

## 📂 Project Structure

```
mlproject/
├── src/
│   ├── api/              # FastAPI application for model serving
│   ├── dashboard/        # Streamlit interface with visualizations
│   ├── database/         # Database models and connection utilities
│   ├── features/         # Feature engineering pipeline
│   ├── models/           # ML model definition, training and evaluation
│   ├── security/         # Authentication and authorization
│   └── utils/            # Shared utility functions
├── models/               # Serialized model artifacts
├── tests/                # Automated test suite
├── airflow/              # Airflow DAGs for scheduled tasks
├── monitoring/           # Monitoring and alerting components
├── docker-compose.yml    # Docker configuration
└── requirements.txt      # Python dependencies
```

## 🚀 Local Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/mlproject.git
cd mlproject

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start MLflow tracking server
mlflow ui --port 8888 --host 0.0.0.0

# Run the API
python -m src.api.main

# In another terminal, start the dashboard
streamlit run src/dashboard/app.py
```

## 🔐 Authentication

The system uses JWT-based authentication with the following demo credentials:
- **Username**: johndoe
- **Password**: secret

Or admin access:
- **Username**: admin
- **Password**: admin

## 📸 Screenshots

<div align="center">
  <p float="left">
    <img src="reports/metrics_overview.png" width="400" />
    <img src="reports/store_comparison.png" width="400" /> 
  </p>
  <p float="left">
    <img src="reports/model_explanations.png" width="400" />
    <img src="reports/mlflow_tracking.png" width="400" />
  </p>
</div>

## 📄 License

MIT

## 📧 Contact

Your Name - your.email@example.com

## Running with Attractive Landing Page

This project now offers a modern and attractive landing page as the entry point to the sales forecasting dashboard. The landing page was designed with a user experience similar to professional websites, including animations, smooth transitions, and modern design with a dark theme.

### Landing Page Features

- **Modern Design**: Dark theme interface with smooth gradients and animations
- **Visual Presentation**: Displays key metrics (54 stores, 34 product families, etc.)
- **Seamless Integration**: Automatically connects to the Streamlit dashboard
- **Professional Experience**: Provides an impactful first impression

### How to Run

To start the complete system with the landing page and dashboard:

```bash
python run_with_landing.py
```

This command will:
1. Start the landing page server at `http://localhost:8000`
2. Automatically start the Streamlit dashboard at `http://localhost:8501`
3. Open the browser on the landing page

The landing page offers "Login Dashboard" and "Access Dashboard" buttons that redirect to Streamlit, where all the main analysis functionality is available.

### Screenshots

The landing page displays information about the project, including the number of stores (54), product families (34), average sales ($55.92), and forecast accuracy (80.2%).

## Advanced Features

### Explainable AI Implementation

One of the most powerful features of this system is its **advanced explainability framework**. The project implements:

- **SHAP-like feature importance**: Measures each feature's contribution to predictions
- **Domain-aware explanations**: Uses retail-specific knowledge for meaningful insights
- **Interactive visualizations**: Shows feature impacts with proper context and tooltips
- **AI-powered recommendations**: Converts model outputs into actionable business insights

#### Explainability in Action

The dashboard showcases how ML explanations can be translated into business value through:

1. **Feature contribution visualization**: Shows exactly how each store, product, promotion, and time factor affects the prediction.
2. **Business recommendations**: Automatically generates inventory optimization advice based on predictions.
3. **Interactive tooltips**: Helps users interpret feature impacts without requiring ML knowledge.

This approach bridges the gap between complex machine learning models and business users who need transparency and interpretability.

### ML Pipeline

The project follows MLOps best practices with:

- **Modular architecture**: Separate components for data processing, model training, and serving.
- **API-first design**: All functionality accessible through RESTful endpoints.
- **Database integration**: Efficiently stores predictions and historical data.
- **Testing framework**: Ensures reliable model performance.

# Forecast Pipeline Dashboard

Um dashboard avançado para visualização de previsões de vendas de varejo, destinado a demonstrar habilidades técnicas em ML e design de interface.

## Funcionalidades Avançadas

- **Previsão de Vendas de Varejo**: Previsão automatizada de vendas para 54 lojas e 34 famílias de produtos.
- **Visualização de Dados**: Dashboard interativo com gráficos e métricas de desempenho.
- **Framework de Explicabilidade de ML**: Explicações detalhadas sobre como o modelo chega a cada previsão.
- **Pipeline de Dados Completo**: Desde o processamento de dados até a implantação do modelo.
- **Autenticação e Segurança**: Sistema completo de login e tokens JWT.
- **Recomendações de Negócios**: Insights acionáveis baseados em previsões para otimização de estoque e estratégias de vendas.

## Arquitetura

O projeto é estruturado em três componentes principais:

1. **Landing Page (Porta 8000)**: Página inicial e autenticação
2. **API (Porta 8002)**: Backend para previsões e explicabilidade do modelo
3. **Dashboard (Porta 8501)**: Interface de usuário Streamlit para visualização

## Instalação e Execução

### Requisitos

- Python 3.9+
- Pip ou Conda

### Configuração

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/forecast-pipeline-dashboard.git
   cd forecast-pipeline-dashboard
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Inicie a aplicação completa usando o script fornecido:
   ```bash
   bash restart.sh
   ```
   
   Isso iniciará todos os três componentes:
   - Landing Page em http://localhost:8000
   - API em http://localhost:8002
   - Dashboard em http://localhost:8501

4. Alternativamente, inicie cada componente manualmente:
   ```bash
   # Terminal 1 - Landing Page
   python -m uvicorn src.landing.server:app --host 0.0.0.0 --port 8000
   
   # Terminal 2 - API
   cd src/api && python -m uvicorn main:app --host 0.0.0.0 --port 8002
   
   # Terminal 3 - Dashboard
   python -m streamlit run src/dashboard/app.py --server.port=8501
   ```

5. Acesse o dashboard em seu navegador: http://localhost:8501

### Credenciais de Acesso

- Usuário: admin
- Senha: admin

## Framework de Explicabilidade ML

O sistema inclui um framework avançado de explicabilidade que torna as previsões do modelo interpretáveis para usuários de negócios:

- **Explicações baseadas em SHAP**: Utiliza valores SHAP (SHapley Additive exPlanations) quando disponíveis
- **Fallback Robusto**: Quando SHAP não está disponível, utiliza um mecanismo de fallback que:
  - Analisa importâncias de features para modelos baseados em árvores
  - Gera explicações baseadas em conhecimento de domínio para o setor de varejo
  - Implementa reshaping automático de arrays para compatibilidade com sklearn (resolvendo o erro "Expected 2D array, got 1D array instead")
  - Garante valores de contribuição balanceados e realistas
  - Detecta e corrige formatos de entrada incorretos para evitar falhas na explicação

- **Visualização Amigável**: Apresenta contribuições de features em formato visual intuitivo
- **Insights Acionáveis**: Traduz explicações técnicas em recomendações de negócios

## Recentes Melhorias (Maio 2025)

- **Correção de Bugs de Explicabilidade**: Resolvido problema "Expected 2D array, got 1D array instead" ao gerar explicações SHAP
- **Reshaping Automático de Features**: Implementado pre-processamento que garante formato correto dos arrays antes da predição
- **Interface do Usuário Aprimorada**: Texto da landing page reescrito para maior clareza sobre o propósito do projeto
- **Scripts de Deployment**: Adicionados scripts bash para facilitar a inicialização de todos os componentes do sistema
- **Documentação Expandida**: Instruções detalhadas e capturas de tela para guiar novos usuários

## Recomendações de Negócios

O sistema gera recomendações de negócios baseadas em:

- **Otimização de Estoque**: Sugere níveis ideais de estoque com base em previsões
- **Estratégia de Compras**: Calcula o Economic Order Quantity (EOQ) para compras eficientes
- **Análise de Desempenho**: Monitora crescimento ano-a-ano e padrões sazonais

## Contribuição

Contribuições são bem-vindas! Por favor, siga estas etapas:

1. Fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para detalhes.