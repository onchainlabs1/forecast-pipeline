# Guia de Preparação para Entrevista: Projeto de Previsão de Vendas

Este documento detalha os componentes principais do sistema de previsão de vendas, o fluxo de dados entre eles, potenciais pontos de falha e como demonstrar o projeto de maneira robusta em uma entrevista técnica.

## Visão Geral da Arquitetura

O projeto consiste em quatro componentes principais:

1. **API REST (FastAPI)** - Fornece endpoints para autenticação, previsões e explicabilidade
2. **Dashboard (Streamlit)** - Interface de usuário para visualizar previsões e métricas
3. **Pipeline de ML (scikit-learn/LightGBM/XGBoost)** - Pré-processamento, treinamento e avaliação de modelos
4. **Rastreamento (MLflow)** - Rastreamento de experimentos, armazenamento de modelos e métricas

### Fluxo de Dados

```
[Dados Históricos] → [Pipeline de ML] → [Modelo Treinado] → [MLflow]
                                                           ↓
[Entrada do Usuário] → [Dashboard] → [API] → [Previsão] → [Visualização]
                         ↑               ↓
                      [Autenticação] ← [JWT]
```

## Pontos Críticos e Potenciais Falhas

Aqui estão os pontos críticos do sistema e como eles foram tratados para garantir robustez:

### 1. Geração de Features

**Problema identificado:** A função `generate_features()` tinha um bug de índice quando processsava lojas com identificadores altos.

**Solução:** Implementamos um mapeamento correto dos índices e adicionamos verificações de limite para evitar erros de acesso fora dos limites do array. O código agora mapeia cada número de loja corretamente para seu índice dentro do vetor de features.

**Teste:** O script `test_features.py` realiza testes extensivos da função com diferentes valores de loja, família e datas, incluindo casos extremos.

### 2. Dependências Externas (SHAP)

**Problema potencial:** O pacote SHAP é necessário para a explicabilidade do modelo, mas pode não estar disponível em todos os ambientes.

**Solução:** Implementamos um sistema de fallback que permite que o sistema continue funcionando mesmo sem o SHAP instalado. Quando o SHAP não está disponível, o sistema fornece uma explicação simplificada baseada nas importâncias de features do modelo.

**Teste:** Verifique a explicabilidade com o script `test_prediction.py`, que testa as previsões e explicações em diferentes configurações.

### 3. Autenticação e Segurança

**Potencial problema:** Falhas na autenticação podem expor dados sensíveis ou permitir acesso não autorizado.

**Solução:** Implementamos autenticação JWT com token de acesso, validação de escopo e proteção contra ataques comuns. Os tokens expiram após um período configurável.

**Teste:** O sistema inclui verificações de autenticação no fluxo normal de uso e testes para casos de acesso não autorizado.

### 4. Precisão do Modelo

**Potencial problema:** A precisão do modelo pode deteriorar-se com o tempo (model drift).

**Solução:** O sistema monitora continuamente o desempenho do modelo em produção, comparando previsões com valores reais. Alertas são gerados quando o desempenho cai abaixo de um limiar configurável.

**Teste:** O endpoint `/model_drift` fornece métricas atualizadas sobre o desempenho do modelo.

### 5. Tratamento de Erros

**Potencial problema:** Erros não tratados podem interromper o serviço.

**Solução:** Implementamos tratamento abrangente de exceções em todo o código, com fallbacks adequados para cada componente crítico. Logs detalhados são gerados para facilitar o diagnóstico.

**Teste:** O script `test_prediction.py` inclui teste de casos extremos para garantir que o sistema responda adequadamente em situações adversas.

## Como Demonstrar o Projeto em uma Entrevista

### 1. Preparação

Antes da entrevista:

- Execute o script `check_dependencies.py` para garantir que todas as dependências estão instaladas
- Execute `test_features.py` e `test_prediction.py` para verificar a integridade do sistema
- Prepare o ambiente executando os serviços API e MLflow em segundo plano

### 2. Demonstração do Fluxo Completo

Durante a entrevista, demonstre o fluxo completo:

1. **Login e Autenticação:** Mostre o processo de autenticação e obtenção do token JWT
2. **Dashboard Principal:** Apresente as métricas principais e KPIs do negócio
3. **Previsão em Tempo Real:** Faça uma previsão para uma combinação específica de loja/produto
4. **Explicabilidade:** Mostre como o modelo explica suas decisões usando SHAP
5. **Monitoramento de Performance:** Acesse o MLflow para mostrar métricas de desempenho histórico

### 3. Pontos a Destacar

Destaque estes aspectos técnicos durante sua apresentação:

- **Robustez:** Como o sistema lida com casos extremos e entrada inválida
- **Escalabilidade:** Como o design permite expansão para mais lojas/produtos
- **Explicabilidade:** Como tornamos o modelo interpretável para usuários de negócio
- **Monitoramento:** Como rastreamos a precisão do modelo ao longo do tempo
- **DevOps:** Como o sistema seria implantado em produção (CI/CD, monitoramento)

### 4. Perguntas Técnicas Comuns e Respostas

Esteja preparado para essas perguntas comuns:

1. **"Como você escolheu o algoritmo de ML para este problema?"**
   - Resposta: Avaliamos diversos algoritmos (Linear, RandomForest, XGBoost, LightGBM) e escolhemos LightGBM por seu equilíbrio entre precisão e velocidade. Para séries temporais, consideramos também ARIMA e Prophet, mas o LightGBM com features temporais teve melhor desempenho.

2. **"Como você garante que o modelo não está sobreajustado?"**
   - Resposta: Usamos validação cruzada com séries temporais (TimeSeriesSplit), regularização adequada e monitoramento contínuo da diferença entre métricas de treino e teste.

3. **"Como você lidaria com novos produtos sem histórico?"**
   - Resposta: Implementamos uma estratégia de cold-start que usa dados de produtos similares na mesma categoria e aplica ajustes baseados nas características da loja.

4. **"Como você avalia se a precisão do modelo é suficiente?"**
   - Resposta: Além das métricas técnicas (RMSE, MAE), definimos métricas de negócio como "% de estoque não vendido" e "% de demanda não atendida" para avaliar o impacto financeiro das previsões.

5. **"Como você implementaria este sistema em escala?"**
   - Resposta: Para escalabilidade, usaríamos contêineres com orquestração Kubernetes, técnicas de caching para previsões comuns, e atualizações incrementais do modelo.

## Recursos Adicionais

Para preparação avançada:

- Explore os notebooks na pasta `notebooks/` que mostram a exploração de dados e seleção de modelo
- Revise os arquivos de configuração para entender os parâmetros do modelo
- Estude o esquema do banco de dados para entender como as previsões são armazenadas 