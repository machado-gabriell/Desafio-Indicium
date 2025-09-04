# Desafio IMDb — Gabriel Machado (LH_CD_SEUNOME)

## Objetivo
Análise exploratória, modelagem e previsão da nota IMDb a partir do dataset fornecido.

## Estrutura do repositório
- `data/` — base original (adicione aqui `desafio_indicium_imdb.csv`)
- `notebooks/01_EDA_modelagem.ipynb` — notebook com EDA, NLP e modelagem
- `src/main.py` — script para pipeline (rodar no local/Colab)
- `src/predict.py` — utilitário para carregar `.pkl` e prever
- `models/imdb_rating_model.pkl` — modelo treinado (gerado por você)
- `reports/EDA_report.md` — relatório com respostas às perguntas
- `requirements.txt` — dependências (gerar com `pip freeze`)

## Como rodar (Google Colab)
1. Abrir `notebooks/01_EDA_modelagem.ipynb` no Colab.
2. Fazer upload do CSV em `data/`.
3. Rodar todas as células: o notebook gera `models/imdb_rating_model.pkl` e os relatórios em `outputs/`.
4. Compactar e baixar: notebook inclui célula para gerar `artefatos_imdb.zip`.

## Como rodar localmente
1. Criar e ativar virtualenv:
