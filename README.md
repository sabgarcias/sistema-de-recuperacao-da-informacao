
git add README.md
git commit -m "Adiciona documentação do projeto"
git push origin main

1. ⚙️ Funcionalidades Principais
- Pré-processamento: Remoção de stopwords (Português) e uso de stemmer RSLP para redução de termos.
- Indexação: Construção de um Índice Invertido posicional para buscas mais precisas (frase).
- Ponderação: Utilização da Matriz TF-IDF (normalizada pela norma L2) para ranqueamento de documentos.
- Consultas Suportadas:
- Booleana: Para correspondência exata de todos os termos.
- Similaridade: Ranqueamento por similaridade de cosseno.
- Por Frase: Busca de termos em ordem e proximidade exata.

2. 🚀 Como Executar
- Pré-requisitos: Python 3.x.
- Instalação de Dependências: pip install nltk
- Execução: python atv.py

3. Desenvolvedores: 
- Sabrina Garcia da Silveira
- João Pedro Inocêncio Campos

# Projeto: Organização e Recuperação da Informação
Sistema de Recuperação da Informação (IR) implementado em Python. Inclui pré-processamento, stemming RSLP, cálculo de Matriz TF-IDF, Índice Invertido e suporte a consultas Booleana, Similaridade e por Frase.

