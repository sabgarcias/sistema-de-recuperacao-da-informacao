import json
import re
from collections import Counter
import math
import nltk
try:
    nltk.download('rslp', quiet=True)
    nltk.download('stopwords', quiet=True)
except LookupError:
    pass
    
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

try:
    stopwords_pt = set(stopwords.words('portuguese'))
    stemmer = RSLPStemmer()
except LookupError:
    stopwords_pt = set(stopwords.words('portuguese'))
    stemmer = RSLPStemmer()



class SistemaIR:
    def __init__(self):
        self.colecao_original = {}
        self.vocabulario = {}
        self.indice_invertido = {}
        self.matriz_tf_idf = {}
        self.proximo_doc_id = 1
        self.dados_json_restantes = [] 

        print("Sistema de Recuperação de Informação Inicializado.")
    
    def _preprocessar_texto(self,texto):
        texto = texto.lower()
        texto = re.sub(r'[^\w\s]','', texto) 
        tokens = texto.split()

        tokens_processados = []
        for token in tokens:
            if token and token not in stopwords_pt: 
                radical = stemmer.stem(token)
                tokens_processados.append(radical)
        return tokens_processados
    
    def _reconstruir_estruturas(self):
        print("\nReconstruindo Vocabulário, Índice Invertido e Matriz TF-IDF...")

        self.vocabulario = {}
        self.indice_invertido = {}
        documentos_processados = {}

        for doc_id, doc in self.colecao_original.items():
            texto = doc['texto']
            tokens = self._preprocessar_texto(texto)
            documentos_processados[doc_id] = tokens

            for posicao, termo in enumerate(tokens):
                if termo not in self.vocabulario:
                    self.vocabulario[termo] = len(self.vocabulario) + 1

                if termo not in self.indice_invertido:
                    self.indice_invertido[termo] = {}

                if doc_id not in self.indice_invertido[termo]:
                    self.indice_invertido[termo][doc_id] = []

                self.indice_invertido[termo][doc_id].append(posicao)
        
        self._calcular_matriz_tf_idf(documentos_processados) 
        print("Estruturas reconstruídas com sucesso.")

    def _calcular_matriz_tf_idf(self, documentos_processados):
        self.matriz_tf_idf = {}
        N = len(self.colecao_original)
        
        idf = {}
        for termo in self.vocabulario:
            df_termo = len(self.indice_invertido.get(termo,{}))
            idf[termo] = math.log10(N/df_termo) if df_termo > 0 else 0 
        
        for doc_id, tokens in documentos_processados.items():
            tf_doc = Counter(tokens)
            self.matriz_tf_idf[doc_id] = {}
            max_freq = max(tf_doc.values()) if tf_doc else 1 

            # CORREÇÃO: O cálculo e atribuição do peso TF-IDF devem estar DENTRO deste loop
            for termo in set(tokens):
                tf_simples = tf_doc[termo]
                peso_tf_idf = tf_simples * idf.get(termo, 0)
                self.matriz_tf_idf[doc_id][termo] = peso_tf_idf

        self._normalizar_tf_idf()

    def _normalizar_tf_idf(self):
        for doc_id, vetor_tf_idf in self.matriz_tf_idf.items():
            soma_quadrados = sum(peso ** 2 for peso in vetor_tf_idf.values())
            norma_l2 = math.sqrt(soma_quadrados)

            if norma_l2 > 0:
                for termo in vetor_tf_idf:
                    vetor_tf_idf[termo] /= norma_l2

    def carregar_documentos_json(self, nome_arquivo):
        try:
            with open(nome_arquivo, 'r', encoding='utf-8') as f:
                dados = json.load(f)
            documentos_formatados = []
            for item in dados:
                documentos_formatados.append({'texto': item.get('content', '')}) 
            return documentos_formatados
            
        except FileNotFoundError:
            print(f"Erro: Arquivo '{nome_arquivo}' não encontrado. Certifique-se que o arquivo está na pasta correta.")
            return []
        except json.JSONDecodeError:
            print(f"Erro: O arquivo '{nome_arquivo}' não é um JSON válido.")
            return []
        
    def adicionar_documento(self, doc_json):
        doc_id = self.proximo_doc_id

        self.colecao_original[doc_id] = {
            'id': doc_id, 
            'texto': doc_json.get('texto', doc_json.get('Text', ''))
        }
        self.proximo_doc_id += 1

        self._reconstruir_estruturas()
        print(f"Documento ID {doc_id} adicionado e estruturas atualizadas.")
        return doc_id
    
    def remover_documento(self, doc_id):
        try:
            doc_id = int(doc_id)
        except ValueError:
            print("ID do documento inválido. Deve ser um número inteiro.")
            return
        
        if doc_id in self.colecao_original:
            del self.colecao_original[doc_id]
            if not self.colecao_original: 
                self.proximo_doc_id = 1
            self._reconstruir_estruturas()
            print(f"Documento ID {doc_id} removido e estruturas atualizadas.")
        else:
            print(f"Erro: Documento ID {doc_id} não encontrado.")
        
    def consulta_booleana(self, consulta):
        termos_consulta = self._preprocessar_texto(consulta)
        if not termos_consulta:
            print("Consulta vazia ou contém stopwords.")
            return
        documentos_candidatos = set()
        
        for doc_id, vetor_tf_idf in self.matriz_tf_idf.items():
            doc_contem_todos = True
            for termo in termos_consulta:
                if vetor_tf_idf.get(termo, 0) == 0: 
                    doc_contem_todos = False
                    break
            if doc_contem_todos:
                documentos_candidatos.add(doc_id)
                
        print(f"\nConsulta Booleana (Termos: {termos_consulta}):")
        if documentos_candidatos:
            print("Documentos que satisfazem a consulta (ID):", sorted(list(documentos_candidatos)))
            for doc_id in documentos_candidatos:
                print(f" ID {doc_id}: \"{self.colecao_original[doc_id]['texto'][:50]}...\"")  
        else:
            print("Nenhum documento satisfaz a consulta.")   

    def consulta_similaridade(self, consulta):
        termos_consulta = self._preprocessar_texto(consulta)
        if not termos_consulta:
            print("Consulta vazia ou contém apenas stopwords.")
            return 
        
        vetor_consulta = self._calcular_vetor_consulta(termos_consulta) 

        doc_candidatos = set()
        for termo in vetor_consulta:
            doc_candidatos.update(self.indice_invertido.get(termo, {}).keys())

        similaridades = {}
        soma_quadrados_consulta = sum(peso ** 2 for peso in vetor_consulta.values())
        norma_consulta = math.sqrt(soma_quadrados_consulta)

        for doc_id in doc_candidatos:
            vetor_documento = self.matriz_tf_idf.get(doc_id, {})
            produto_escalar = 0
            for termo, peso_consulta in vetor_consulta.items():
                peso_documento = vetor_documento.get(termo, 0)
                produto_escalar += peso_consulta * peso_documento
            
            if norma_consulta > 0:
                similaridade = produto_escalar / norma_consulta
                similaridades[doc_id] = similaridade
                
        ranqueamento = sorted(similaridades.items(), key=lambda item: item[1], reverse=True)     

        print(f"\nConsulta por Similaridade (Termos: {termos_consulta}):")
        if ranqueamento:
            for doc_id, similaridade in ranqueamento:
                if similaridade > 0:
                    print(f"   ID {doc_id} - Similaridade: {similaridade:.4f} - \"{self.colecao_original[doc_id]['texto'][:50]}...\"")
        else:
            print("Nenhum documento relevante encontrado.")

    def _calcular_vetor_consulta(self, termos_consulta):
        tf_consulta = Counter(termo for termo in termos_consulta if termo in self.vocabulario)
        vetor_consulta = {}
        N = len(self.colecao_original)
        
        for termo, freq in tf_consulta.items():
            if termo in self.vocabulario:
                df_termo = len(self.indice_invertido.get(termo, {}))
                idf = math.log10(N / df_termo) if df_termo > 0 else 0
                peso_tf_idf = freq * idf
                vetor_consulta[termo] = peso_tf_idf
                
        return vetor_consulta
    
    def consulta_por_frase(self, frase):
        termos_consulta = self._preprocessar_texto(frase)
        if len(termos_consulta) < 2:
            print("A busca por frase requer pelo menos dois termos após o pré-processamento.")
            return
        
        documentos_comuns = set(self.colecao_original.keys()) 
        for termo in termos_consulta:
            docs_do_termo = set(self.indice_invertido.get(termo, {}).keys())
            documentos_comuns = documentos_comuns.intersection(docs_do_termo)
            
        if not documentos_comuns:
             print(f"\n Busca por Frase (Frase: \"{frase}\" -> Termos: {termos_consulta}):")
             print("Nenhum documento contém todos os termos da frase.")
             return
            
        documentos_frase = []
        for doc_id in documentos_comuns:
            posicoes_inicio = self.indice_invertido[termos_consulta[0]][doc_id] 
        
            for pos_inicial in posicoes_inicio:
                ocorrencia_valida = True
                for i, termo in enumerate(termos_consulta[1:]): 
                    posicao_esperada = pos_inicial + i + 1 
                    posicoes_termo_atual = self.indice_invertido.get(termo, {}).get(doc_id, [])

                    if posicao_esperada not in posicoes_termo_atual:
                        ocorrencia_valida = False
                        break
                
                if ocorrencia_valida:
                    documentos_frase.append(doc_id)
                    break 
        
        print(f"\n Busca por Frase (Frase: \"{frase}\" -> Termos: {termos_consulta}):")
        if documentos_frase:
            ranqueamento_frase = self._calcular_similaridade_doc_especificos(termos_consulta, set(documentos_frase))
            for doc_id, similaridade in ranqueamento_frase:
                 print(f"   ID {doc_id} - Similaridade: {similaridade:.4f} - \"{self.colecao_original[doc_id]['texto'][:50]}...\"")
        else:
            print("A frase exata não foi encontrada em nenhum documento na ordem correta.")
            
    def _calcular_similaridade_doc_especificos(self, termos_consulta, doc_ids_candidatos):
        vetor_consulta = self._calcular_vetor_consulta(termos_consulta)
        similaridades = {}
        soma_quadrados_consulta = sum(peso ** 2 for peso in vetor_consulta.values())
        norma_consulta = math.sqrt(soma_quadrados_consulta)

        for doc_id in doc_ids_candidatos:
            vetor_documento = self.matriz_tf_idf.get(doc_id, {})
            produto_escalar = sum(vetor_consulta.get(termo, 0) * vetor_documento.get(termo, 0) for termo in vetor_consulta)
            
            if norma_consulta > 0:
                similaridades[doc_id] = produto_escalar / norma_consulta
        
        ranqueamento = sorted(similaridades.items(), key=lambda item: item[1], reverse=True)
        return ranqueamento

    def exibir_vocabulario(self):
        print("\n VOCABULÁRIO ATUALIZADO (Termo: ID)")
        print("---" * 15)
        for termo, termo_id in sorted(self.vocabulario.items()):
            print(f"  {termo}: {termo_id}")
        print(f"Total de Termos Únicos: {len(self.vocabulario)}")

    def exibir_matriz_tf_idf(self):
        print("\n MATRIZ TF-IDF ATUAL")
        print("---" * 15)
        if not self.matriz_tf_idf:
            print("Matriz TF-IDF vazia.")
            return
        termos_ordenados = sorted(self.vocabulario.keys()) 

        print(f"{'Doc ID':<8}", end="")
        for termo in termos_ordenados:
            print(f"{termo[:8]:<8}", end="  ") 
        print()
        
        print(f"{'-'*8:<8}", end="") 
        for _ in termos_ordenados:
            print(f"{'-'*8:<8}", end="  ")
        print()

        for doc_id, vetor_tf_idf in self.matriz_tf_idf.items():
            print(f"{doc_id:<8}", end="")
            for termo in termos_ordenados:
                peso = vetor_tf_idf.get(termo, 0)
                if peso > 0:
                    print(f"{peso:.4f}{'':<4}", end="") 
                else:
                    print(f"{'':<10}", end="") 
            print()

    def exibir_indice_invertido(self):
        print("\n ÍNDICE INVERTIDO COMPLETO (com Posições)")
        print("---" * 15)
        for termo, postings in sorted(self.indice_invertido.items()):
            postings_str = ", ".join([f"D{doc_id}: {posicoes}" for doc_id, posicoes in postings.items()])
            print(f"  {termo}: {postings_str}")

    def exibir_menu(self):
        while True:
            print("\n" + "=" * 40)
            print("SISTEMA DE RECUPERAÇÃO DA INFORMAÇÃO")
            print("=" * 40)
            print("1) Adicionar um documento à coleção (próximo do JSON)")
            print("2) Adicionar todos os documentos do JSON")
            print("3) Remover um documento (pelo ID)")
            print("---" * 13)
            print("4) Exibir o Vocabulário")
            print("5) Exibir a Matriz TF-IDF")
            print("6) Exibir o Índice Invertido (com posições) ")
            print("---" * 13)
            print("7) Realizar Consulta Booleana")
            print("8) Realizar Consulta por Similaridade")
            print("9) Realizar Consulta por Frase")
            print("10) Outras Operações (Sair)")
            print("=" * 40)
            
            escolha = input("Digite sua opção: ").strip()

            if escolha == '1':
                if not self.dados_json_restantes:
                    self.dados_json_restantes = self.carregar_documentos_json('colecao - trabalho 01.json')

                if self.dados_json_restantes:
                    doc = self.dados_json_restantes.pop(0)
                    self.adicionar_documento(doc)
                else:
                    print("Todos os documentos do JSON já foram adicionados ou o arquivo não foi encontrado/está vazio.")

            elif escolha == '2':
                dados_json = self.carregar_documentos_json('colecao - trabalho 01.json')
                
                if dados_json:
                    self.colecao_original = {}
                    self.proximo_doc_id = 1
                    
                    for doc in dados_json:
                        self.adicionar_documento(doc)
                    self.dados_json_restantes = [] 
                else:
                    print("Erro ao carregar documentos ou lista está vazia.")

            elif escolha == '3':
                doc_id = input("Digite o ID do documento a ser removido: ")
                self.remover_documento(doc_id)

            elif escolha == '4':
                self.exibir_vocabulario()

            elif escolha == '5':
                self.exibir_matriz_tf_idf()

            elif escolha == '6':
                self.exibir_indice_invertido()

            elif escolha == '7':
                consulta = input("Digite a consulta booleana (ex: termo1 termo2): ")
                self.consulta_booleana(consulta)

            elif escolha == '8':
                consulta = input("Digite a consulta por similaridade: ")
                self.consulta_similaridade(consulta)

            elif escolha == '9':
                frase = input("Digite a frase exata a ser buscada: ")
                self.consulta_por_frase(frase)

            elif escolha == '10':
                print("Saindo do sistema. Até logo!")
                break
            
            else:
                print("Opção inválida. Tente novamente.")

if __name__ == '__main__':
    
    sistema = SistemaIR()
    sistema.exibir_menu()
