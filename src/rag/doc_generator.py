import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google.genai.errors import ClientError, ServerError

from llm.gemini_model import client

load_dotenv()

PROMPTS = {

"prompt_ts_metrics": f"""
Você é um cientista de dados e engenheiro de machine learning.
Você tem acesso ao arquivo CSV {os.getenv('TS_METRICS_PATH')} que contém métricas de experimentos de Thompson Sampling.
Você deve analisar os dados e escrever um relatório que compare a progressão da baseline com a progressão do modelo de Thompson Sampling.
Destacando as diferenças e insights obtidos a partir dos resultados.
Escreva o relatório em português e salve o resultado no arquivo {os.getenv('TS_METRICS_REPORT_PATH')}.
Não se esqueça de incluir apenas o texto do arquivo markdown, sem nenhuma nota extra sobre o processo de geração do relatório.
Use {os.getenv('METRICS_SUMMARY_PATH')} como referência para o resumo das métricas.
""",

"prompt_arm_counts_bl": f"""
Você é um cientista de dados e engenheiro de machine learning.
Você tem acesso ao arquivo CSV {os.getenv('ARM_COUNTS_BL_PATH')} que contém a contagem de execuções de cada braço do experimento de Thompson Sampling com a política BaselineFixedPolicy.
Você deve analisar os dados e escrever um relatório que descreva a contagem de execuções de cada braço.
Escreva o relatório em português e salve o resultado no arquivo {os.getenv('ARM_COUNTS_REPORT_BL_PATH')}.
Não se esqueça de incluir apenas o texto do arquivo markdown, sem nenhuma nota extra sobre o processo de geração do relatório.
Use {os.getenv('METRICS_SUMMARY_PATH')} como referência para o resumo das métricas.
""",

"prompt_arm_counts_ts": f"""
Você é um cientista de dados e engenheiro de machine learning.
Você tem acesso ao arquivo CSV {os.getenv('ARM_COUNTS_TS_PATH')} que contém a contagem de execuções de cada braço do experimento de Thompson Sampling.
Você deve analisar os dados e escrever um relatório que descreva a contagem de execuções de cada braço.
Escreva o relatório em português e salve o resultado no arquivo {os.getenv('ARM_COUNTS_REPORT_TS_PATH')}.
Não se esqueça de incluir apenas o texto do arquivo markdown, sem nenhuma nota extra sobre o processo de geração do relatório.
Use {os.getenv('METRICS_SUMMARY_PATH')} como referência para o resumo das métricas.
"""
}

def generate_report(prompt, source_path, report_path):

    report_name = report_path.split("/")[-1]

    print(f"Fazendo upload de {source_path}...")
    doc_upload = client.files.upload(file=source_path, config={"mime_type": "text/csv"})

    #tentativas por causa do rate limit da API do Gemini
    tentativas = 3
    for i in range(tentativas):
        try:
            print(f"Gerando o relatório {report_name}... (Tentativa {i+1}/{tentativas})")
            result = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=[prompt, doc_upload]
            )
            break

        except ClientError as e:
            # Se for erro de cota esgotada (429), espera o tempo recomendado e tenta de novo
            if e.code == 429:
                tempo_espera = 60
                print(f"\n⚠️ Limite de cota atingido (429). Aguardando {tempo_espera} segundos para liberar a API...")
                time.sleep(tempo_espera)
            else:
                raise e

        except ServerError as e:
            # Trata instabilidade do servidor do Google (503 / Alta Demanda)
            tempo_espera = 20 * (i + 1)  # Backoff incremental: 20s, 40s, 60s...
            print(f"\n⚠️ Servidor sob alta demanda (503). Aguardando {tempo_espera}s antes de tentar novamente...")
            time.sleep(tempo_espera)

        except Exception as e:
            print(f"Erro inesperado: {str(e)}")
            raise e
    else:
        raise Exception("Não foi possível gerar o relatório devido a limites persistentes da API.")

    caminho_relatorio = Path(report_path)
    caminho_relatorio.parent.mkdir(parents=True, exist_ok=True)

    with open(caminho_relatorio, "w", encoding="utf-8") as f:
        f.write(result.text)

    print(f"Relatório gerado e salvo com sucesso em: {report_path}!\n")
