import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google.genai.errors import ClientError, ServerError

from llm.gemini_model import gemini_client
from llm.groq_model import groq_client

load_dotenv()

TS_METRICS_PATH = os.getenv(
    "TS_METRICS_PATH",
    "data/experiments/thompson_sampling/metrics_timeseries.csv",
)
TS_METRICS_REPORT_PATH = os.getenv(
    "TS_METRICS_REPORT_PATH",
    "data/rag/thompson_sampling/metrics_timeseries_report.md",
)
ARM_COUNTS_BL_PATH = os.getenv(
    "ARM_COUNTS_BL_PATH",
    "data/experiments/thompson_sampling/arm_counts_BaselineFixedPolicy.csv",
)
ARM_COUNTS_REPORT_BL_PATH = os.getenv(
    "ARM_COUNTS_REPORT_BL_PATH",
    "data/rag/thompson_sampling/baseline_count_report.md",
)
ARM_COUNTS_TS_PATH = os.getenv(
    "ARM_COUNTS_TS_PATH",
    "data/experiments/thompson_sampling/arm_counts_ThompsonSamplingPolicy.csv",
)
ARM_COUNTS_REPORT_TS_PATH = os.getenv(
    "ARM_COUNTS_REPORT_TS_PATH",
    "data/rag/thompson_sampling/thompson_sampling_count_report.md",
)
OFFER_CATALOG_PATH = os.getenv(
    "OFFER_CATALOG_PATH",
    "data/kaggle/synthetic_enrichment/offer_catalog.json",
)
OFFER_CATALOG_REPORT_PATH = os.getenv(
    "OFFER_CATALOG_REPORT_PATH",
    "data/rag/thompson_sampling/offer_catalog_report.md",
)
METRICS_SUMMARY_PATH = os.getenv(
    "METRICS_SUMMARY_PATH",
    "data/experiments/thompson_sampling/metrics_summary.csv",
)

# =============================================================================
# LinUCB — caminhos de leitura CSVs gerados por LinUCBSimulator.run_linucb()
# e de destino (relatórios em markdown que o LLM vai escrever a partir deles).
# =============================================================================

# Leitura
LINUCB_METRICS_SUMMARY_PATH = os.getenv(
    "LINUCB_METRICS_SUMMARY_PATH",
    "data/experiments/linucb/metrics_summary.csv",
)
ARM_COUNTS_LINUCB_PATH = os.getenv(
    "ARM_COUNTS_LINUCB_PATH",
    "data/experiments/linucb/arm_counts_LinUCBPolicy.csv",
)
# Um CSV por coluna de perfil — cada um mostra qual braço a LinUCBPolicy
# recomendou mais para cada valor daquela coluna (ex: cada profissão).
ARM_BY_JOB_PATH = os.getenv(
    "ARM_BY_JOB_PATH",
    "data/experiments/linucb/arm_by_job.csv",
)
ARM_BY_EDUCATION_PATH = os.getenv(
    "ARM_BY_EDUCATION_PATH",
    "data/experiments/linucb/arm_by_education.csv",
)
ARM_BY_POUTCOME_PATH = os.getenv(
    "ARM_BY_POUTCOME_PATH",
    "data/experiments/linucb/arm_by_poutcome.csv",
)
ARM_BY_AGE_GROUP_PATH = os.getenv(
    "ARM_BY_AGE_GROUP_PATH",
    "data/experiments/linucb/arm_by_age_group.csv",
)

# Destino: relatórios em markdown gerados pelo LLM
LINUCB_METRICS_REPORT_PATH = os.getenv(
    "LINUCB_METRICS_REPORT_PATH",
    "data/rag/linucb/metrics_summary_report.md",
)
ARM_COUNTS_REPORT_LINUCB_PATH = os.getenv(
    "ARM_COUNTS_REPORT_LINUCB_PATH",
    "data/rag/linucb/arm_counts_linucb_report.md",
)
ARM_BY_JOB_REPORT_PATH = os.getenv(
    "ARM_BY_JOB_REPORT_PATH",
    "data/rag/linucb/arm_by_job_report.md",
)
ARM_BY_EDUCATION_REPORT_PATH = os.getenv(
    "ARM_BY_EDUCATION_REPORT_PATH",
    "data/rag/linucb/arm_by_education_report.md",
)
ARM_BY_POUTCOME_REPORT_PATH = os.getenv(
    "ARM_BY_POUTCOME_REPORT_PATH",
    "data/rag/linucb/arm_by_poutcome_report.md",
)
ARM_BY_AGE_GROUP_REPORT_PATH = os.getenv(
    "ARM_BY_AGE_GROUP_REPORT_PATH",
    "data/rag/linucb/arm_by_age_group_report.md",
)

PROMPTS = {

"prompt_ts_metrics": f"""
Você é um cientista de dados e engenheiro de machine learning.
Você tem acesso ao arquivo CSV {TS_METRICS_PATH} que contém métricas de experimentos de Thompson Sampling.
Você deve analisar os dados e escrever um relatório que compare a progressão da baseline com a progressão do modelo de Thompson Sampling.
Destacando as diferenças e insights obtidos a partir dos resultados.
Escreva o relatório em português e salve o resultado no arquivo {TS_METRICS_REPORT_PATH}.
Não se esqueça de incluir apenas o texto do arquivo markdown, sem nenhuma nota extra sobre o processo de geração do relatório.
Use {METRICS_SUMMARY_PATH} como referência para o resumo das métricas.
""",

"prompt_arm_counts_bl": f"""
Você é um cientista de dados e engenheiro de machine learning.
Você tem acesso ao arquivo CSV {ARM_COUNTS_BL_PATH} que contém a contagem de execuções de cada braço do experimento de Thompson Sampling com a política BaselineFixedPolicy.
Você deve analisar os dados e escrever um relatório que descreva a contagem de execuções de cada braço.
Escreva o relatório em português e salve o resultado no arquivo {ARM_COUNTS_REPORT_BL_PATH}.
Não se esqueça de incluir apenas o texto do arquivo markdown, sem nenhuma nota extra sobre o processo de geração do relatório.
Use {METRICS_SUMMARY_PATH} como referência para o resumo das métricas.
""",

"prompt_arm_counts_ts": f"""
Você é um cientista de dados e engenheiro de machine learning.
Você tem acesso ao arquivo CSV {ARM_COUNTS_TS_PATH} que contém a contagem de execuções de cada braço do experimento de Thompson Sampling.
Você deve analisar os dados e escrever um relatório que descreva a contagem de execuções de cada braço.
Escreva o relatório em português e salve o resultado no arquivo {ARM_COUNTS_REPORT_TS_PATH}.
Não se esqueça de incluir apenas o texto do arquivo markdown, sem nenhuma nota extra sobre o processo de geração do relatório.
Use {METRICS_SUMMARY_PATH} como referência para o resumo das métricas.
""",

"prompt_offer_catalog": f"""
Você é um cientista de dados e engenheiro de machine learning.
Você tem acesso ao arquivo JSON {OFFER_CATALOG_PATH} que contém o catálogo de ofertas.
Você deve analisar os dados e escrever um relatório que descreva as características e benefícios de cada oferta.
Escreva o relatório em português e salve o resultado no arquivo {OFFER_CATALOG_REPORT_PATH}.
Não se esqueça de incluir apenas o texto do arquivo markdown, sem nenhuma nota extra sobre o processo de geração do relatório.
Use {METRICS_SUMMARY_PATH} como referência para o resumo das métricas.
""",

# =============================================================================
# LinUCB — prompts para os relatórios gerados a partir dos CSVs do experimento
# contextual (ver LinUCBSimulator.run_linucb()).
# =============================================================================

"prompt_linucb_metrics": f"""
Você é um cientista de dados e engenheiro de machine learning.
Você tem acesso ao arquivo CSV {LINUCB_METRICS_SUMMARY_PATH} que contém as métricas agregadas do experimento de LinUCB (bandit contextual): impressões, conversões, taxa de conversão, recompensa acumulada, regret acumulado e entropia de exploração.
Você deve analisar os dados e escrever um relatório que explique o desempenho da política LinUCB, destacando o que os números indicam sobre o equilíbrio entre exploração e explotação e sobre o valor gerado pelas recomendações.
Escreva o relatório em português e salve o resultado no arquivo {LINUCB_METRICS_REPORT_PATH}.
Não se esqueça de incluir apenas o texto do arquivo markdown, sem nenhuma nota extra sobre o processo de geração do relatório.
""",

"prompt_arm_counts_linucb": f"""
Você é um cientista de dados e engenheiro de machine learning.
Você tem acesso ao arquivo CSV {ARM_COUNTS_LINUCB_PATH} que contém a contagem de quantas vezes cada braço (oferta) foi escolhido pela política LinUCB durante o experimento.
Você deve analisar os dados e escrever um relatório que descreva a distribuição de escolhas entre os braços, destacando qual oferta foi mais recomendada e possíveis razões de negócio para isso.
Escreva o relatório em português e salve o resultado no arquivo {ARM_COUNTS_REPORT_LINUCB_PATH}.
Não se esqueça de incluir apenas o texto do arquivo markdown, sem nenhuma nota extra sobre o processo de geração do relatório.
Use {LINUCB_METRICS_SUMMARY_PATH} como referência para o resumo das métricas.
""",

"prompt_arm_by_job": f"""
Você é um cientista de dados e engenheiro de machine learning.
Você tem acesso ao arquivo CSV {ARM_BY_JOB_PATH}, que mostra, para cada profissão (job) dos clientes, quantas vezes a política LinUCB recomendou cada braço (oferta). A coluna 'predominante' indica o braço mais recomendado para aquela profissão, e 'total' é o número de clientes elegíveis considerados.
Você deve analisar os dados e escrever um relatório que descreva quais ofertas a LinUCB associou a cada profissão, e o que isso sugere sobre o perfil de cliente que cada oferta atrai.
Escreva o relatório em português e salve o resultado no arquivo {ARM_BY_JOB_REPORT_PATH}.
Não se esqueça de incluir apenas o texto do arquivo markdown, sem nenhuma nota extra sobre o processo de geração do relatório.
Use {LINUCB_METRICS_SUMMARY_PATH} como referência para o resumo das métricas.
""",

"prompt_arm_by_education": f"""
Você é um cientista de dados e engenheiro de machine learning.
Você tem acesso ao arquivo CSV {ARM_BY_EDUCATION_PATH}, que mostra, para cada nível de escolaridade (education) dos clientes, quantas vezes a política LinUCB recomendou cada braço (oferta). A coluna 'predominante' indica o braço mais recomendado para aquele nível de escolaridade, e 'total' é o número de clientes elegíveis considerados.
Você deve analisar os dados e escrever um relatório que descreva quais ofertas a LinUCB associou a cada nível de escolaridade, e o que isso sugere sobre o perfil de cliente que cada oferta atrai.
Escreva o relatório em português e salve o resultado no arquivo {ARM_BY_EDUCATION_REPORT_PATH}.
Não se esqueça de incluir apenas o texto do arquivo markdown, sem nenhuma nota extra sobre o processo de geração do relatório.
Use {LINUCB_METRICS_SUMMARY_PATH} como referência para o resumo das métricas.
""",

"prompt_arm_by_poutcome": f"""
Você é um cientista de dados e engenheiro de machine learning.
Você tem acesso ao arquivo CSV {ARM_BY_POUTCOME_PATH}, que mostra, para cada resultado da campanha anterior (poutcome) do cliente, quantas vezes a política LinUCB recomendou cada braço (oferta). A coluna 'predominante' indica o braço mais recomendado para aquele resultado, e 'total' é o número de clientes elegíveis considerados.
Você deve analisar os dados e escrever um relatório que descreva quais ofertas a LinUCB associou a cada resultado de campanha anterior, e o que isso sugere sobre como o histórico do cliente influencia a recomendação.
Escreva o relatório em português e salve o resultado no arquivo {ARM_BY_POUTCOME_REPORT_PATH}.
Não se esqueça de incluir apenas o texto do arquivo markdown, sem nenhuma nota extra sobre o processo de geração do relatório.
Use {LINUCB_METRICS_SUMMARY_PATH} como referência para o resumo das métricas.
""",

"prompt_arm_by_age_group": f"""
Você é um cientista de dados e engenheiro de machine learning.
Você tem acesso ao arquivo CSV {ARM_BY_AGE_GROUP_PATH}, que mostra, para cada faixa etária (age_group) dos clientes, quantas vezes a política LinUCB recomendou cada braço (oferta). A coluna 'predominante' indica o braço mais recomendado para aquela faixa etária, e 'total' é o número de clientes elegíveis considerados.
Você deve analisar os dados e escrever um relatório que descreva quais ofertas a LinUCB associou a cada faixa etária, e o que isso sugere sobre o perfil de cliente que cada oferta atrai.
Escreva o relatório em português e salve o resultado no arquivo {ARM_BY_AGE_GROUP_REPORT_PATH}.
Não se esqueça de incluir apenas o texto do arquivo markdown, sem nenhuma nota extra sobre o processo de geração do relatório.
Use {LINUCB_METRICS_SUMMARY_PATH} como referência para o resumo das métricas.
"""
}

def generate_report_with_groq(prompt, source_path):
    """
    Função interna para realizar a chamada ao Groq usando o groq_client importado.
    Como o Groq não aceita Files API nativa para CSV/JSON locais, lemos o texto e unificamos.
    """

    print("🔄 Iniciando fallback: Gerando relatório via Groq (Llama 3)...")

    try:
        with open(source_path, "r", encoding="utf-8") as f:
            file_content = f.read()
    except Exception as e:
        raise IOError(f"Falha ao ler o arquivo local {source_path}: {e}")

    prompt_completo = (
        f"{prompt}\n\n"
        f"--- CONTEÚDO DO ARQUIVO DE ORIGEM ---\n"
        f"{file_content}\n"
        f"--- FIM DO ARQUIVO ---"
    )

    # Executa a requisição usando o padrão de completions do groq_client
    chat_completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt_completo,
            }
        ]
    )

    return chat_completion.choices[0].message.content


def generate_report(prompt, source_path, report_path):
    report_name = report_path.split("/")[-1]
    mime_type = "application/json" if source_path.endswith(".json") else "text/csv"
    resposta_texto = None

    try:
        print(f"Fazendo upload de {source_path} no Gemini...")
        doc_upload = gemini_client.files.upload(file=source_path, config={"mime_type": mime_type})

        print(f"Gerando o relatório {report_name} via Gemini...")

        # Chamada usando o gemini_client importado
        result = gemini_client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[prompt, doc_upload]
        )
        resposta_texto = result.text

    except Exception as e:
        print(f"\n⚠️ Falha ao processar com o Gemini ({type(e).__name__}): {e}")
        try:
            resposta_texto = generate_report_with_groq(prompt, source_path)
        except Exception as groq_err:
            print(f"❌ Erro crítico: Fallback para o Groq também falhou: {groq_err}")
            raise groq_err

    if resposta_texto:
        caminho_relatorio = Path(report_path)
        caminho_relatorio.parent.mkdir(parents=True, exist_ok=True)

        with open(caminho_relatorio, "w", encoding="utf-8") as f:
            f.write(resposta_texto)

        print(f"Relatório gerado e salvo com sucesso em: {report_path}!\n")
    else:
        raise Exception(f"Falha de execução: Nenhum modelo gerou conteúdo para {report_name}.")
