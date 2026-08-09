import os
import tempfile
import time
from pathlib import Path
from dotenv import load_dotenv
from google.genai.errors import ClientError, ServerError
import pandas as pd

from llm.gemini_model import gemini_client
from llm.groq_model import groq_client
import rag.paths as paths

load_dotenv()

# Limite de linhas a partir do qual um CSV é reamostrado antes de ser enviado
# ao LLM, para não estourar o limite de tokens do Gemini/Groq.
MAX_ROWS_FOR_LLM = 2000

PROMPTS = {

"prompt_ts_metrics": f"""
Você é um cientista de dados e engenheiro de machine learning.
Você tem acesso ao arquivo CSV {paths.TS_METRICS_PATH} que contém métricas de experimentos de Thompson Sampling.
Você deve analisar os dados e escrever um relatório que compare a progressão da baseline com a progressão do modelo de Thompson Sampling.
Destacando as diferenças e insights obtidos a partir dos resultados.
Escreva o relatório em português e salve o resultado no arquivo {paths.TS_METRICS_REPORT_PATH}.
Não se esqueça de incluir apenas o texto do arquivo markdown, sem nenhuma nota extra sobre o processo de geração do relatório.
Use {paths.METRICS_SUMMARY_PATH} como referência para o resumo das métricas.
""",

"prompt_arm_counts_bl": f"""
Você é um cientista de dados e engenheiro de machine learning.
Você tem acesso ao arquivo CSV {paths.ARM_COUNTS_BL_PATH} que contém a contagem de execuções de cada braço do experimento de Thompson Sampling com a política BaselineFixedPolicy.
Você deve analisar os dados e escrever um relatório que descreva a contagem de execuções de cada braço.
Escreva o relatório em português e salve o resultado no arquivo {paths.ARM_COUNTS_REPORT_BL_PATH}.
Não se esqueça de incluir apenas o texto do arquivo markdown, sem nenhuma nota extra sobre o processo de geração do relatório.
Use {paths.METRICS_SUMMARY_PATH} como referência para o resumo das métricas.
""",

"prompt_arm_counts_ts": f"""
Você é um cientista de dados e engenheiro de machine learning.
Você tem acesso ao arquivo CSV {paths.ARM_COUNTS_TS_PATH} que contém a contagem de execuções de cada braço do experimento de Thompson Sampling.
Você deve analisar os dados e escrever um relatório que descreva a contagem de execuções de cada braço.
Escreva o relatório em português e salve o resultado no arquivo {paths.ARM_COUNTS_REPORT_TS_PATH}.
Não se esqueça de incluir apenas o texto do arquivo markdown, sem nenhuma nota extra sobre o processo de geração do relatório.
Use {paths.METRICS_SUMMARY_PATH} como referência para o resumo das métricas.
""",

"prompt_offer_catalog": f"""
Você é um cientista de dados e engenheiro de machine learning.
Você tem acesso ao arquivo JSON {paths.OFFER_CATALOG_PATH} que contém o catálogo de ofertas.
Você deve analisar os dados e escrever um relatório que descreva as características e benefícios de cada oferta.
Escreva o relatório em português e salve o resultado no arquivo {paths.OFFER_CATALOG_REPORT_PATH}.
Não se esqueça de incluir apenas o texto do arquivo markdown, sem nenhuma nota extra sobre o processo de geração do relatório.
Use {paths.METRICS_SUMMARY_PATH} como referência para o resumo das métricas.
""",

# =============================================================================
# LinUCB — prompts para os relatórios gerados a partir dos CSVs do experimento
# contextual (ver LinUCBSimulator.run_linucb()).
# =============================================================================

"prompt_linucb_metrics": f"""
Você é um cientista de dados e engenheiro de machine learning.
Você tem acesso ao arquivo CSV {paths.LINUCB_METRICS_SUMMARY_PATH} que contém as métricas agregadas do experimento de LinUCB (bandit contextual): impressões, conversões, taxa de conversão, recompensa acumulada, regret acumulado e entropia de exploração.
Você deve analisar os dados e escrever um relatório que explique o desempenho da política LinUCB, destacando o que os números indicam sobre o equilíbrio entre exploração e explotação e sobre o valor gerado pelas recomendações.
Escreva o relatório em português e salve o resultado no arquivo {paths.LINUCB_METRICS_REPORT_PATH}.
Não se esqueça de incluir apenas o texto do arquivo markdown, sem nenhuma nota extra sobre o processo de geração do relatório.
""",

"prompt_arm_counts_linucb": f"""
Você é um cientista de dados e engenheiro de machine learning.
Você tem acesso ao arquivo CSV {paths.ARM_COUNTS_LINUCB_PATH} que contém a contagem de quantas vezes cada braço (oferta) foi escolhido pela política LinUCB durante o experimento.
Você deve analisar os dados e escrever um relatório que descreva a distribuição de escolhas entre os braços, destacando qual oferta foi mais recomendada e possíveis razões de negócio para isso.
Escreva o relatório em português e salve o resultado no arquivo {paths.ARM_COUNTS_REPORT_LINUCB_PATH}.
Não se esqueça de incluir apenas o texto do arquivo markdown, sem nenhuma nota extra sobre o processo de geração do relatório.
Use {paths.LINUCB_METRICS_SUMMARY_PATH} como referência para o resumo das métricas.
""",

"prompt_arm_by_job": f"""
Você é um cientista de dados e engenheiro de machine learning.
Você tem acesso ao arquivo CSV {paths.ARM_BY_JOB_PATH}, que mostra, para cada profissão (job) dos clientes, quantas vezes a política LinUCB recomendou cada braço (oferta). A coluna 'predominante' indica o braço mais recomendado para aquela profissão, e 'total' é o número de clientes elegíveis considerados.
Você deve analisar os dados e escrever um relatório que descreva quais ofertas a LinUCB associou a cada profissão, e o que isso sugere sobre o perfil de cliente que cada oferta atrai.
Escreva o relatório em português e salve o resultado no arquivo {paths.ARM_BY_JOB_REPORT_PATH}.
Não se esqueça de incluir apenas o texto do arquivo markdown, sem nenhuma nota extra sobre o processo de geração do relatório.
Use {paths.LINUCB_METRICS_SUMMARY_PATH} como referência para o resumo das métricas.
""",

"prompt_arm_by_education": f"""
Você é um cientista de dados e engenheiro de machine learning.
Você tem acesso ao arquivo CSV {paths.ARM_BY_EDUCATION_PATH}, que mostra, para cada nível de escolaridade (education) dos clientes, quantas vezes a política LinUCB recomendou cada braço (oferta). A coluna 'predominante' indica o braço mais recomendado para aquele nível de escolaridade, e 'total' é o número de clientes elegíveis considerados.
Você deve analisar os dados e escrever um relatório que descreva quais ofertas a LinUCB associou a cada nível de escolaridade, e o que isso sugere sobre o perfil de cliente que cada oferta atrai.
Escreva o relatório em português e salve o resultado no arquivo {paths.ARM_BY_EDUCATION_REPORT_PATH}.
Não se esqueça de incluir apenas o texto do arquivo markdown, sem nenhuma nota extra sobre o processo de geração do relatório.
Use {paths.LINUCB_METRICS_SUMMARY_PATH} como referência para o resumo das métricas.
""",

"prompt_arm_by_poutcome": f"""
Você é um cientista de dados e engenheiro de machine learning.
Você tem acesso ao arquivo CSV {paths.ARM_BY_POUTCOME_PATH}, que mostra, para cada resultado da campanha anterior (poutcome) do cliente, quantas vezes a política LinUCB recomendou cada braço (oferta). A coluna 'predominante' indica o braço mais recomendado para aquele resultado, e 'total' é o número de clientes elegíveis considerados.
Você deve analisar os dados e escrever um relatório que descreva quais ofertas a LinUCB associou a cada resultado de campanha anterior, e o que isso sugere sobre como o histórico do cliente influencia a recomendação.
Escreva o relatório em português e salve o resultado no arquivo {paths.ARM_BY_POUTCOME_REPORT_PATH}.
Não se esqueça de incluir apenas o texto do arquivo markdown, sem nenhuma nota extra sobre o processo de geração do relatório.
Use {paths.LINUCB_METRICS_SUMMARY_PATH} como referência para o resumo das métricas.
""",

"prompt_arm_by_age_group": f"""
Você é um cientista de dados e engenheiro de machine learning.
Você tem acesso ao arquivo CSV {paths.ARM_BY_AGE_GROUP_PATH}, que mostra, para cada faixa etária (age_group) dos clientes, quantas vezes a política LinUCB recomendou cada braço (oferta). A coluna 'predominante' indica o braço mais recomendado para aquela faixa etária, e 'total' é o número de clientes elegíveis considerados.
Você deve analisar os dados e escrever um relatório que descreva quais ofertas a LinUCB associou a cada faixa etária, e o que isso sugere sobre o perfil de cliente que cada oferta atrai.
Escreva o relatório em português e salve o resultado no arquivo {paths.ARM_BY_AGE_GROUP_REPORT_PATH}.
Não se esqueça de incluir apenas o texto do arquivo markdown, sem nenhuma nota extra sobre o processo de geração do relatório.
Use {paths.LINUCB_METRICS_SUMMARY_PATH} como referência para o resumo das métricas.
"""
}

def _prepare_source_for_llm(source_path):
    """
    Reduz o tamanho de um CSV antes de ele ser enviado a um LLM (Gemini/Groq).

    Alguns CSVs de origem (ex: metrics_timeseries.csv) guardam uma linha por
    step de simulação. Com o dataset completo (45 mil+ eventos), isso gera
    arquivos de vários MB, que estouram o limite de contexto do Gemini
    (1.048.576 tokens) e do Groq. Para o relatório de progressão ainda fazer
    sentido, não basta truncar o arquivo: é preciso manter uma amostra
    representativa da evolução ao longo de TODOS os steps, para TODAS as
    policies presentes — não só o começo do arquivo.

    Passo a passo:
      1. Se o arquivo não for .csv (ex: offer_catalog.json), devolve
         source_path sem alteração. A reamostragem só faz sentido para CSVs
         tabulares.
      2. Se o CSV já for pequeno (linhas <= MAX_ROWS_FOR_LLM), também
         devolve source_path sem alteração — não vale a pena criar um
         arquivo temporário para algo que já cabe no contexto do LLM.
      3. Caso contrário, reamostra o CSV (ver _downsample_dataframe).
      4. Salva o resultado em um CSV temporário à parte — o arquivo original
         em disco nunca é modificado — e devolve o caminho desse temporário.
         Quem chama esta função é responsável por apagá-lo depois de usá-lo
         (ver o bloco `finally` em generate_report()).

    Args:
        source_path: caminho do arquivo CSV/JSON de origem do relatório.

    Returns:
        source_path original (quando não precisa reamostrar) ou o caminho
        de um CSV temporário já reamostrado, pronto para envio ao LLM.
    """
    eh_csv = source_path.endswith(".csv")
    if not eh_csv:
        # JSON (ex: catálogo de ofertas) e outros formatos não têm a noção
        # de "linha" usada aqui, e já são pequenos o bastante.
        return source_path

    df = pd.read_csv(source_path)
    arquivo_ja_e_pequeno = len(df) <= MAX_ROWS_FOR_LLM
    if arquivo_ja_e_pequeno:
        return source_path

    df_sampled = _downsample_dataframe(df, max_rows=MAX_ROWS_FOR_LLM)

    # Escreve a versão reamostrada em um arquivo temporário; o CSV original
    # permanece intocado em disco.
    tmp_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    )
    tmp_file.close()
    df_sampled.to_csv(tmp_file.name, index=False)

    print(
        f"⚠️ {source_path} tem {len(df)} linhas; reamostrado para {len(df_sampled)} "
        f"linhas antes de enviar ao LLM."
    )
    return tmp_file.name


def _downsample_dataframe(df, max_rows):
    """
    Reduz um DataFrame a, no máximo, aproximadamente `max_rows` linhas,
    preservando a ordem original e cobrindo do início ao fim dos dados
    (amostragem por passo/stride uniforme, não um corte no fim do arquivo).

    Caso especial — coluna "policy": o metrics_timeseries.csv concatena uma
    "faixa" de linhas por policy (todas as linhas da BaselineFixedPolicy,
    depois todas as da ThompsonSamplingPolicy, etc). Se reamostrássemos o
    arquivo inteiro de uma vez com um único stride, uma policy poderia ficar
    sub-representada (ou até de fora) na amostra final, dependendo de onde
    suas linhas caem no arquivo. Por isso, quando existe a coluna "policy",
    reamostramos CADA GRUPO separadamente, dividindo o orçamento de
    `max_rows` igualmente entre as policies — garantindo que a progressão de
    cada uma apareça, do primeiro ao último step.
    """
    tem_coluna_policy = "policy" in df.columns

    if not tem_coluna_policy:
        stride = max(len(df) // max_rows, 1)
        return df.iloc[::stride]

    numero_de_policies = df["policy"].nunique()
    linhas_alvo_por_policy = max(max_rows // numero_de_policies, 1)

    def _downsample_um_grupo_de_policy(grupo):
        # Stride necessário para reduzir este grupo a ~linhas_alvo_por_policy
        # linhas, mantendo a ordem original (ex: por "step") e cobrindo do
        # início ao fim da série dessa policy.
        stride = max(len(grupo) // linhas_alvo_por_policy, 1)
        return grupo.iloc[::stride]

    return df.groupby("policy", group_keys=False)[df.columns.tolist()].apply(
        _downsample_um_grupo_de_policy
    )


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

    llm_source_path = _prepare_source_for_llm(source_path)

    try:
        try:
            print(f"Fazendo upload de {llm_source_path} no Gemini...")
            doc_upload = gemini_client.files.upload(file=llm_source_path, config={"mime_type": mime_type})

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
                resposta_texto = generate_report_with_groq(prompt, llm_source_path)
            except Exception as groq_err:
                print(f"❌ Erro crítico: Fallback para o Groq também falhou: {groq_err}")
                raise groq_err
    finally:
        if llm_source_path != source_path:
            os.remove(llm_source_path)

    if resposta_texto:
        caminho_relatorio = Path(report_path)
        caminho_relatorio.parent.mkdir(parents=True, exist_ok=True)

        with open(caminho_relatorio, "w", encoding="utf-8") as f:
            f.write(resposta_texto)

        print(f"Relatório gerado e salvo com sucesso em: {report_path}!\n")
    else:
        raise Exception(f"Falha de execução: Nenhum modelo gerou conteúdo para {report_name}.")
