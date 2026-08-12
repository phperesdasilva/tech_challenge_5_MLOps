import os

from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Políticas de negócio — documentos Markdown usados como base de consulta no
# RAG e no catálogo de governança.
# =============================================================================

POLICIES_DIR = os.getenv("POLICIES_DIR", "data/policies")
FAQ_OFERTAS_PATH = os.getenv("FAQ_OFERTAS_PATH", f"{POLICIES_DIR}/faq-ofertas.md")
GOVERNANCA_DADOS_PATH = os.getenv(
    "GOVERNANCA_DADOS_PATH",
    f"{POLICIES_DIR}/governanca-dados.md",
)
POLITICA_COMUNICACAO_PATH = os.getenv(
    "POLITICA_COMUNICACAO_PATH",
    f"{POLICIES_DIR}/politica-comunicacao.md",
)
POLITICA_ELEGIBILIDADE_PATH = os.getenv(
    "POLITICA_ELEGIBILIDADE_PATH",
    f"{POLICIES_DIR}/politica-elegibilidade.md",
)
POLITICA_SUITABILITY_PATH = os.getenv(
    "POLITICA_SUITABILITY_PATH",
    f"{POLICIES_DIR}/politica-suitability.md",
)

POLICIES_REPORT_PATHS = [
    FAQ_OFERTAS_PATH,
    GOVERNANCA_DADOS_PATH,
    POLITICA_COMUNICACAO_PATH,
    POLITICA_ELEGIBILIDADE_PATH,
    POLITICA_SUITABILITY_PATH,
]

# =============================================================================
# Golden set — casos de avaliação e documentação associada ao conjunto de
# testes de recomendação.
# =============================================================================

GOLDEN_SET_PATH = os.getenv("GOLDEN_SET_PATH", "data/golden_set/golden_set.md")

# =============================================================================
# Thompson Sampling — caminhos de origem (CSVs/JSON do experimento) e destino
# (relatórios em markdown gerados pelo LLM a partir deles).
# =============================================================================

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
METRICS_SUMMARY_PATH = os.getenv(
    "METRICS_SUMMARY_PATH",
    "data/experiments/thompson_sampling/metrics_summary.csv",
)
OFFER_CATALOG_PATH = os.getenv(
    "OFFER_CATALOG_PATH",
    "data/kaggle/synthetic_enrichment/offer_catalog.json",
)
OFFER_CATALOG_REPORT_PATH = os.getenv(
    "OFFER_CATALOG_REPORT_PATH",
    "data/rag/thompson_sampling/offer_catalog_report.md",
)

# =============================================================================
# LinUCB — caminhos de leitura CSVs gerados por LinUCBSimulator.run_linucb()
# (data/experiments/linucb/) e de destino: relatórios em markdown
# (data/rag/linucb/).
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
