import os
from time import sleep

from dotenv import load_dotenv

from eda.DataManager import DataManager
from event_generator.SyntheticEventGenerator import SyntheticEventGenerator
from experiments.ThompsonSamplingSimulator import ThompsonSamplingSimulator
from experiments.LinUCBSimulator import LinUCBSimulator
from rag.doc_generator import generate_report, PROMPTS
from rag.index_docs import index_documents
from rag.retriever import retrieve_context
from rag.prompt_builder import build_rag_prompt
from llm.llm_main import run_llm

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

class CLI:
    def __init__(self):
        self.data_manager = DataManager()
        self.event_generator = SyntheticEventGenerator()
        self.thompson_sampling = ThompsonSamplingSimulator()
        self.linucb = LinUCBSimulator()

    def run_eda(self):
        self.data_manager.run_eda()

    def generate_events(self):
        self.event_generator.generate_events()

    def run_thompson_sampling(self):
        self.thompson_sampling.run_thompson_sampling()

    def run_linucb(self):
        self.linucb.run_linucb()

    def generate_report_ts_metrics(self):
        generate_report(
            prompt=PROMPTS["prompt_ts_metrics"],
            source_path=TS_METRICS_PATH,
            report_path=TS_METRICS_REPORT_PATH
        )

    def generate_report_arm_counts_bl(self):
        generate_report(
            prompt=PROMPTS["prompt_arm_counts_bl"],
            source_path=ARM_COUNTS_BL_PATH,
            report_path=ARM_COUNTS_REPORT_BL_PATH
        )

    def generate_report_arm_counts_ts(self):
        generate_report(
            prompt=PROMPTS["prompt_arm_counts_ts"],
            source_path=ARM_COUNTS_TS_PATH,
            report_path=ARM_COUNTS_REPORT_TS_PATH
        )

    def generate_report_offer_catalog(self):
        generate_report(
            prompt=PROMPTS["prompt_offer_catalog"],
            source_path=OFFER_CATALOG_PATH,
            report_path=OFFER_CATALOG_REPORT_PATH
        )

    def generate_all_reports(self):
        self.generate_report_ts_metrics()
        sleep(10)
        self.generate_report_arm_counts_bl()
        sleep(10)
        self.generate_report_arm_counts_ts()
        sleep(10)
        self.generate_report_offer_catalog()

    def index_documents(self):
        index_documents()

    def retrieve_context(self, query: str, top_k: int = 2):
        """
        Método que chama a busca vetorial no FAISS e formata o resultado no terminal.
        """
        results = retrieve_context(query, top_k=top_k)

        if not results:
            print("\n❌ Nenhum contexto relevante foi encontrado para a consulta informada.")
            return

        print(f"\n🔍 Buscando por: '{query}'")
        print(f"📊 Foram encontrados {len(results)} trechos mais relevantes:\n")
        print("=" * 80)

        for idx, res in enumerate(results):
            score = res["score"]
            origem = res["metadata"]["source"]
            chunk_id = res["metadata"]["chunk_id"]
            texto = res["chunk"]

            print(f"📌 Resultado #{idx + 1} | Origem: {origem} (Chunk: {chunk_id}) | Distância L2: {score:.4f}")
            print("-" * 80)
            print(texto.strip())
            print("=" * 80)

    def build_rag_prompt(self, prompt: str) -> str:
        """
        Método que constrói o prompt RAG com base na consulta do usuário e no contexto recuperado.
        """
        rag_prompt = build_rag_prompt(prompt)
        return rag_prompt

    def ask_llm(self, prompt: str = None):
        run_llm(prompt=prompt)
