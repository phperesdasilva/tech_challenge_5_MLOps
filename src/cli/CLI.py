import os
from time import time

from dotenv import load_dotenv

from eda.DataManager import DataManager
from event_generator.SyntheticEventGenerator import SyntheticEventGenerator
from experiments.ThompsonSamplingSimulator import ThompsonSamplingSimulator
from rag.doc_generator import generate_report, PROMPTS
from rag.index_docs import index_documents
from rag.retriever import retrieve_context

load_dotenv()

TS_METRICS_PATH = os.getenv("TS_METRICS_PATH")
TS_METRICS_REPORT_PATH = os.getenv("TS_METRICS_REPORT_PATH")
ARM_COUNTS_BL_PATH = os.getenv("ARM_COUNTS_BL_PATH")
ARM_COUNTS_REPORT_BL_PATH = os.getenv("ARM_COUNTS_REPORT_BL_PATH")
ARM_COUNTS_TS_PATH = os.getenv("ARM_COUNTS_TS_PATH")
ARM_COUNTS_REPORT_TS_PATH = os.getenv("ARM_COUNTS_REPORT_TS_PATH")
METRICS_SUMMARY_PATH = os.getenv("METRICS_SUMMARY_PATH")

class CLI:
    def __init__(self):
        self.data_manager = DataManager()
        self.event_generator = SyntheticEventGenerator()
        self.thompson_sampling = ThompsonSamplingSimulator()

    def run_eda(self):
        self.data_manager.run_eda()

    def generate_events(self):
        self.event_generator.generate_events()

    def run_thompson_sampling(self):
        self.thompson_sampling.run_thompson_sampling()

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

    def generate_all_reports(self):
        self.generate_report_ts_metrics()
        time.sleep(10)
        self.generate_report_arm_counts_bl()
        time.sleep(10)
        self.generate_report_arm_counts_ts()

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
