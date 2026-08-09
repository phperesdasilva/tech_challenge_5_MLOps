"""
Algoritmo LinUCB (Linear Upper Confidence Bound)
O LinUCB é um algoritmo de bandit contextual que assume que a recompensa esperada de cada braço 
é uma função linear do vetor de contexto do cliente. Ele mantém uma estimativa dos pesos para cada braço e 
atualiza esses pesos com base nas recompensas observadas.

A diferença em relação ao Thompson Sampling é que o LinUCB utiliza informações contextuais (features do cliente) 
para tomar decisões, enquanto o Thompson Sampling é não-contextual e baseia-se apenas nas recompensas observadas.

"""

import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from bandit.catalog import get_eligible_offers, load_catalog
from bandit.features import BankContextEncoder
from bandit.policies import LinUCBPolicy
from bandit.simulator import run_simulation
from bandit.tracking import configure_mlflow, log_dataset, log_policy_metrics


class LinUCBSimulator:
    def __init__(self):
        load_dotenv()

        self.output_dir = Path(os.getenv("LINUCB_OUT_DIR", "data/experiments/linucb"))
        self.seed = int(os.getenv("SEED", "42"))
        self.default_bank_path = Path(os.getenv("DEFAULT_BANK_PATH", "data/kaggle/processed/clean_bank.parquet"))
        
        if not self.verifica_eda_executado():
            raise FileNotFoundError("O arquivo clean_bank.parquet não foi encontrado. Execute o EDA primeiro!")

    def verifica_eda_executado(self):
        return self.default_bank_path.exists()

    def run_linucb(self):
        """Executa o experimento apenas com a LinUCBPolicy e salva os resultados."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        configure_mlflow(os.getenv("MLFLOW_EXPERIMENT_LINUCB", "LinUCB"))

        ofertas = load_catalog()
        id_bracos = [oferta["id_braco"] for oferta in ofertas]
        df_clean_bank = pd.read_parquet(self.default_bank_path)

        # Cria o encoder de contexto (transforma cada cliente em um vetor numérico de features)
        # * Esse vetor é o que a LinUCBPolicy usa para decidir qual oferta recomendar.
        encoder = BankContextEncoder(df_clean_bank)
        print(f"[LinUCB] dimensão do vetor de contexto: {encoder.dim}")
        print(f"[LinUCB] features: {encoder.feature_names()}\n")

        policy = LinUCBPolicy(id_bracos, dim=encoder.dim)

        with mlflow.start_run(run_name=policy.name()):
            log_dataset(df_clean_bank, source_path=str(self.default_bank_path), name="clean_bank")

            # Roda a simulação com a política LinUCB, passando o encoder para que ela possa usar o contexto do cliente.
            rng = np.random.default_rng(self.seed) # Número aleatório para reprodutibilidade
            metrics = run_simulation(policy, df_clean_bank, ofertas, rng, encoder=encoder)
            log_policy_metrics(
                policy.name(),
                {"alpha": policy.alpha, "seed": self.seed, "context_dim": encoder.dim},
                metrics,
            )

            # 5. Salva o resumo (reward médio, regret, etc.) e o histórico completo (linha a linha)
            pd.DataFrame([metrics.summary(policy.name())]).to_csv(
                self.output_dir / "metrics_summary.csv", index=False
            )
            pd.DataFrame([{"policy": policy.name(), **linha} for linha in metrics.history]).to_parquet(
                self.output_dir / "metrics_timeseries.parquet", index=False
            )

            # Monta e salva a contagem de quantas vezes cada braço foi escolhido,
            # ordenado do mais escolhido para o menos escolhido
            arm_df = pd.DataFrame(
                [
                    {"policy": policy.name(), "arm_id": arm_id, "count": count}
                    for arm_id, count in metrics.arm_counts.items()
                ]
            ).sort_values("count", ascending=False)
            arm_df.to_csv(self.output_dir / f"arm_counts_{policy.name()}.csv", index=False)

            # Resumo final do experimento
            print(pd.DataFrame([metrics.summary(policy.name())]).to_string(index=False))

            print("\n=== Braços mais escolhidos (LinUCB) ===")
            print(arm_df.to_string(index=False))

            # Com a LinUCBPolicy já treinada (o loop de simulação atualizou seus pesos
            # internos a cada impressão), descobre qual braço ela recomenda
            # predominantemente para cada perfil de cliente
            self._analisar_perfil_linucb(policy, df_clean_bank, ofertas, encoder)

            # Todos os CSV/parquet acima (summary, timeseries, arm_counts, arm_by_*)
            # já estão em self.output_dir neste ponto — anexa tudo como artifacts do run.
            mlflow.log_artifacts(str(self.output_dir))

    def _analisar_perfil_linucb(self, policy, df, offers, encoder):
        """Usa a LinUCBPolicy já treinada para inferir o braço recomendado por perfil de cliente.

        Para cada cliente da base, pergunta à política treinada: "dado o contexto
        deste cliente, qual oferta você recomendaria agora?". Depois, agrupa essas
        recomendações por job, education, poutcome e faixa etária, para entender
        se o modelo aprendeu a diferenciar o público-alvo de cada oferta.
        """
        registros = []

        for _, client in df.iterrows():
            client_dict = client.to_dict()

            # Só considera ofertas para as quais o cliente é elegível
            # (ex: refinanciamento imobiliário exige que o cliente já tenha financiamento)
            eligible = get_eligible_offers(client_dict, offers)
            if not eligible:
                continue

            eligible_ids = [oferta["id_braco"] for oferta in eligible]

            # Transforma o cliente em vetor de contexto e pergunta à política
            # qual braço ela escolheria para esse cliente
            x = encoder.encode(client_dict)
            braco_escolhido = policy.select_arm(eligible_ids, context=x)

            registros.append(
                {
                    "arm_id": braco_escolhido,
                    "job": client_dict.get("job", "?"),
                    "education": client_dict.get("education", "?"),
                    "poutcome": client_dict.get("poutcome", "?"),
                    "age": float(client_dict.get("age", 0)),
                }
            )

        df_recomendacoes = pd.DataFrame(registros)

        # Agrupa a idade em faixas, para facilitar a leitura do relatório
        df_recomendacoes["age_group"] = pd.cut(
            df_recomendacoes["age"],
            bins=[0, 30, 40, 50, 60, 200],
            labels=["<30", "30-40", "40-50", "50-60", "60+"],
        )

        print("\n=== LinUCB — Braço predominante por perfil de cliente ===")

        # Colunas pelas quais vamos agrupar os clientes, uma de cada vez,
        # para descobrir qual braço a LinUCBPolicy recomenda mais para cada grupo.
        colunas_de_perfil = ("job", "education", "poutcome", "age_group")

        for coluna in colunas_de_perfil:
            # Conta, para cada valor da coluna (ex: cada profissão), quantas vezes
            # cada braço foi recomendado
            pivot = df_recomendacoes.groupby([coluna, "arm_id"], observed=True).size().unstack(fill_value=0)

            # Identifica o braço mais recomendado para cada valor da coluna
            pivot["predominante"] = pivot.idxmax(axis=1)
            pivot["total"] = pivot.drop(columns="predominante").sum(axis=1)

            # Imprime no terminal, para conferência rápida
            print(f"\n[Por {coluna}]")
            print(pivot[["predominante", "total"]].to_string())

            # Salva a tabela completa em CSV, para que o LLM possa ler.
            # O índice do pivot é o próprio valor da coluna (ex: cada profissão),
            nome_do_arquivo = f"arm_by_{coluna}.csv"
            caminho_do_arquivo = self.output_dir / nome_do_arquivo
            pivot.to_csv(caminho_do_arquivo, index=True)

            print(f"[LinUCB] tabela '{coluna}' salva em: {caminho_do_arquivo}")


if __name__ == "__main__":
    simulator = LinUCBSimulator()
    simulator.run_linucb()
