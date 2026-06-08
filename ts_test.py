import numpy as np
import matplotlib.pyplot as plt

# Configurações do experimento
n_tentativas = 1000
n_braços = 3
taxas_reais = [0.12, 0.25, 0.60]

sucessos = np.ones(n_braços)
fracassos = np.ones(n_braços)
historico_escolhas = []

# Execução do Thompson Sampling
for jogada in range(n_tentativas):
    amostras = [np.random.beta(sucessos[i], fracassos[i]) for i in range(n_braços)]
    braço_escolhido = np.argmax(amostras)
    historico_escolhas.append(braço_escolhido)
    
    recompensa = 1 if np.random.rand() < taxas_reais[braço_escolhido] else 0
    if recompensa == 1:
        sucessos[braço_escolhido] += 1
    else:
        fracassos[braço_escolhido] += 1

# Contagem de escolhas por braço
escolhas, contagens = np.unique(historico_escolhas, return_counts=True)
contagens_totais = {0: 0, 1: 0, 2: 0}
for e, c in zip(escolhas, contagens):
    contagens_totais[e] = c

# Ordenando conforme a regra de gráficos de barra (da maior frequência para a menor)
dados_ordenados = sorted(contagens_totais.items(), key=lambda x: x[1], reverse=True)
braços_ord = [f"Braço {item[0]}\n(Taxa: {taxas_reais[item[0]]*100:.0f}%)" for item in dados_ordenados]
valores_ord = [item[1] for item in dados_ordenados]

# Criando os gráficos em subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Barras com total de escolhas (ordenado)
ax1.bar(braços_ord, valores_ord, color=['#2ca02c', '#1f77b4', '#ff7f0e'], edgecolor='black', alpha=0.8)
ax1.set_title('Total de Escolhas por Braço (Ordenado)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Número de Vezes Escolhido', fontsize=11)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Adicionando rótulos com os valores exatos nas barras
for i, v in enumerate(valores_ord):
    ax1.text(i, v + 15, str(v), ha='center', fontweight='bold')

# Gráfico 2: Evolução das escolhas ao longo do tempo (Histórico cumulativo)
historico_escolhas = np.array(historico_escolhas)
for i in range(n_braços):
    escolhas_acumuladas = np.cumsum(historico_escolhas == i)
    ax2.plot(escolhas_acumuladas, label=f'Braço {i} ({taxas_reais[i]*100:.0f}%)', linewidth=2)

ax2.set_title('Evolução Cumulativa das Escolhas', fontsize=12, fontweight='bold')
ax2.set_xlabel('Tentativas (Iterações)', fontsize=11)
ax2.set_ylabel('Total Acumulado de Escolhas', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper left')

plt.tight_layout()
plt.savefig('distribuicao_escolhas_thompson.png', dpi=300)