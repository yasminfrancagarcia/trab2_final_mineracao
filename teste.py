# ==============================================================================
# CÉLULA 1: Importação das Bibliotecas
# ==============================================================================
from io import StringIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold, cross_val_score, cross_validate
)
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, classification_report, ConfusionMatrixDisplay
)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# ==============================================================================
# CÉLULA 2: Carregamento do Dataset Principal
# ==============================================================================
caminho = "https://www-usr.inf.ufsm.br/~joaquim/UFSM/DM/ds/AgeOfMythology_work.csv"
df_raw = pd.read_csv(caminho, dtype="string", keep_default_na=False, encoding="utf-8")

# ==============================================================================
# CÉLULA 3: Função de Limpeza e Engenharia de Atributos
# ==============================================================================
RECURSOS = {
    "F": "cost_food",
    "W": "cost_wood",
    "G": "cost_gold",
    "Fv": "cost_favor",
}

COLUNAS_NUMERICAS = [
    "age", "pop", "dmg_hack", "dmg_pierce", "dmg_crush",
    "range", "hp", "armor_hack", "armor_pierce",
    "armor_crush", "speed", "los", "train_time",
]

def extrair_custos(valor):
    if pd.isna(valor) or valor in ("", "--"):
        return {nome: 0 for nome in RECURSOS.values()}

    custos = {nome: 0 for nome in RECURSOS.values()}
    # Ajusta possíveis inconsistências de vírgula/espaço
    itens = valor.replace(",", " ").split()
    for item in itens:
        recurso = next((r for r in RECURSOS if item.endswith(r)), None)
        if recurso is not None:
            qtd = item.removesuffix(recurso)
            if qtd.isdigit():
                custos[RECURSOS[recurso]] += int(qtd)
    return custos

def limpar_dataset(df_in) -> pd.DataFrame:
    df = df_in.copy()

    # Limpeza de nomes de colunas e espaços extras
    df.columns = [" ".join(col.replace("\u00a0", " ").split()) for col in df.columns]
    for coluna in df.columns:
        df[coluna] = df[coluna].map(
            lambda x: " ".join(x.replace("\u00a0", " ").split()) if isinstance(x, str) else x
        )
    df = df.replace("", pd.NA)

    # Renomear colunas
    df = df.rename(columns={
        "Unit": "unit", "Age": "age", "Cost": "cost_raw", "Pop": "pop",
        "DmgHack": "dmg_hack", "DmgPierce": "dmg_pierce", "DmgCrush": "dmg_crush",
        "Range": "range", "HP": "hp", "ArmorHack": "armor_hack",
        "ArmorPierce": "armor_pierce", "ArmorCrush": "armor_crush",
        "Speed": "speed", "LOS": "los", "Train time": "train_time",
        "Bonus": "bonus", "Other": "other", "Class": "unit_class"
    })

    # Correção de ortografia nas classes
    for coluna in ["unit_class", "bonus"]:
        if coluna in df.columns:
            df[coluna] = df[coluna].str.replace("Infantary", "Infantry", regex=False)

    # Extração de custos individuais
    custos = df["cost_raw"].apply(extrair_custos)
    custos_df = pd.DataFrame(custos.tolist(), index=df.index)
    for col in custos_df.columns:
        df[col] = custos_df[col].astype("Int64")

    # Tratamento de caracteres especiais
    for col in ["dmg_hack", "dmg_pierce", "dmg_crush"]:
        df[col] = df[col].str.replace("*", "", regex=False)

    df["hp"] = df["hp"].str.removesuffix("_")

    for col in ["armor_hack", "armor_pierce", "armor_crush"]:
        df[col] = df[col].str.removesuffix("%")

    # Flag de ataque
    colunas_dmg = ["dmg_hack", "dmg_pierce", "dmg_crush"]
    df["does_attack"] = df[colunas_dmg].ne("--").any(axis=1).astype(int)

    # Conversão para tipos numéricos
    for col in COLUNAS_NUMERICAS:
        substituicoes = {
            "--": 0 if col.startswith("dmg_") else pd.NA,
            "none": pd.NA,
        }
        df[col] = pd.to_numeric(df[col].replace(substituicoes), errors="coerce")

    # Formatação final de inteiros e decimais
    colunas_inteiras = ["age", "pop", "range", "hp", "armor_hack", "armor_pierce", "armor_crush", "los", "train_time"]
    colunas_decimais = ["dmg_hack", "dmg_pierce", "dmg_crush", "speed"]

    df[colunas_inteiras] = df[colunas_inteiras].astype("Int64")
    df[colunas_decimais] = df[colunas_decimais].astype("Float64")

    return df.drop_duplicates().reset_index(drop=True)

df_limpo = limpar_dataset(df_raw)

# ==============================================================================
# CÉLULA 4: Preparação das Features de Treino
# ==============================================================================
# Inclusão dos custos adicionais para aumentar o poder preditivo do modelo
ATRIBUTOS = [
    "age", "pop", "dmg_hack", "dmg_pierce", "dmg_crush", "range", "hp",
    "armor_hack", "armor_pierce", "armor_crush", "speed", "train_time",
    "cost_favor", "cost_food", "cost_wood", "cost_gold", "does_attack"
]

X = df_limpo[ATRIBUTOS].fillna(0).astype(float)
y = df_limpo["unit_class"]

# ==============================================================================
# CÉLULA 5: Divisão Treino/Teste e Comparação de Modelos (Decision Tree vs Random Forest)
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 1. Árvore de Decisão Simples
arvore = DecisionTreeClassifier(criterion="entropy", random_state=42)
arvore.fit(X_train, y_train)
y_pred_dt = arvore.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)

# 2. Random Forest (Modelo Ensembled)
floresta = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42)
floresta.fit(X_train, y_train)
y_pred_rf = floresta.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print(f"Acurácia no teste (Árvore de Decisão): {acc_dt * 100:.2f}%")
print(f"Acurácia no teste (Random Forest):      {acc_rf * 100:.2f}%")

# ==============================================================================
# CÉLULA 6: Avaliação por Validação Cruzada (10-Fold CV)
# ==============================================================================
kf = KFold(n_splits=10, shuffle=True, random_state=42)

cv_dt = cross_val_score(arvore, X, y, cv=kf, scoring="accuracy").mean()
cv_rf = cross_val_score(floresta, X, y, cv=kf, scoring="accuracy").mean()

print(f"Acurácia Média 10-Fold CV (Decision Tree): {cv_dt * 100:.2f}%")
print(f"Acurácia Média 10-Fold CV (Random Forest): {cv_rf * 100:.2f}%")

# ==============================================================================
# CÉLULA 7: Predição do Conjunto Sem Classe (Teste Cego)
# ==============================================================================
caminho_noClass = "https://www-usr.inf.ufsm.br/~joaquim/UFSM/DM/ds/AgeOfMythology_noClass.csv"
teste_noclass = pd.read_csv(caminho_noClass, encoding="utf-8-sig")

mapa_cols = {
    "Age": "age", "Pop": "pop", "DmgHack": "dmg_hack", "DmgPierce": "dmg_pierce",
    "DmgCrush": "dmg_crush", "Range": "range", "HP": "hp", "ArmorHack": "armor_hack",
    "ArmorPierce": "armor_pierce", "ArmorCrush": "armor_crush", "Speed": "speed",
    "Train time": "train_time", "Favor": "cost_favor"
}

teste_noclass = teste_noclass.rename(columns=mapa_cols)

# Adiciona colunas faltantes no dataset sem classe preenchendo com 0
for col in ATRIBUTOS:
    if col not in teste_noclass.columns:
        if col == "does_attack":
            teste_noclass["does_attack"] = (teste_noclass[["dmg_hack", "dmg_pierce", "dmg_crush"]].sum(axis=1) > 0).astype(int)
        else:
            teste_noclass[col] = 0

X_sem_classe = teste_noclass[ATRIBUTOS].fillna(0).astype(float)

# Treinar o modelo de Random Forest em TODO o dataset e realizar as predições
modelo_final = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42)
modelo_final.fit(X, y)

predicoes = modelo_final.predict(X_sem_classe)
teste_noclass["Classe_Predita"] = predicoes

teste_noclass[["Classe_Predita"] + ATRIBUTOS]

print(predicoes)