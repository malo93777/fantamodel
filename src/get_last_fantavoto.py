import pandas as pd
import config
import utils

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def evaluate_index(
    df,
    index_col="Index",
    vote_col="Fantavoto preso",
    plot=True
):

    df = df.copy()

    # pulizia voti
    df = df[df[vote_col] != "s.v."]
    df[vote_col] = df[vote_col].astype(float)

    # ordina per index
    df = df.sort_values(index_col, ascending=False).reset_index(drop=True)

    def hit_rate(df, top_n, threshold=7):
        subset = df.head(top_n)
        hits = (subset[vote_col] >= threshold).sum()
        return hits / top_n

    def avg_vote(df, top_n):
        return df.head(top_n)[vote_col].mean()

    results = {}

    # hit rate
    results["hit_rate_top10"] = hit_rate(df, 10)
    results["hit_rate_top20"] = hit_rate(df, 20)
    results["hit_rate_top50"] = hit_rate(df, 50)

    # medie voto
    results["avg_vote_top10"] = avg_vote(df, 10)
    results["avg_vote_top20"] = avg_vote(df, 20)
    results["avg_vote_top50"] = avg_vote(df, 50)
    results["avg_vote_top150"] = avg_vote(df, min(150, len(df)))

    # spearman correlation
    corr, _ = spearmanr(df[index_col], df[vote_col])
    results["spearman_corr"] = corr

    # stampa risultati
    print("\n=== INDEX EVALUATION ===\n")

    print("Hit rate (>=7):")
    print(f"Top10  : {results['hit_rate_top10']:.2f}")
    print(f"Top20  : {results['hit_rate_top20']:.2f}")
    print(f"Top50  : {results['hit_rate_top50']:.2f}")

    print("\nFantavoto medio:")
    print(f"Top10  : {results['avg_vote_top10']:.2f}")
    print(f"Top20  : {results['avg_vote_top20']:.2f}")
    print(f"Top50  : {results['avg_vote_top50']:.2f}")
    print(f"Top150 : {results['avg_vote_top150']:.2f}")

    print(f"\nSpearman correlation: {results['spearman_corr']:.3f}")

    # plot
    if plot:
        plt.scatter(df[index_col], df[vote_col], alpha=0.4)
        plt.xlabel("Index")
        plt.ylabel("Fantavoto")
        plt.title("Index vs Fantavoto")
        plt.show()

    return results

def append_last_fantavoto(df_voti, df_index):
    """
    Prende l'ultimo fantavoto per (player_norm, fanta_role) da df_voti
    e lo concatena al dataset caricato da df_index.
    """

    # prendiamo ultima riga per player + role
    last_votes = (
        df_voti
        .sort_values("date")
        .groupby(["player_norm"])
        .tail(1)[["player_norm","fantavoto"]]
    )

    #sostituisco NaN con s.v. (caso giocatore non ha ancora voti)
    last_votes["fantavoto"] = last_votes["fantavoto"].fillna("s.v.")
    #rinomino colonna fantavoto in last_fantavoto
    last_votes = last_votes.rename(columns={"fantavoto": "Fantavoto preso"})

    df_index["Giocatore_norm"] = df_index["Giocatore"].apply(utils.normalize_fn)
    # merge
    df_index = df_index.merge(
        last_votes,
        left_on=["Giocatore_norm"],
        right_on=["player_norm"],
        how="left"
    )

    df_index = df_index.drop(columns=["player_norm", "Giocatore_norm", "Unnamed: 0"], errors="ignore")

    return df_index

if __name__ == "__main__":
    #chiedi a utente last giornata
    last_giornata = input("Inserisci l'ultima giornata: ")

    # carica dataset storico voti
    df_voti = pd.read_csv(config.DATASET_DATA_DIR / config.PROD_DATA_FILE_VOTI)

    roles = ["dif", "cc", "att"]

    is_model = True

    for role in roles:
        print(f"Processing role: {role}")
        if not is_model:
            index_df = pd.read_csv(config.DATASET_DATA_DIR /"dataset_index"/  (role + "_" + last_giornata + ".csv"))
        else:
            index_df = pd.read_csv(config.DATASET_DATA_DIR / "dataset_index" / f"{role.upper()}_2026-03-16.csv")
        
        if "Fantavoto preso" not in index_df.columns:
            # append ultimo fantavoto al dataset di input per il modello
            df_fv_last_giornata = append_last_fantavoto(df_voti, index_df)
        else:
            df_fv_last_giornata = index_df

        # salva dataset aggiornato
        if not is_model:
            df_fv_last_giornata.to_csv(config.DATASET_DATA_DIR /"dataset_index"/  (role + "_" + last_giornata + ".csv"), index=False)
        else:
            df_fv_last_giornata.to_csv(config.DATASET_DATA_DIR / "dataset_index" / f"{role.upper()}_2026-03-16.csv", index=False)
        # valutazione index
        print(f"Evaluating index for role: {role}")
        results=evaluate_index(df_fv_last_giornata, index_col="Index", vote_col="Fantavoto preso", plot=False)
        print(results)
        
        #salvare risultati in un file csv se is_model è True aggiungo _model al nome del file
        results_df = pd.DataFrame([results])
        if is_model:
            results_df.to_csv(config.DATASET_DATA_DIR /"dataset_index"/ "metriche" / (role + "_" + last_giornata + "_model_evaluation.csv"), index=False)
        else:
            results_df.to_csv(config.DATASET_DATA_DIR /"dataset_index"/ "metriche" / (role + "_" + last_giornata + "_evaluation.csv"), index=False)
