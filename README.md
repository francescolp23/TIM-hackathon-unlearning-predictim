# Machine Unlearning — Hackathon TIM x Sapienza

Pipeline di unlearning per il modello DynamicMLP fornito: smorzamento
selettivo dei pesi guidato dalla Fisher Information + riparazione sul
retain set con loss non pesata.

## Metodo

1. **Split**: forget (9.085 utenti da `forget_data.csv`), retain, validation
   e test ricavati dal pool con seed fisso; overlap forget-validation
   verificato a zero via assert.
2. **Smorzamento (SSD-style)**: Fisher diagonale calcolata separatamente su
   forget e su un sottocampione del retain (16k righe, per efficienza);
   i pesi con rapporto forget/retain > 1 vengono smorzati
   proporzionalmente (~2.400 parametri su 45k).
3. **Riparazione**: 4 epoche di fine-tuning sul retain con
   `BCEWithLogitsLoss()` **senza** pos_weight. Oltre a ripristinare
   l'utility dopo lo smorzamento, la loss non pesata ricalibra le
   probabilità distorte dai pesi di classe del training originale
   (che gonfiavano le label rare nel ranking), migliorando la
   Precision@10.
4. **Verifica**: P@10 su validation, MIA loss-based (forget vs test
   held-out, AUC ~0.5) e relearn time (>60 passi per ri-raggiungere la
   loss del modello base sul forget → oblio profondo, non soppressione).

## Risultati (leaderboard)

| Metrica | Valore |
|---|---|
| Precision@10 | 0.719 |
| MIA resistance | 0.996 |
| Tempo unlearning | ~10s |
| **Score totale** | **86.2** |

## Riproduzione

```bash
pip install -r requirements.txt
python main.py
```

Genera i tre file di submission: `model_artifact`, `execution_time.txt`,
`validation_ids.csv`. Seed fissato (42): l'esecuzione è deterministica.
