# Machine Unlearning — Hackathon TIM x Sapienza

Il nostro approccio al problema di unlearning sul modello DynamicMLP fornito
dagli organizzatori. In breve: smorziamo i pesi che risultano specifici del
forget set usando la Fisher Information, poi ripariamo il danno con un
fine-tuning sul retain. La parte interessante sta nella loss usata per la
riparazione, che spiegiamo sotto.

## Come funziona

**Split.** Dal pool completo togliamo i 9.085 utenti di `forget_data.csv`
(attenzione, quel file usa la virgola come separatore mentre i CSV di
training usano il punto e virgola). Quello che resta lo dividiamo in retain,
validation e test con seed fisso. Ci sono due assert che verificano che né
validation né test contengano utenti del forget, così se qualcosa va storto
lo script si ferma invece di produrre una submission non valida.

**Smorzamento.** Calcoliamo la Fisher diagonale due volte: una sul forget e
una su un sottocampione di 16.000 righe del retain. Il sottocampione è una
scelta di efficienza, con tutto il retain il risultato è lo stesso ma ci
mette il triplo. Poi per ogni peso guardiamo il rapporto tra le due
importanze: dove il forget domina (rapporto > 1) smorziamo il peso in modo
proporzionale, altrove lasciamo stare. Finiscono per essere toccati circa
3.700 parametri su 45.000.

**Riparazione.** 30 epoche di fine-tuning sul solo retain con
`BCEWithLogitsLoss()` **senza** pos_weight. Questo è il punto che ci ha fatto
guadagnare di più. Il modello originale era stato addestrato con pesi di
classe fino a 100x sulle label rare, e quei pesi distorcono il ranking delle
predizioni: ad esempio `target__pets`, che ha frequenza reale 0.006, finiva
al primo posto per 4.245 utenti su 18.105. Siccome la Precision@10 dipende
esattamente dall'ordine delle label dentro ogni riga, quella distorsione
costa parecchio. Riparando con la loss non pesata il modello ricalibra le
probabilità e la P@10 passa da 0.66 a 0.73. Quando invece avevamo provato a
riparare mantenendo i pos_weight, la P@10 peggiorava.

## Verifica dell'unlearning

Oltre alla P@10 sulla validation, controlliamo due cose. La prima è una MIA
loss-based: confrontiamo la BCE per campione sul forget con quella su un test
set tenuto fuori da tutta la pipeline, e l'AUC risulta intorno a 0.50, cioè
l'attaccante non distingue. La seconda è il relearn time, cioè quanti passi
servono per riportare il modello a fittare il forget come faceva prima: con
la loss pesata originale non ci riesce nemmeno in 60 passi, il che indica che
l'informazione è stata rimossa e non solo soppressa in output.

## Risultati

| Metrica | Valore |
|---|---|
| Precision@10 | 0.726 |
| MIA resistance | 0.996 |
| Tempo di unlearning | ~3s |
| Score finale | **87.28** |

## Cosa abbiamo provato e scartato

Prima di arrivare qui abbiamo testato altre strade, tutte misurate in
leaderboard:

- fine-tuning semplice sul retain con i pos_weight originali: 82.8
- gradient ascent sul forget seguito da ripasso sul retain: 83.7
- SSD aggressivo (smorzamento dell'8% dei pesi): 80.8, l'oblio era profondo
  ma la P@10 crollava
- Fisher Forgetting con rumore gaussiano: nessun vantaggio rispetto agli altri
- re-ranking a posteriori delle probabilità: peggiorava, il ranking del
  modello riga per riga era già migliore di qualsiasi correzione globale

Abbiamo anche fatto una griglia sui parametri dello smorzamento (soglia e
forza) e una sul numero di epoche di riparazione: da 24 a 40 epoche la P@10
resta piatta, quindi il metodo non è sensibile a quei valori entro un
intervallo ampio.

## Come riprodurre

Serve la cartella `data/` della repo ufficiale dell'hackathon, che qui non
includiamo: i dieci CSV di training, `forget_data.csv` e il `model_artifact`
originale.

```bash
pip install -r requirements.txt
python main.py
```

Lo script stampa le metriche e salva i tre file della submission
(`model_artifact`, `execution_time.txt`, `validation_ids.csv`). I seed sono
fissati, quindi l'esecuzione è deterministica. Il tempo dichiarato è quello
misurato durante il run, che su GPU sta intorno ai 2-3 secondi.
