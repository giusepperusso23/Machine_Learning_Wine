# Machine_Learning_Wine
Immaginando di collaborare con un’azienda che si occupa di analisi e certificazione della qualità di vino prodotto a basso impatto ambientale, ho sviluppato un modello predittivo che potrà supportare i processi di controllo qualità, contribuendo a identificare in modo automatico la provenienza e le caratteristiche dei vini, e garantendo standard elevati lungo tutta la filiera.

Il dataset “Wine” di scikit-learn include i risultati delle analisi chimiche di vini provenienti da tre diversi produttori italiani. 
Il mio compito è stato determinare il target, ovvero il produttore del vino basandomi sui suoi valori chimici.

1. SETUP INIZIALE e CARICAMENTO DATI

2. ANALISI ESPLORATIVA dei DATI (EDA):
- 2.1. Distribuzione univariata
   - Istogrammi e boxplot per i dati generali
   - Boxplot per ogni singola feature divisa per categoria nel 'target'
   - Osservazioni outliers
- 2.2. Distribuzione bivariata
     - Heatmap per ciascun produttore (categoria in 'target')
     - Pairplot per coppia di variabili con distinzione di produttore
     - Calcolo di un indice quantitativo di separabilità tra produttori di vino (per un dataset con etichette di classe) analizzando tutte le coppie di feature numeriche.
- 2.3. Distribuzione multivariata
     - Distribuzione e visualizzaione della PCA
     - Visualizzazione del contributo ciascuna variabile
    
3. GESTIONE degli OUTLIERS

   Dopo aver mantenuto una copia originale del dataset, sono state testate due diverse strategie di gestione degli outlier:
   - Rimozione tramite IQR (Interquartile Range).
     
   - Capping/Flooring tramite quantili (Cap-Floor). 
 
4. PREPARAZIONE dei DATI per la modellazione (Split dataset)

  Divisione dei 3 database in train e test set per garantire una valutazione affidabile del modello, con random_state=42 e stratify=y.

5. TRAINING

  Per la fase di classificazione è stato scelto un modello di Regressione Logistica Multiclasse. La pipeline completa di modellazione è:
    StandardScaler → PCA → Regressione Logistica, includendo la ricerca degli iperparametri tramite GridSearchCV.
  Gli iperparametri da validare tramite cross-validation standard sono il numero di componenti principali della PCA (pca__n_components) e il coefficiente di regolarizzazione della regressione logistica (clf__C), che controlla la complessità del modello.
 
6. VALUTAZIONE sul TEST
   
  Valutazione del modello ottimizzato (ottenuto da GridSearchCV) sul set di test, stampando le metriche principali e, opzionalmente, la matrice di confusione.
  La test accuracy risulta 1.00 su tutti e 3 idataset

7. PROVA con DIVERSO RANDOM_STATE

   - Split con random_state=2028 e senza stratify settato. Dopo tarining e test l'accuratezza resta 1.00 solo sul dataset "no_outliers"
  
   - Test di una lista di random_state che fa split sul daset "no_outliers". Ricerca di acuratezza media dopo training e test.
  
   - Test di una lista di random_state che fa split sul daset "capped". Ricerca di acuratezza media dopo training e test.
  
   - Test di una lista di random_state che fa split sul daset originale. Ricerca di acuratezza media dopo training e test.

8. DECISIONE e CONSIDERAZIONI
   
   - Scelta modello più appropriato e analisi sue componenti PCA
  
   - Confronto tra logistic regresion:
                             - con PCA con validazione numero parametri
                             - con PCA con percentuale varianza
                             - senza PCA
