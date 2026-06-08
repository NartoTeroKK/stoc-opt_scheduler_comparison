# Ottimizzazione stocastica con scheduler avanzati: confronto tra strategie di learning rate

## Academic Course "Metodi di Ottimizzazione Stocastici" Specs

### Obiettivo
Implementare e confrontare le seguenti strategie di scheduling:
- Cosine Annealing,
- Cyclical Learning Rate,
- One-Cycle policy,
- Exponential decay.

Utilizzare come metodo di base sia SGD che Adam.

### Metodologia

Applicazione consigliata: iniziare con un problema convesso (regressione logistica o least squares) e successivamente considerare un problema non convesso (rete neurale su MNIST o Fashion-MNIST).

### Analisi dei risultati richieste

- analisi della velocità di convergenza,
- stabilità del training, con comparazione delle curve del learning rate e della loss nel tempo.
- Interazione tra scheduler e metodo di base.

### Materiale documentale da produrre

Ogni tesina deve includere:

- una descrizione teorica dell’approccio scelto, con formule e pseudocodice;
- applicazione dell’approccio a un problema di interesse;
- un’analisi sistematica delle prestazioni con variazione di parametri chiave;
- un report scritto e una presentazione finale.