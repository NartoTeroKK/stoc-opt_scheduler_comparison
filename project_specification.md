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

## My Specs on the project

### Concetti Teorici (NotebookLM)

Per implementare un progetto di questo tipo, i documenti forniti non contengono dettagli specifici sugli scheduler avanzati che hai menzionato (come Cosine Annealing, Cyclical Learning Rate, One-Cycle policy o decadimento esponenziale). Tuttavia, **le fonti offrono un'ampia base teorica sui principi del learning rate, sul comportamento di SGD e Adam, e sulle dinamiche dell'ottimizzazione convessa e non convessa**. 

Di seguito sono riportati gli aspetti teorici più importanti da considerare e analizzare nel tuo progetto, supportati dalle fonti a disposizione:

#### 1. Ruolo del Learning Rate e Dinamiche di Convergenza (Teoria di Base)
Il motivo per cui si utilizzano gli scheduler è legato alla natura stessa del gradiente stocastico:
*   **Rumore e passo costante:** Nei metodi stocastici, il calcolo del gradiente contiene rumore. Se si utilizza un learning rate (passo) costante, l'algoritmo non converge esattamente al minimo, ma raggiunge una "zona stazionaria" in cui oscilla; l'errore asintotico è proporzionale al learning rate $\eta$. Un passo grande garantisce una discesa rapida iniziale ma oscillazioni finali ampie, mentre un passo piccolo è più stabile ma molto più lento.
*   **Condizioni di convergenza esatta:** Per ottenere una convergenza esatta al minimo (azzerando le oscillazioni), la teoria richiede che il learning rate decresca nel tempo. Questo principio è formalizzato dalle **condizioni di Robbins-Monro**, le quali richiedono che i passi non si annullino troppo presto ($\sum \eta_k = \infty$) ma che il loro effetto sul rumore resti limitato nel tempo ($\sum \eta_k^2 < \infty$). Un corretto scheduling serve proprio a bilanciare l'esplorazione veloce iniziale con la stabilità finale.

#### 2. Interazione tra Scheduler e Metodo di Base (SGD vs Adam)
L'interazione tra la politica di scheduling e il metodo di ottimizzazione scelto è un punto cruciale di analisi:
*   **SGD (Stochastic Gradient Descent):** In SGD, il learning rate $\eta$ è globale e viene applicato indistintamente a tutte le componenti del vettore dei parametri. SGD è **fortemente sensibile** alla scelta del learning rate: un $\eta$ troppo grande porta a instabilità o divergenza, mentre uno troppo piccolo blocca l'apprendimento. Applicare uno scheduler a SGD modificherà direttamente l'ampiezza dei passi lungo tutte le direzioni.
*   **Adam (Adaptive Moment Estimation):** A differenza di SGD, Adam calcola un passo **adattivo per singola componente** utilizzando una stima del momento primo (media dei gradienti) e del momento secondo (varianza non centrata). L'aggiornamento reale dei pesi in Adam dipende sì dal learning rate di base $\eta$, ma questo viene diviso per $\sqrt{\hat{v}_k} + \epsilon$. Grazie a questo fattore di normalizzazione locale, Adam risulta **molto più robusto e tollerante** rispetto alla scelta di $\eta$ globale. 
*   **Confronto Teorico:** Nelle tue analisi, dovresti notare che l'impatto di un decremento aggressivo del learning rate (es. Exponential decay) potrebbe essere molto più limitante per SGD che per Adam. Adam infatti compensa dinamicamente le scalature e possiede già una forma di adattamento interno, riducendo la necessità di un "tuning fine" dell'iperparametro globale $\eta$ rispetto a SGD.

#### 3. Applicazione: Convesso vs Non Convesso
Le differenze teoriche tra i due scenari applicativi da te proposti sono marcate:
*   **Problemi Convessi (es. Least Squares):** La funzione di perdita è un paraboloide con un unico minimo globale globale. In questo scenario, non c'è il rischio di rimanere intrappolati in minimi locali. L'analisi si concentrerà puramente sulla velocità di convergenza (quante iterazioni servono per arrivare alla precisione desiderata) e sullo smorzamento delle oscillazioni finali grazie allo scheduler.
*   **Reti Neurali (Non Convesse, es. MNIST):** Le funzioni di costo (come la classificazione con layer multipli e attivazioni non lineari) creano un paesaggio "accidentato" e altamente non convesso. In questo caso, **il rumore del gradiente stocastico è utile**, in quanto favorisce l'esplorazione e permette di sfuggire a minimi locali e punti di sella. Utilizzare strategie come i cicli (Cyclical LR o Cosine Annealing, anche se non trattati esplicitamente nelle fonti) aiuta teoricamente a mantenere un livello di rumore benefico per saltare fuori dai minimi locali poveri prima di stabilizzarsi.

#### 4. Stabilità del Training e Varianza
Per analizzare la stabilità delle curve di loss, devi considerare la **varianza del gradiente**. La teoria indica che:
*   La varianza del gradiente controlla la dimensione delle oscillazioni del training.
*   Un fattore che modulerà fortemente la stabilità, insieme allo scheduler, è l'utilizzo dei **mini-batch**. L'impiego di un batch di dimensione $B$ riduce la varianza del gradiente approssimativamente di un fattore $1/B$. 
*   Di conseguenza, quando analizzerai la curva della loss nel tempo, ricordati che una traiettoria fluida non dipenderà solo da quanto lo scheduler abbassa $\eta$, ma anche dall'uso di mini-batch che "puliscono" intrinsecamente il segnale di discesa.

*(Nota per la ricerca indipendente: Per le formule esatte matematiche e le implementazioni di One-Cycle policy, Cosine Annealing e Cyclical Learning Rate, dovrai fare riferimento a letteratura esterna ai documenti finora analizzati).*

## Report di Pianificazione

### Versione 0 — Setup e Baseline

**Obiettivo:** Costruire l'infrastruttura di base e stabilire le metriche di riferimento.

- Configurare l'ambiente (PyTorch o NumPy puro, logging, seed riproducibili)
- Implementare SGD e Adam *senza* scheduler (learning rate costante)
- Applicare entrambi al problema convesso: **regressione logistica su un dataset sintetico o su MNIST binarizzato**
- Tracciare loss e accuracy per ogni epoch come baseline
- Definire le metriche di valutazione: velocità di convergenza (epoch a soglia), loss finale, stabilità (varianza delle ultime N loss)

> Questa versione risponde alla domanda: *"Quanto guadagnamo rispetto al non usare nessuno scheduler?"*

### Versione 1 — Problema Convesso + Scheduler Semplici

**Obiettivo:** Introdurre i due scheduler più intuitivi sul problema convesso.

- Implementare **Exponential Decay**: $\alpha_t = \alpha_0 \cdot \gamma^t$
- Implementare **Cosine Annealing**: $\alpha_t = \alpha_{min} + \frac{1}{2}(\alpha_{max} - \alpha_{min})\left(1 + \cos\frac{t\pi}{T}\right)$
- Testare entrambi con SGD e Adam su regressione logistica
- Produrre grafici comparativi: curva del learning rate nel tempo, curva della loss, curva dell'accuracy
- Analisi parametrica: variare $\gamma$ per Exponential Decay e $T_{max}$ per Cosine Annealing


### Versione 2 — Problema Convesso + Scheduler Ciclici

**Obiettivo:** Aggiungere i due scheduler con comportamento ciclico.

- Implementare **Cyclical Learning Rate (CLR)** con le policy *triangular* e *triangular2*
- Implementare **One-Cycle Policy** (fase warmup + fase discesa con momentum inverso)
- Testare su regressione logistica con SGD e Adam
- Confronto diretto con i risultati della V1: tabella riassuntiva con velocità di convergenza e loss finale per ogni combinazione scheduler × optimizer
- Discussione teorica: perché i metodi ciclici possono sfuggire ai minimi piatti in problemi convessi?


### Versione 3 — Problema Non Convesso: Rete Neurale su MNIST/Fashion-MNIST

**Obiettivo:** Estendere tutti gli scheduler a un setting non convesso.

- Definire un'architettura MLP semplice (es. 2–3 layer hidden, ReLU)
- Rieseguire tutti e 4 gli scheduler × 2 optimizer sulla nuova architettura
- Monitorare: train loss, validation loss, validation accuracy, gradient norm
- Analisi della **stabilità**: evidenziare oscillazioni, overfitting precoce, o slow start
- Confronto con le baseline (V0) sullo stesso problema



### Versione 4 — Analisi Sistematica e Sensitivity Analysis

**Obiettivo:** Variazione sistematica dei parametri chiave per ogni scheduler.

Per ogni scheduler, isolare e variare un parametro alla volta mantenendo fissi gli altri:

- Exponential Decay: $\gamma \in \{0.99, 0.95, 0.90, 0.80\}$
- Cosine Annealing: $T_{max} \in \{10, 25, 50, 100\}$ e restart con **SGDR**
- CLR: step size, $\alpha_{min}$, $\alpha_{max}$
- One-Cycle: percentuale di warmup, $\alpha_{max}$

Produrre heatmap o curve multiple sovrapposte per visualizzare l'effetto di ogni iperparametro.


### Versione 5 — Report Finale e Presentazione

**Obiettivo:** Consolidare tutti i risultati in documentazione accademica.

Il report scritto deve includere:

- Descrizione teorica di ogni scheduler con formule, intuizione geometrica e pseudocodice
- Sezione metodologica: setup sperimentale, dataset, architetture, metriche
- Risultati V1→V4 con figure e tabelle
- Discussione: interazione tra scheduler e optimizer (es. Adam + One-Cycle è ridondante?)
- Conclusioni: raccomandazioni pratiche su quale strategia usare e quando

La presentazione finale può seguire questa struttura: motivazione (5%) → teoria (25%) → esperimenti (40%) → analisi e conclusioni (30%).

***
