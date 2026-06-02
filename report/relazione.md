# Reti di Hopfield per il recupero di pattern logici LTLf

**Nome:** Nicola Ggante

**Matricola:** SM3201239

**Corso:** Introduzione al Machine Learning 

**Anno scolastico:** 2025/2026

## Abstract

Questo progetto studia l'uso delle reti di Hopfield classiche e delle Modern Hopfield Networks per il recupero associativo di tracce LTLf corrotte. L'obiettivo è verificare se pattern logicamente validi possano essere memorizzati come minimi energetici e recuperati a partire da versioni rumorose o inconsistenti. Il lavoro combina una pipeline di generazione dati basata su Spot e LTLf2DFA, un'implementazione classica in C e una moderna in Python, e una valutazione sperimentale su accuratezza, capacità di memoria, convergenza ed energia.

**Parole chiave:** Hopfield networks, Modern Hopfield Networks, LTLf, memoria associativa, energy-based models, neurosymbolic AI

## 1. Introduzione

### 1.1 Contesto del problema

Il problema di ricerca consiste nel verificare se una rete di memoria associativa possa correggere sequenze temporali finite corrotte, riportandole verso una configurazione logicamente valida.
Le formule LTLf costituiscono un linguaggio per descrivere processi temporali su tracce finite e permettono quindi di costruire un dataset di esempi validi e perturbati.
Le reti di Hopfield, sia classiche che moderne, rappresentano un candidato naturale per questo compito, poiché implementano una dinamica di minimizzazione energetica che conduce a stati stazionari interpretati come memorie associative.
In questo contesto, le tracce LTLf forniscono un dominio strutturato e logicamente interpretabile, rendendo possibile studiare se una rete energetica possa apprendere regolarità temporali direttamente dai dati.

### 1.2 Obiettivo del progetto

L'obbiettivo principale del progetto è verificare se una Modern Hopfield Network possa memorizzare tracce LTLf valide come minimi energetici e recuperare la configurazione corretta a partire da input corrotti.
In termini operativi, il lavoro mira a verificare se una rete energetica possa svolgere una funzione di correzione di errori naturale su sequenze logicamente strutturate.
A questo obbiettivo principale si affianca il confronto con la rete di Hopfield classica, al fine di analizzare differenze in capacità di stoccaggio, velocità di convergenza e tasso di recupero corretto.
Un ulteriore obiettivo del progetto è la costruzione di un dataset LTLf riutilizzabile anche nel lavoro di tesi, così da mantenere continuità tra la sperimentazione controllata del progetto e l'analisi su modelli più complessi.


### 1.3 Idea generale della soluzione

L'idea generale della soluzione consiste nel trasformare il problema logico in un problema di memoria associativa.
Partendo da formule LTLf generate automaticamente, viene costruito un dataset di tracce valide, successivamente perturbate tramite **bit-flip** casuali per simulare input corrotti.
Le tracce corrette vengono quindi memorizzate in due modelli distinti, una rete di Hopfield classica e una Modern Hopfield Network, che vengono testati sulla capacità di ricondurre gli input rumorosi verso la configurazione valida più vicina.
La valutazione della soluzione avviene attraverso una serie di esperimenti su accuratezza di recupero, capacità di stoccaggio, velocità di convergenza e andamento dell'energia, affiancati da visualizzazioni geometriche tramite PCA e t-SNE.
Questa impostazione permette di studiare il recupero di pattern logici sia dal punto di vista quantitativo, tramite misure sperimentali, sia dal punto di vista geometrico, tramite l'analisi dello spazio degli stati.

### 1.4 Struttura della relazione

La relazione è organizzata come segue.
Nel *Capitolo 2* vengono presentati i fondamenti teorici relativi alle reti di Hopfield classiche, alle Modern Hopfield Networks e al legame con il meccanismo di attention.
Il *Capitolo 3* descrive la costruzione del dataset LTLf, mentre il *Capitolo 4* illustra le scelte implementative adottate per i due modelli.
Nel *Capitolo 5* vengono riportati i test di sanità utilizzati per verificare la correttezza delle implementazioni prima della fase sperimentale.
Il *Capitolo 6* presenta gli esperimenti principali, seguiti nel *Capitolo 7* dalle visualizzazioni geometriche tramite **PCA** e **t-SNE**.
Infine, il *Capitolo 8* discute i risultati ottenuti e il *Capitolo 9* raccoglie le conclusioni e le possibili direzioni future del lavoro.
In appendice sono inoltre raccolti alcuni dettagli tecnici aggiuntivi relativi ai parametri sperimentali e alla struttura del progetto.

## 2. Fondamenti teorici

### 2.1 Reti di Hopfield classiche

La rete di Hopfield classica è un sistema di $N$ neuroni binari $\sigma_i \in \{\pm 1\}$ con connessioni simmetriche e senza auto-connessioni.
Lo stato del sistema è descritto da un vettore $\sigma_i \in \{\pm 1\}^N$ e l'intera dinamica è governata dalla minimizzazione della funzione di energia:

$$ E = - \frac{1}{2}\sum_{i \neq j} W_{ij}\sigma_i\sigma_j $$

I pattern memorizzati corrispondono ai minimi locali di questa funzione: il recupero di un pattern corrotto equivale quindi al rilassamento del sistema verso il minimo più vicino.
La memorizzazione avviene tramite la regola di Hebb, che fissa la matrice dei pesi come somma dei prodotti esterni del pattern da memorizzare:

$$ W_{ij} = \frac{1}{N} \sum_{\mu = 1}^p \xi_i^{\mu} \xi_j^{\mu} , \quad W_{ii} = 0 $$

Il recupero avviene tramite l'aggiornamento dei pesi in modo asincrono: a ogni passo viene aggiornato un singolo neurone secondo la regola $\sigma_i \leftarrow sgn \big(\sum_j W_{ij} \sigma_j \big)$, che garantisce la non crescita dell'energia.
La capacità di stoccaggio della rete è limitata: Hopfield (1982) ha dimostrato che una rete con $N$ neuroni può memorizzare correttamente fino a circa $0.138 \cdot N$ pattern prima che il recupero degeneri.
Questa limitazione sulla capacità di stoccaggio costituisce il principale motivo per cui le reti classiche vengono superate dalle MHN, trattate successivamente. 

### 2.2 Modello di Ising e interpretazione energetica

Il modello di Ising è un modello della meccanica statistica che descrive un reticolo di spin binari $s_i \in \{\pm 1\}$ che interagiscono a coppie.
La Hamiltoniana del sistema è:

$$ \mathcal{H} = -\frac{1}{2} \sum_{i \neq j} J_{ij} s_i s_j $$

dove $J_{ij}$ rappresenta la costante di accoppiamento tra gli spin $i$ e $j$.
L'isomorfismo con la rete di Hopfield è formale e preciso: i neuroni $\sigma_i$ corrispondono agli spin $s_i$, la matrice dei pesi $W_{ij}$ gioca il ruolo della Hamiltoniana di accoppiamento, e i pattern memorizzati corrispondono agli stati fondamentali del sistema, cioè ai minimi globali dell'energia.
Il recupero di un pattern corrotto corrisponde quindi al rilassamento termodinamico del sistema verso il minimo più vicino: partendo da uno stato iniziale perturbato, il sistema evolve riducendo monotonicamente la propria energia fino a raggiungere un attratore stabile.
Questo parallelismo non è solo descrittivo: l'implementazione della rete classica in `C` adottata in questo progetto, riflette direttamente questa natura, rendendo esplicito il legame tra peso sinaptico e costante di accoppiamento.
Le MHN, presentate nella sezione successiva, estendono questo framework sostituendo la funzione di energia quadratica con una versione continua, che consente una capacità di stoccaggio esponenzialmente maggiore.

### 2.3 Modern Hopfield Networks


Le Modern Hopfield Networks (Ramsauer et al., 2020) nascono dall'esigenza di superare il limite di capacità delle reti classiche, sostituendo la funzione di energia quadratica con una versione continua che ammette una capacità di stoccaggio esponenziale nel numero di neuroni.
La nuova funzione di energia è:

$$ E(\mathbf{q}) = - \frac{1}{\beta} \log \sum_i \exp(\beta \mathbf{x}_i^\top \mathbf{q}) + \frac{1}{2}\lVert \mathbf{q} \rVert^2 $$

dove $\mathbf{q}$ è lo stato corrente, $\mathbf{x}_i$ sono i pattern memorizzati e $\beta > 0$ è un parametro di temperatura inversa. Minimizzando questa energia rispetto a $\mathbf{q}$ si ottiene la regola di update:

$$ \mathbf{q}^{(t+1)} = \mathbf{X}^\top \text{softmax}(\beta \mathbf{X}\mathbf{q}^{(t)}) $$

Il parametro $\beta$ controlla la concentrazione del meccanismo di recupero: per valori alti di $\beta$ la softmax tende a selezionare un singolo pattern, realizzando un recupero netto in un solo passo; 
per valori bassi il sistema produce una combinazione pesata di più pattern.
A differenza della rete classica, la MHN può memorizzare un numero esponenziale di pattern rispetto alla dimensione dello spazio, rendendo il sistema molto più scalabile.

| Proprietà              | Hopfield classica      | Modern Hopfield Network |
| ---------------------- |------------------------| ----------------------- |
| Funzione energia       | Quadratica             | Log-sum-exp continua    |
| Regola di update       | Segno del campo locale | Softmax attention       |
| Capacità di stoccaggio | ≈0.138N                | Esponenziale in N       |
| Step a convergenza     | Multipli               | Tipicamente uno         |
| Tipo di neuroni        | Binari $\{−1,+1\}$     | Continui                |

La forma della regola di update non è casuale: come mostrato nella sezione successiva, essa coincide formalmente con un meccanismo di self-attention dei Transformer, stabilendo un collegamento profondo tra memoria associativa e modelli linguistici moderni.

### 2.4 Relazione con il meccanismo di attention

Il meccanismo di self-attention, introdotto da Vaswani et al. (2017) e alla base dei moderni Transformer, calcola una rappresentazione pesata dei valori in funzione della similarità tra query e chiavi:

$$ \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\Big(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\Big) \mathbf{V} $$

Confrontando questa formula con la regola di update delle MHN, l'equivalenza è formale e precisa: la query $\mathbf{Q}$ corrisponde allo stato corrente del sistema, le chiavi $\mathbf{K}$ corrispondono ai pattern memorizzati, e il fattore di scala $1/\sqrt{d_k}$ gioca il ruolo del parametro di temperatura inversa $\beta$.
In questa lettura, ogni testa di attention esegue un passo di recupero associativo: dato un token in ingresso, il sistema recupera le rappresentazioni più simili presenti nel contesto.
Questo collegamento non è solo teorico: i pesi appresi da un modello pre-addestrato come CodeBERT codificano già, nelle matrici $\mathbf{K}$ e $\mathbf{V}$, una forma implicita di memoria associativa sui pattern del codice sorgente, che il presente progetto sfrutta come base per il retrieval sistematico.

### 2.5 Collegamento con la tesi

CodeBERT (Feng et al., 2020) è un modello di linguaggio pre-addestrato su coppie codice-linguaggio naturale, basato sull'architettura Trasformer di BERT.
Il pre-training combina due obbiettivi: **Masked Language Modeling** su testo e codice sorgente, e **Replace Token Detection** su coppie bimodali, addestrando il modello a costruire rappresentazioni che catturano sia la sintassi che la semantica del codice.
Per ogni sequenza in input, CodeBERT produce un vettore di embedding contestualizzato per ciascun token; la rappresentazione globale della sequenza è convenzionalmente estratta dal token speciale `[CLS]`, che aggrega l'informazione dell'intera sequenza attraverso i layer di self-attention.
Alla luce dell'equivalenza formale stabilita nella sezione precedente, ogni layer di CodeBERT esegue un passo di recupero associativo di Hopfield: le matrici key $\mathbf{K}$ e value $\mathbf{V}$ codificano implicitamente i pattern appresi durante il pre-training, e l'output di ogni testa di attention è una query che si è spostata verso il pattern più simile nello spazio latente.
Ne consegue che lo spazio delle rappresentazioni di CodeBERT è, a tutti gli effetti, una superfice energetica di Hopfield: il pre-training sui dati LTLf consolida dei minimi energetici corrispondenti alle classi logiche, rendendo la geometria dello spazio latente coerente rispetto alla semantica delle formule.

## 3. Dataset LTLf

### 3.1 Definizione del problema logico

Linear Temporal Logic over finite traces (LTLf) è un formalismo logico che permette di esprimere propietà temporali su sequenze finite di stati. A differenza della LTL classica, che ragiona su tracce infinite, LTLf interpreta le formule su traccie di lunghezza finita, rendendola adatta a modellare processi con inizio e fine definiti.
Una formula LTLf su $n$ proposizioni atomiche definisce un insieme di tracce accettanti: sequenze di vettori binari $\mathbf{s}_t \in \{\pm 1\}^n$ che soddisfano i vincoli temporali espressi dalla formula.
In questo progetto, le tracce accettanti costituiscono i **pattern validi** da memorizzare nella rete di Hopfield: il recupero associativo corrisponde quindi alla correzione di una traccia corrotta verso la configurazione logicamente valida più vicina.

### 3.2 Generazione delle formule con Spot

Le formule LTLf vengono generate tramite `randltl`, lo strumento di generazione casuale della libreria [Spot](https://spot.lre.epita.fr/).
Ogni formula è definita su 4 proposizioni atomiche $\{p_0, p_1, p_2, p_3\}$ e viene campionata a tre livelli di complessità crescente, controllata dal parametro di profondità dell'albero sintattico: **simple** (profondità $\leq 3$), **medium** (profondità $4-6$), **complex** (profondità $\geq 7$).
La scelta di 4 proposizioni atomiche bilancia espressività e dimensionalità: ogni stato è un vettore a 4 componenti, mantenendo il problema trattabile per entrambe le implementazioni.

### 3.3 Traduzione in DFA

Ogni formula LTLf viene tradotta in un automa finito deterministico (DFA) tramite `ltlf2dfa`, che sfrutta la decidibilità di LTLf per produrre un automa equivalente.
Il DFA riconosce esattamente l'insieme delle tracce che soddisfano la formula: uno stato accettante del DFA corrisponde a una traccia valida.
Questa rappresentazione automatica è necessaria poiché consente di campionare tracce accettanti in modo sistematico, senza dover enumerare lo spazio esponenziale di tutte le possibili sequenze.

### 3.4 Campionamento delle tracce valide

Le traccie accettanti vengono estratte percorrendo i cammini accettanti del DFA tramite vista in ampiezza con campionamento casuale degli archi.
Ogni traccia di lunghezza $T$ su $n$ proposizioni atomiche è codificata come una matrice $\mathbf{S} \in \{\pm 1 \}^{T \times n}$, che viene quindi appiattita a un vettore $\mathbf{s} \in \{\pm 1 \}^{T \cdot n}$ per l'uso come pattern nella rete.
La codifica in $\{\pm 1 \}$ invece di $\{0, 1\}$ è necessaria per la compatibilità con la regola di Hebb e con la funzione di energia della rete classica. 
La correttezza logica di ogni traccia campionata viene verificata a posteriori tramite **Spot** stesso, garantendo l'assenza di errori nella pipeline.

### 3.5 Corruzione delle tracce

Per generare gli input corrotti da sottoppore alla rete, ogni traccia valida viene perturbata tramite bit-flip casuale: ciascun componente del vettore viene invertito con probabilità $\rho$, indipendentemente dagli altri.
Vengono considerati cinque livelli di corruzione: $\rho \in \{5\%, 10\%, 15\%, 20\%, 25\% \}$, che corrisponono rispettivamente a $1-5$ bit invertiti su un vettore di dimensione $20 \ (T=5,n=4)$.
Ogni combinazione formula-livello di corruzione produce un esempio di test inidpendente, consentendo di misurare il tasso di recupero corretto in funzione del rumore introdotto.

### 3.6 Formato finale del dataset

Il dataset finale è organizzato in file `.npy` separati per livello di complessità e split (memorizzazione / test).
Ogni file contiene un array NumPy di shape $(N, T \cdot n)$ con $T=5, n=4$, ovvero vettori a $20$ componenti in $\{\pm 1 \}$.
Il dataset comprende circa 1500 tracce valide totali, suddivise in set di memorizzazione (usato per il training della rete) e set di test (usato per misurare il recupero).
Per ogni traccia del set di test sono disponibili le versioni corrotte ai cinque livelli di rumore, per un totale di circa 7500 esempi corrotti.

### 3.7 Suddivisione per complessità

Il dataset è suddiviso in tre sottoinsiemi per complessità della formula generatrice: **simple**, **medium** e **complex**.
Le formule semplici coinvolgono operatori temporali di base (`X`, `F`, `G`) con alberi sintattici poco profondi; le formule medie introducono combinazioni di operatori e vincoli temporali più articolati; le formule complesse presentano nesting profondo e dipendenze temporali a lungo raggio.
Questa stratificazione permette di misurare separatamente la robustezza della rete al variare della complessità logica dei pattern memorizzati, testando l'ipotesi che formule complesse — con bacini di attrazione geometricamente più irregolari — siano più difficili da recuperare.

| Livello | Struttura tipica                      | Esempio                      | Cosa vincola                             |
| ------- | ------------------------------------- | ---------------------------- | ---------------------------------------- |
| Simple  | Un solo operatore, nessun nesting     | F p0, G p1, X p2             | Un vincolo su una proposizione           |
| Medium  | Combinazioni di 2–3 operatori         | G(p0 → F p1), F(p0 ∧ p1)     | Vincoli condizionali o congiunti         |
| Complex | Nesting profondo, dipendenze multiple | G(p0 → X(p1 ∧ F(p2 → G p3))) | Catene temporali con condizioni annidate |

## 4. Implementazione

### 4.1 Architettura generale del progetto

Il progetto è organizzato in moduli distinti con responsabilità separate.
La cartella `dataset/` contiene gli script di generazione e corruzione delle tracce LTLf e i file `.npy` prodotti dalla pipeline.
La cartella `classical/` contiene l'implementazione della rete classica in `C`, con i file sorgente per la regola di Hebb, l'update asincrono e la parallelizzazione **OpenMP**.
La cartella `mhn/` contiene il codice `Python` di implementazione della classe `ModernHopfiledNetwork` e gli script di recupero.
La cartella `experiments/` raccoglie gli script che eseguono i quattro esperimenti principali e producono i risultati in formato CSV.
La cartella `visualization/` contiene gli script per la generazione dei plot dei risultati, il plot delle energie e la visualizzazione PCA e t-SNE.
La cartella `tests/` contiene gli script di test per il controllo della correttezza dei dati e del recupero associativo.
Ogni modulo è indipendente e può essere eseguito separatamente; gli esperimenti leggono i dati da `dataset/` e i modelli da `classical/hopfield.c` e `mhn/modern_hopfield.py`, producendo output in `results/`. 

### 4.2 Hopfield classica in C

#### 4.2.1 Regola di Hebb

La memorizzazione dei pattern avviene tramite la regola di Hebb, implementata nella funzione `hebb_learning()`.
Per ogni coppia di neuroni $(i, j)$, il peso $W_{ij}$ viene calcolato come somma dei prodotti delle attivazioni nei pattern memorizzati, normalizzata per il numero di neuroni $N$:

$$ W_{ij} = \frac{1}{N} \sum_{\mu = 1}^p \xi_i^{\mu} \xi_j^{\mu}, \quad W_{ii} = 0$$

La normalizzazione per $N$ garantisce che i pesi rimangano in un intervallo stabile indipendentemente dalla dimensione della rete.
L'assenza di auto-connessioni $(W_{ii} = 0)$ è imposta esplicitamente dopo il calcolo.

#### 4.2.2 Update asincrono

Il recupero di un pattern corrotto avviene tramite update asincrono, implementato nelle funzioni `async_update()` e `retrieve_with_energy()`.
A ogni step, i neuroni vengono aggiornati in un ordine specificato dall'array `order`, che viene preparato e shufflato dal chiamante Python tramite `numpy.random.permutation` prima di ogni epoca. 
Per ogni neurone selezionato, lo stato viene aggiornato secondo il segno del campo locale $h_j = \sum_k W_{jk} \sigma_k$:
$$ \sigma_j \leftarrow \operatorname{sgn}\Big(\sum_k W_{jk} \sigma_k\Big) $$

La convergenza viene verificata esplicitamente a ogni step confrontando lo stato corrente con una copia dello stato precedente: il sistema è considerato converso quando nessun neurone cambia valore tra uno step e il successivo. 
La funzione `async_update()` restituisce `1` in caso di convergenza effettiva e `0` se viene raggiunto il limite `max_steps` senza stabilizzazione. 
L'ordine di aggiornamento viene randomizzato lato `Python` tramite `numpy.random.permutation` prima di ogni chiamata a `async_update()`.
La funzione `retrieve_with_energy()` estende questo meccanismo registrando il valore di energia a ogni step tramite `compute_energy()`, producendo la traiettoria energetica usata per i grafici di convergenza del Capitolo 5.


#### 4.2.3 Parallelizzazione con OpenMP

La parallelizzazione con OpenMP è applicata a due funzioni: `hebb_learnng()` e `compute_energy()`.
Nella prima, il ciclo esterno sui neuroni `i` è parallelizzato con `#pragma omp parallel for schedule(static)`: il calcolo di ogni riga della matrice `W` è indipendente dalle altre, rendendo la parallelizzazione priva di race condition.
In `compute_energy()`, la somma doppia è penalizzata tramite `#pragma omp simd reduction(-:E)`, che accumula i contributi parziali di ogni thread in modo thread-safe.
L'update asincrono in `retrieve_with_energy()` non viene parallelizzato: la natura sequenziale dell'aggiornamento neurone per neurone è una garanzia necessaria per la decrescita monotonica dell'energia, e parallelizzarlo introdurrebbe aggiornamenti concorrenti sullo stato condiviso `query`.
La compilazione con supporto OpenMP avviene tramite il flag -fopenmp passato a GCC, come indicato nell'header del file sorgente.

### 4.3 MHN in Python

#### 4.3.1 Memorizzazione dei pattern

La memorizzazione nella MHN è implementata nel metodo `store()`, che salva i pattern come righe della matrice `self.memories` di shape $(p,d)$, dove $p$ è il numero di pattern e $d$ la loro dimensione. 
A differenza della rete classica, non viene calcolata alcuna matrice di pesi: i pattern vengono mantenuti esplicitamente in memoria e usati direttamente come riferimento durante il recupero. 
Questa differenza architetturale è fondamentale: nella rete classica l'informazione è compressa nella matrice $W$, mentre nella MHN ogni pattern rimane individualmente accessibile, il che rende possibile la capacità di stoccaggio esponenziale.

#### 4.3.2 Recupero associativo

Il recupero è strutturato in tre livelli. Il metodo privato `_step()` implementa un singolo passo della regola di update:

$$ \mathbf{q}^{t+1)} = \mathbf{X}^{\top} \operatorname{softmax} \Big(\beta \mathbf{X} \mathbf{q}^{(t)} \Big)$$

La stabilizzazione numerica è applicata sottraendo il massimo dei punteggi prima dell'esponenziale, prevenendo overflow senza alterare il risultato della softmax.
Il metodo `retrieve()` itera `_step()` per un numero fisso di passi e restituisce lo stato continuo finale.
Il metodo `retrieve_tracked()` aggiunge convergenza anticipata tramite tolleranza $\varepsilon = 10^{-8}$ sulla variazione massima tra passi consecutivi, restituendo la tripletta (`stato_finale`, `energy_trace`, `n_steps_effettivi`).
Il metodo `retrive_binary()` applica `np.sign()` all'output continuo con gestione esplicita dello zero (assegnato a $+1$) per garantire output in $\{\pm 1 \}$.

#### 4.3.3 Funzione energia

La funzione `energy()` implementa la log-sum-exp stabile:

$$ E(\mathbf{q}) = -\frac{1}{\beta} \log \sum_i \exp \Big( \beta \mathbf{x}_i^{\top} \mathbf{q} \Big) + \frac{1}{2} ||\mathbf{q}||^2 $$

Il metodo `nerest_memory()` individua il pattern memorizzato più vicino alla distanza L2, utile per valutare il recupero senza binarizzazione nei casi in cui lo stato finale non coincide esattamente con un pattern.
Il metodo `recovery_rate()` incapsula l'intera pipline di valutazione: corrompe ogni pattern tramite bit-flip al tasso specificato, esegue il recupero, e misura la frazione di recuperi esatti (distanza di Hamming zero) su n_trials prove, con seed fisso per la riproduzione dei risultati.


## 5. Verifica preliminare

### 5.1 Obiettivi dei test di sanità

Prima di eseguire gli esperimenti principali è necessario verificare che le implementazioni si comportino correttamente sui casi più semplici e controllabili.
I test di sanità hanno tre obbiettivi distinti: verificare la correttezza funzionale delle implementazioni su casi con risposta nota analiticamente; garantire che la proprietà teoriche fondamentali — recupero esatto e decrescita monotonica dell'energia — siano effettivamente soddisfatte; e validare il dataset prima che venga usato negli esperimenti, escludendo errori silenziosi nella pipeline di generazione.
Un bug che supera i test di sanità si propagherà in tutti gli esperimenti successivi, rendendo i risultati non interpretabili: investire nella verifica preliminare è quindi una scelta di correttezza scientifica prima che di ingegneria.

### 5.2 Recupero perfetto senza corruzione

Il primo test verifica la condizione necessaria più elementare: una rete che ha memorizzato un pattern deve recuperarlo esattamente quando viene interrogata con quel pattern stesso, senza corruzione.
Per entrambi i modelli, vengono memorizzati un insieme di pattern e ciascuno di essi viene usato direttamente come query; il recupero è considerato corretto se l'output coincide bit per bit con il pattern originale, ovvero se la distanza di Hamming normalizzata è zero.
Il superamento di questo test esclude errori nella regola di Hebb, nella normalizzazione dei pesi e nella regola di update, e costittuisce la precondizione per tutti i test successivi.

### 5.3 Recupero da query corrotte

Il secondo test verifica che la rete sia in grado di recuperare un pattern memorizzato a partire da una versione corrotta al $20\%$ tramite bit-flip casuale.
La scelta del $20\%$ è motivata dalla posizione di questo tasso all'interno del range sperimentale $[5\%, 25\%]$: abbastanza alto da costruire una perturbazione significativa, abbastanza basso da essere ben dentro la capacità di recupero attesa per entrambi i modelli con un numero ridotto di pattern memorizzati.
Il test viene ripetuto su più pattern e più seed casuali, e il recupero è considerato corretto se l'output binarizzato coincide con il pattern originale.
Un tasso di successo inferiore al $95\%$ su questo test segnala un problema nell'implementazione prima di qualsiasi considerazione sulla capacità.

### 5.4 Monotonia dell'energia

Il terzo test verifica la proprietà teorica fondamentale delle reti di Hopfield: la funzione di energia deve essere non crescente durante il recupero, ovvero $E(\mathbf{q}^{(t+1)}) \leq E(\mathbf{q}^{(t)})$ per ogni passo $t$.
La verifica avviene tramite `retrieve_tracked()` per la MHN e `retrieve_with_energy()` per la rete classica, che restituiscono la traiettoria energetica completa durante il recupero.
Per ogni query di test viene controllato che la sequenza di valori energetici sia monotonicamente non crescente; una violazione, anche in un singolo passo, indica un'inconsistenza nell'implementazione della regola di update o della funzione di energia.
Questo test è particolaremente critico per la rete classica, dove la garanzia di monotonia dipende dall'ordine asincrono degli aggiornamenti.

### 5.5 Capacità di stoccaggio

Il quarto test verifica il comportamento delle due reti in prossimità della capacità teorica. Per la rete classica, la capacità teorica è $p^* \approx 0.138 \cdot N$: con $N=20$ neuroni, il limite teorico è circa $2-3$ pattern.
Il test memorizza un numero crescente di pattern ortogonali sintetici e misura il tasso di recupero corretto al variare del numero di pattern memorizzati, verificando che la degradazione si manifesti nell'intorno del limite teorico.
Per la MHN, il test analogo mostra che il recupero rimane corretto ben oltre $p^*$, confermando empiricamente la capacità esponenziale.
I pattern usati in questo test sono ortogonali per costruzione, in modo da eliminare l'interferenza tra pattern come variabile confondente e isolare l'effetto della capacità.

### 5.6 Validazione del dataset

Il test di validazione del dataset verifica tre proprietà prima che il file vengano usati negli esperimenti.
La prima è la **correttezza della shape**: ogni array `.npy` deve avere shape $(N, T\cdot n)$ con $T=5$ e $n=4$, ovvero vettori a 20 componenti; una shape diversa indica un errore nella pipeline di flattening.
La seconda è la validità dei valori: ogni componente deve appartenere a $\{\pm 1\}$; valori fuori da questo insieme segnalano un errore nella codifica o nella corruzione.
La terza è la presenza di tutti i file attesi: per ciascuno dei tre livelli di complessità e dei cinque tassi di corruzione deve esistere il file corrispondente, sia per il set di memorizzazione che per il set di test.
Il superamento di questi controlli garantisce che gli esperimenti operino su dati corretti e completi, escludendo errori silenziosi nella pipeline di generazione descritta nel Capitolo 3.

## 6. Esperimenti

### 6.1 Setup sperimentale

Tutti gli esperimenti condividono un insieme comune di parametri. I pattern usati sono vettori binari in $\{\pm 1\}$ di dimensione $N=20$, estratti dal dataset LTLf descritto nel Capitolo 3.
I tassi di corruzione considerati sono $\rho \in \{5\%, 10\%, 15\%, 20\%, 25\% \}$, applicati tramite bit-flip indipendente su ogni componente.
Per ogni combinazione di parametri vengono eseguiti 200 trial con seed fisso($\text{seed=0}$) per garantire la riproducibilità.
Le metriche principali sono il **tasso di recupero corretto** (frazione di trial in cui l'output binarizzato coincide esattamente con il pattern originale), il **numero di step a convergenza**, e il **profilo egergetico** durante il recupero.
La MHN è configurata con $\beta = 1.0$ e un massimo di $20$ step; la rete classica con un massimo di $20$ epoche di update asincrono.
L'exact match rate è la metrica principale: un recupero è contato come corretto solo se il vettore binarizzato recuperato coincide esattamente con il pattern originale in tutte le componenti, coerentemente con il fatto che una traccia LTLf parzialmente corretta è comunque logicamente invalida.

### 6.2 Esperimento 1 — Recupero al variare della corruzione

#### 6.2.1 Obiettivo

L'obbiettivo è misurare come il tasso di recupero corretto degrada al crescere del rumore introdotto, confrontando la rete classica e la MHN sullo stesso dataset di pattern LTLf.

#### 6.2.2 Metodo

Per ciascuno dei cinque livelli di corruzione, ogni pattern del set di memorizzazione viene perturbato tramite bit-flip al tasso $\rho$ e sottoposto a recupero.
Il tasso di recupero corretto è calcolato come media su 200 trial per livello.
Il confronto è diretto: stessi pattern, stessa corruzione, stessi seed per entrambi i modelli.

#### 6.2.3 Risultati

![Exp1](../results/exp1_correction.png)

Entrambi i modelli mostrano un tasso di recupero decrescente al crescere del rumore, ma con profili distinti.
La MHN mantiene un tasso di recupero più alto a tutti i livelli di corruzione, con un degrado più graduale.
La rete classica mostra una caduta più marcata oltre il $15\%$ di corruzione, coerentemente con la maggiore rigidità dei suoi bacini di attrazione.
A bassa corruzione $(5-10\%)$ entrambi i modelli operano vicino al $100\%$, confermando la correttezza delle implementazioni verificata nel Capitolo 5.

### 6.3 Esperimento 2 — Capacità di stoccaggio

#### 6.3.1 Obiettivo

L'obbiettivo è stimare quante tracce LTLf distinte possono essere memorizzate prima che il recupero degeneri, e verificare empiricamente la differenza di capacità tra rete classica e MHN. 

#### 6.3.2 Metodo

Il numero di pattern memorizzati viene fatto variare da $1$ fino oltre il limite teorico classico $p^* \approx 0.138 \cdot N = 2-3$ per la rete classica, e fino a valori significativamente più alti per le MHN.
Per ogni valore di $p$, il tasso di recupero corretto viene misurato a corruzione fissa del $10\%$ su $200$ trials.
Il degrado è identificato come il valore $p$ oltre il quale il tasso scende sotto il $90\%$.

#### 6.3.3 Risultati

![Exp2](../results/exp2_capacity.png)

La rete classica mostra un degrado netto del recupero nell'intorno di $p^* \approx 0.138 \cdot N$, confermando la previsione teorica.
La MHN mantiene un tasso di recupero elevato per un numero di pattern significativamente superiore, con degradazione molto più graduale.
Questo risultato conferma empiricamente la capacità di stoccaggio esponenziale della MHN descritta nella sezione 2.3 e suggerisce la motivazione principale per il suo utilizzo nel progetto.

### 6.4 Esperimento 3 — Confronto tra Hopfield classica e MHN

#### 6.4.1 Obiettivo

L'obbiettivo è un confronto diretto tra i due modelli su tre dimensioni: accuratezza di recupero, numero di step a convergenza e profilo energetico durante il recupero.

#### 6.4.2 Metodo

Il confronto è eseguito su un set fisso di pattern LTLf ai tre livelli di complessità, con corruzione al $15\%$ e 200 trials per modello, inoltre vengono testati 3 livelli di $\beta$ per la MHN. 
Per ogni trial vengono registrati: l'esito del recupero (corretto/errato), il numero di step fino a convergenza, e la sequenza di valori energetici durante il recupero tramite `retrieve_trecked()` e `retrieve_with_energy()`.

#### 6.4.3 Risultati: accuracy

![Exp3 Accuracy](../results/exp3_accuracy.png)

Il grafico mostra l'accuratezza di recupero alla corruzione del $15\%$ per tutti i modelli, suddivisa per livello di complessità.
La MHN raggiunge accuratezza prossima a $1.0$ a tutti i livelli di $\beta$ e per tutti i livelli di complessità.
La rete classica mostra un'accuratezza leggermente inferiore su **simple** ($\sim 0.92$) e **medium** ($\sim 0.88$), mentre su **complex** recupera quasi completamente ($\sim 0.98$), un andamento controintuitivo rispetto all'ipotesi che formule complesse siano più difficili da recuperare.
Questo suggerisce che, per la rete classica, la struttura geometrica del pattern LTLf complex nel presente dataset è più favorevole al recupero rispetto ai livelli inferiori, probabilmente per una maggiore separazione tra i pattern memorizzati.

#### 6.4.4 Risultati: step a convergenza

![Exp3 Steps](../results/exp3_steps.png)

Il grafico mostra il numero medio di step a convergenza per modello e livello di complessità.
La rete classica converge in circa 2 step in modo uniforme su tutti i livelli, identico alla maggior parte delle configurazioni MHN.
La MHN con $\beta = 1.0$ è l'unica eccezione: richiede circa 2.5 step su **simple** e **medium**, scendendo leggermente su **complex**.
Questo comportamento è atteso: a $\beta$ basso la softmax è più diffusa, il che rallenta la convergenza rispetto a valori di $\beta$ dove la selezione di pattern è più netta.
La variazione tra livelli di complessità è minima per tutti i modelli, indicando che il numero di step non è una dimensione sensibile alla complessità logica del dataset.

#### 6.4.5 Risultati: profilo energetico

![Exp3 Energy](../results/exp3_energy_profile.png)

Il grafico mostra la traiettoria energetica media su tracce complex per tutti i modelli.
Tutti i modelli partono da un'energia iniziale tra $-6$ e $-8$ e convergono verso un minimo.
Le MHN con $\beta \geq 2.0$ mostrano una discesa brusca al primo step, raggiungendo il minimo già allo step $1-2$ e rimanendo stazionarie fino allo step 10, confermando la convergenza in un singolo passo.
La MHN con $\beta = 1.0$ mostra una discesa leggermente più graduale, coerente con il maggiore numero di step osservato nel grafico precedente.
La rete classica converge più lentamente, raggiungendo il suo minimo intorno allos tep $3-4$, con un valore finale di energia leggermente più alto ($\sim -14$) rispetto alle MHN ($\sim -16$), indicando che la rete classica si assesta su minimi energetici meno profondi.
I valori negativi dell'energia sono attesi per entrambi i modelli: i minimi della funzione di energia corrispondono per costruzione ai valori più negativi, e una discesa verso energie più negative indica convergenza verso uno stato più stabile.

### 6.5 Esperimento 4 — Robustezza rispetto alla complessità

#### 6.5.1 Obiettivo

L'obbiettivo è valutare se la complessità logica della formula LTLf generatrice influenza il tasso di recupero, testando l'ipotesi che formule più complesse producano pattern più difficili da recuperare per via di bacini di attrazione geometricamente più irregolari.

#### 6.5.2 Metodo

Il tasso di recupero corretto viene misurato separatamente sui tre livelli del dataset — **simple**, **medium** e **complex** — per entrambi i modelli, con corruzione fissa e $200$ trials per livello.
Le barre di errore rappresentano l'intervallo di confidenza sulla stima del tasso di recupero tra i trial.
Il confronto diretto tra i due modelli è visualizzato come line plot per rendere leggibile il trend al crescere della complessità. 

#### 6.5.3 Risultati

![Exp4](../results/exp4_complexity.png)

Il grafico mostra un risultato controintuitivo rispetto all'ipotesi iniziale: **entrambi i modelli migliorano al crescere della complessità**, con un trend crescente monotono da simple a complex.
La rete classica parte da $\sim 0.92$ su simple, scende leggermene a $\sim 0.90$ su medium, e risale a $\sim 0.95$ su complex.
La MHN mostra un incremento più marcato e continuo: da $\sim 0.68$ su simple a $\sim 0.74$ su medium fino a $\sim 0.82$ su complex, con intervalli di confidenza più ampi rispetto alla classica.
\
\
Questo andamento suggerisce che le tracce LTLf di livello **complex**, pur essendo generate da formule logicamente più articolate, producono pattern più separati geometricamente nello spazio $\{\pm 1\}^{20}$: la maggiore struttura sintattica della formula vincola le tracce accettanti in modo più stringente, riducendo la densità dei pattern nel dataset e aumentando la distanza media tra pattern memorizzati.
Bacini di attrazione più distanti tra loro rendono il recupero più robusto, non più difficile.
\
\
Il risultato falsifica parzialmente l'ipotesi iniziale formulata nella sezione 3.7: la complessità logica non degrada il recupero, ma lo facilita nel range di pattern e dimensioni considerato.
Un effetto opposto potrebbe emergere a scale maggiori, con un numero di pattern memorizzati molto più alto, dove la densità dello spazio diventerebbe il fattore dominante.
\
\
Il divario sistematico tra classica e MHN osservato in questo esperimento è verosimilmente attribuibile al valore di $\beta = 1.0$ usato per la MHN: come mostrato nell'Esperimento 3, valori bassi di $\beta$ producono una softmax diffusa che degrada l'accuratezza del recupero binarizzato.
In regime con pochi pattern memorizzati — ben sotto la capacità teorica di entrambi i modelli — il vantaggio strutturale della MHN non emerge, e il parametro $\beta$ diventa il fattore dominante per le prestazioni.


## 7. Visualizzazioni geometriche

### 7.1 PCA delle traiettorie di recupero

![PCA](../results/pca_comparison_subplot.png)

La PCA proietta lo spazio $\{ \pm 1 \}^{20}$ sui due assi di massima varianza, che spiegano rispettivamente il $26.6\%$ (PC1) e il $17.6\%$ (PC2) della varianza totale del dataset.
I punti blu rappresentano i pattern memorizzati, le croci grige le query corrotte, le frecce verdi le traiettorie di recupero della rete classica e i quadrati rossi gli stati finali recuperati della MHN.
\
\
Nel pannello della rete classica le frecce mostrano che la maggior parte delle query corrotte viene attratta verso il pattern memorizzato più vicino: la traiettorie sono corte e orientate verso il pattern di riferimento, indicando convergenza corretta.
Si notano tuttavia alcuni casi in cui la query corrotta viene attratta verso un pattern sbagliato — in particolare M1, M6 e M2 mostrano frecce che partono da zone di sovrapposizione tra bacini adiacenti — il che è coerente con il tasso di recupero non perfetto osservato nell'Esperimento 1.
\
\
Nel pannello MHN gli stati recuperati (quadrati rossi) si posizionano in prossimità dei pattern memorizzati corrispondenti, ma con una dispersione maggiore rispetto alla rete classica: questo riflette il fatto che la MHN opera in spazio continuo e il recupero finale è il risultato di binarizzazione, non di un aggiornamento discreto.
I casi di recupero errato sono visibili come quadrati rossi distanti dal pattern di riferimento, in corrispondenza delle stesse zone di sovrapposizione identificate per la rete classica.

### 7.2 t-SNE dei bacini di attrazione

![t-SNE](../results/tsne_comparison_subplot.png)

Il t-SNE preserva le distanze locali e non quelle globali: la disposizione assoluta dei cluster non ha significato, ma la vicinanza tra punti all'interno di ogni cluster riflette similarità genuina nello spazio originale.
I pattern memorizzati (punti blu) appaiono come riferimenti fissi attorno ai quali si aggregano le query corrotte (croci grige) e gli stati recuperati (verde per la classica, rosso per la MHN).
\
\
In entrambi i pannelli la struttura a cluster è evidente: per la maggior parte dei pattern — M0, M3, M5, M7, M8, M10 — le query corrotte e i rispettivi stati recuperati sono visivamente raggruppati attorno alla memoria corrispondente, confermando che il recupero avviene correttamente nel vicinato locale.
I pattern M1, M9 e M4 mostrano invece una maggiore dispersione dei punti recuperati, con alcuni stati che si collocano lontano dalla memoria di riferimento: questi sono i casi di recupero errato già identificati nella PCA.
\
\
La differenza principale tra i due pannelli è nella compattezza dei cluster di recupero: la rete classica (verde) tende a posizionare gli stati recuperati molto vicini alla memoria corrispondente, mentre la MHN (rosso) mostra cluster leggermente più dispersi, coerentemente con la natura continua del suo spazio di recupero prima della binarizzazione. 

### 7.3 Interpretazione geometrica

PCA e t-SNE sono strumenti complementari che descrivono la geometria del recupero a scale diverse.
La PCA fornisce una visione della **struttura globale**: le traiettorie di recupero mostrano la direzione e la lunghezza del percorso nello spazio ridotto, rendendo visibili le zone di confine tra bacini di attrazione adiacenti.
Il t-SNE fornisce invece una visione della **struttura locale**: i cluster mostrano quanto compattamente gli stati recuperati si aggregano attorno alle memorie, indipendentemente dalla posizione globale.
\
\
Messi insieme, i due grafici raccontano la stessa storia in due linguaggi diversi: esiste una struttura geometrica ben definita nello spazio degli stati, i pattern memorizzati sono i suoi attrattori, e il recupero è il processo di scivolamento verso il minimo più vicino.
Le zone in cui i due modelli falliscono — visibili come frecce misdirected nella PCA e come punti isolati nel t-SNE — corrispondono alle stesse regioni di sovrapposizione tra bacini, confermando che gli errori di recupero non sono casuali ma strutturali, legati alla geometria del dataset.
\
\
Questa interpretazione geometrica costituisce la giustificazione visiva delle proprietà teoriche discusse nel Capitolo 2: la funzione di energia non è solo uno strumento matematico, ma descrive una superficie fisica reale, i cui minimi sono visibili e localizzabili nello spazio ridotto. 

## 8. Discussione

### 8.1 Interpretazione dei risultati

Gli esperimenti confermano che entrambi i modelli sono in grado di svolgere il compito di recupero associativo su tracce LTLf, ma con profili di prestazione distinti.
Il risultato più rilevante è che le tracce LTLf di comportano come pattern ben separati nello spazio $\{\pm 1\}^{20}$: il recupero corretto è possibile a tutti i livelli di corruzione testati, e la struttura geometrica visualizzata nel Capitolo 7 mostra bacini di attrazione identificabili e localizzati.
Questo conferma l'ipotesi fondamentale del progetto: la struttura logico-temporale delle tracce LTLf può essere appresa e sfruttata da modelli neurali basati sulla minimizzazione dell'energia, senza supervisione esplicita sulle regole logiche.
\
\
Il risultato controintuitivo dell'Esperimento 4 — miglioramento del recupero al crescere della complessità logica — suggerisce che la complessità sintattica della formula generatrice produce una separazione geometrica maggiore tra i pattern, rendendo i bacini di attrazione più distinti.
Questa è un'osservazione empirica che merita approfondimento a scale maggiori, ma è coerente con l'idea che vincoli logici più stringenti producano distribuzioni di stati più strutturate nello spazio binario.

### 8.2 Vantaggi della MHN

La MHN mostra tre vantaggi strutturali rispetto alla rete classica, tutti confermati empiricamente.
Il primo è la **capacità di stoccaggio**: la MHN mantiene un recupero corretto per un numero di pattern significativamente superiore al limite teorico classico $0.138 \cdot N$, confermando la capacità esponenziale descritta nella sezione 2.3.
Il secondo è la **velocità di convergenza**: la MHN converge tipicamente in 1-2 step per valori alti di $\beta$, rispetto ai 2-4 della rete classica, grazie alla regola di update softmax che seleziona direttamente il pattern più simile invece di aggiornare neurone per neurone.
Il terzo è la **continuità dello spazio di recupero**: operando in uno spazio continuo, la MHN può rappresentare stati intermedi durante il recupero, rendendola più adatta all'integrazione con modelli neurali profondi che operano su rappresentazioni dense.

### 8.3 Limiti della rete classica

La rete classica mostra due limitazioni principali emerse dali esperimenti.
La prima è la **saturazione delle capacità**: già con pochi pattern il recupero degrada nell'intorno del limite teorico, e l'interferenza tra pattern produce errori sistematici nelle zone di sovrapposizione tra bacini, visibili sia nella PCA che nel t-SNE.
La seconda è la **dipendenza dell'ordine di aggiornamento**: l'update asincrono introduce una dipendenza dalla permutazione casuale dei neuroni, che produce varianza nei risultati tra trial diversi sullo stesso input.
Nonostante questi limiti, la rete classica rimane un modello di riferimento prezioso per la semplicità interpretativa e per la diretta corrispondenza con il modello di Ising, che fornisce il framework teorico alla base dell'intero progetto.

### 8.4 Limiti del progetto

Il progetto presenta alcune semplificazioni che ne delimitano la portata.
La dimensione dei vettori di stato è fissa a $N=20$: questo è sufficente per validare le propietà qualitative dei modelli, ma non permette di esplorare il comportamento a scale rilevanti per applicazioni reali.
Il dataset di circa 1500 tracce è generato con un mumero limitato di prposizioni atomiche (4) e tracce di lunghezza fissa ($T=5$): formule su più proposizioni o tracce più lunghe potrebbero mostrare dinamiche diverse.
\
\
La visualizzazione t-SNE presenta una limitazione intrinseca: essendo un metodo stocastico con iperparametri sensibili (perplexity, learining rate, numero di iterazioni), la struttura dei cluster può variare tra esecuzioni diverse e non è stabile in senso assoluto.
I risultati del t-SNE vanno quindi interpretati come supporto qualitativo alla PCA, non come evidenza quantitativa indipendente.
Infine, la MHN è stata testata principalmente con $\beta = 1.0$ nell'Esperimento 4, il che ha probabilmente penalizzato artificialmente le sue prestazioni rispetto alla rete classica: un'analisi sistematica dell'effetto di $\beta$ avrebbe reso il confronto più equo.

### 8.5 Implicazioni per la tesi

Il presente progetto fornisce la giustificazione teorica per cui il fine-tuning su dati LTLf dovrebbe funzionare: se la struttura logico-temporale produce bacini di attrazione identificabili già in un sistema semplice come la MHN, allora un modello della capacità di CodeBERT — che opera in uno spazio a 768 dimensioni con miliardi di parametri — dovrebbe essere in grado di catturare quella struttura in modo ancora più ricco, producendo rappresentazioni migliori rispetto a un modello pre-addestrato solo su codice generico.

## 9. Conclusioni

### 9.1 Risultati principali

Il progetto ha implementato e confrontato una rete di Hopfield classica in `C` e una Modern Hopfield Network in `Python` su un dataset di tracce LTLf generate con `Spot`, verificando empiricamente le proprietà teoriche dei due modelli e la loro capacità di operare come correttori di errori neurali su pattern logicamente validi. I risultati principali sono quattro. 
Primo: entrambi i modelli recuperano correttamente le tracce LTLf corrotte, con tassi di recupero elevati fino al 20% di corruzione, confermando che la struttura logico-temporale delle tracce produce pattern sufficientemente separati nello spazio binario. 
Secondo: la MHN mostra una capacità di stoccaggio superiore e una convergenza più rapida rispetto alla rete classica, confermando le previsioni teoriche di Ramsauer et al. (2020). 
Terzo: la complessità logica della formula generatrice non degrada il recupero, ma lo facilita, suggerendo che vincoli logici più stringenti producono distribuzioni di stati più strutturate. 
Quarto: le visualizzazioni PCA e t-SNE rendono visibile la struttura geometrica dei bacini di attrazione, confermando che il recupero associativo corrisponde a un processo di scivolamento verso i minimi energetici più vicini.

### 9.2 Risposta alla domanda di ricerca

La domanda di ricerca del progetto era: *una rete di Hopfield può apprendere la struttura logico-temporale di tracce LTLf valide e correggerle quando corrotte, senza supervisione esplicita sulle regole logiche?* 
La risposta è sì, con una qualificazione importante.
\
\
Entrambi i modelli dimostrano che la struttura delle tracce valide può essere codificata nei minimi energetici della rete tramite apprendimento non supervisionato (regola di Hebb per la classica, memorizzazione esplicita per la MHN), e che il recupero associativo agisce come un correttore di errori implicito: la rete non conosce le regole LTLf, ma la geometria dello spazio degli stati le riflette. 
La qualificazione è che questo risultato è valido nel regime sperimentale considerato — vettori a 20 componenti, pochi pattern memorizzati, corruzione moderata — e non è garantito a scale maggiori senza ulteriore verifica.

### 9.3 Sviluppi futuri

Tre direzioni di estensione sono naturali a partire dai risultati ottenuti.
\
\
La prima è l'**aumento della scala**: testare i modelli con vettori di dimensione maggiore($N \geq 100$), un numero più alto di proposizioni atomiche e tracce più lunghe permetterebbe di esplorare il regime in cui la capacità della rete classica satura definitivamente e il vantaggio della MHN diventa dominante.
Questo richiederebbe un dataset più ampio e un'ottimizzazione dell'implementazione `C` per reti di grandi dimensioni, eventualmente sfruttando la parallelizzazione `OpenMP` già presente.
\
\
La seconda è l'**analisi sistematica di** $\beta$: i risultati dell'Esperimento 4 mostrano che il parametro $\beta$ è determinante per le prestazioni della MHN, ma non è stato oggetto di ottimizzazione sistematica.
Un'analisi grid-search su $\beta$ in funzione del numero di pattern e del tasso di corruzione fornirebbe una guida per la scelta del parametro e renderebbe il confronto tra modelli più equo.
\
\
La terza direzione, e più rilevante per il percorso complessivo, è la **validazione dell'ipotesi della tesi**: il progetto ML dimostra che un sistema che apprende la struttura logico-temporale delle tracce LTLf — anche nella forma minimalista della regola di Hebb — riesce a correggerle e recuperarle.
La tesi estende questa ipotesi a un modello di scala molto maggiore: un CodeBERT fine-tuned su dati LTLf dovrebbe produrre rappresentazioni semanticamente più coerenti rispetto a un CodeBERT generico, misurabile tramite metriche di retrieval o classificazione su task downstream.
Il collegamento con il presente progetto è che entrambi testano la stessa idea a scale diverse: **insegnare la struttura logica a un sistema neurale lo rende più competente su quel dominio**, indipendentemente dal fatto che le regole siano codificate esplicitamente o emergano dalla geometria dello spazio appreso.

## Appendice A — Struttura del progetto

```text
ML_Course_Project/
│
├── classical/                          # Rete di Hopfield classica in C
│   ├── hopfield.c                      # Implementazione: Hebb, update asincrono, energia
│   ├── hopfield.so                     # Libreria condivisa compilata
│   └── hopfield_bindings.py            # Wrapper ctypes per interfaccia Python
│
├── mhn/                                # Modern Hopfield Network in Python
│   └── modern_hopfield.py              # Classe ModernHopfieldNetwork
│
├── dataset/                            # Generazione e storage del dataset LTLf
│   ├── generate_dataset.py             # Pipeline: randltl → DFA → tracce → corruzione
│   └── data/
│
├── experiments/                        # Script degli esperimenti principali
│   ├── exp1_correction.py              # Recupero al variare della corruzione
│   ├── exp2_capacity.py                # Capacità di stoccaggio
│   ├── exp3_comparison.py              # Confronto classica vs MHN
│   └── exp4_complexity.py              # Robustezza per complessità LTLf
│
├── results/                            # Output degli esperimenti (CSV + PNG)
│
├── visualization/                              # Script di visualizzazione geometrica
│   ├── pca_trajectory.py                     # Traiettorie PCA di recupero
│   ├── tsne_comparison.py                    # Bacini di attrazione t-SNE
│   ├── generate_plots.png
│   └── energy_landscape.png
│
├── tests/                              # Test di sanità (Capitolo 5)
│   └── test_sanity.py                  # Recupero perfetto, monotonia energia, capacità
│
└── report/
│   └── relazione.md                    
```

## Appendice B — Parametri sperimentali

| Parametro                       | Valore                                               |
|---------------------------------|------------------------------------------------------|
| Numero proposizioni atomiche    | 4 $(p_0, p_1, p_2, p_3)$                             |
| Lunghezza traccia $T$           | 5 passi                                              |
| Dimensione vettore di stato $N$ | 20 ($T \cdot n = 5 \times 4$)                        |
| Numero tracce totali            | ≈ 1500 (tracce valide, set di memorizzazione + test) |
| Livelli di complessità          | 3 (simple, medium, complex)                          |
| Corruption rates ρ\\rhoρ        | 5%, 10%, 15%, 20%, 25%                               |
| $\betaβ$ MHN (default)          | 1.0                                                  |
| $\betaβ$ MHN (Esperimento 3)    | 1.0, 2.0, 5.0, 10.0                                  |
| Max steps rete classica         | 20 epoche                                            |
| Max steps MHN                   | 20 passi                                             |
| Numero trial per esperimento    | 200                                                  |
| Seed per riproducibilità        | 0                                                    |
| Tolleranza convergenza MHN      | $10^{-8}$ (su variazione massima tra passi)          |
| Codifica valori                 | $\{-1, +1\}$                                         |

## Appendice C — Note tecniche

### Ambiente di sviluppo

Il progetto è stato sviluppato su sistema operativo `Kubuntu 24.05 LTS` con interprete `Python 3.12`.
L'IDE utilizzato è `PyCharm Professional`. Il controllo di versione è gestito tramite `Git`.

### Compilazione della libreria `C`

La libreria condivisa `hopfield.so` viene compilata con `GCC` tramite il seguente comando, incluso come commento nell'header di `hopfield.c`:
\
```bash
gcc -O2 -fopenmp -shared -fPIC -o hopfield.so hopfield.c
```
\
La libreria viene caricata a runtime da `hopfield_bindings.py` tramite `ctypes.CDLL`.

### Dipendenze Python
Le dipendenze `Python` sono le seguenti:
\
```text
numpy>=1.26
scipy>=1.12
matplotlib>=3.8
scikit-learn>=1.4
spot>=2.11
```
\
La libreria `spot` (Spot Temporal Logic) richiede l'installazione tramite i pacchetti di sistema o tramite `conda`.
Su Kubuntu:
\
```bash
apt install spot libspot-dev python3-spot
```
oppure
```bash
conda install -c conda-forge spot
```

### Note su `Spot` e `ltlf2dfa`

La generazione delle formule LTLf avviene tramite `rendltl` dalla libreria `Spot` via `Python` bindings.
La traduzione in DFA usa `ltlf2dfa`, che richiede `lydia` come backend.
Si richiede quindi che sia installato `lydia` e che il percorso di installazione sia incluso nel percorso di ricerca di `Python`.

### Riproducibilità

Tutti gli esperimenti usano seed fisso (`seed=0`) per il generatore casuale `NumPy` (`np.random.default_rng(seed=0)`).
La rete classica usa `rng.permutation(n)` per la randomizzazione dell'ordine di aggiornamento, inizializzata con lo stesso seed.
I risultati sono riproducibili esattamente rieseguendo gli script degli esperimenti nello stesso ordine su dataset invariato.