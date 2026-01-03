# ICE Conscious

ICE Conscious è il **dominio cognitivo** dell’ecosistema ICE.

Non è un runtime.
Non è un orchestrator.
Non è un backend di storage.

ICE Conscious definisce **come il sistema pensa**, non **come esegue**.

---

## Ruolo nell’ecosistema ICE

ICE è composto da più livelli, ognuno con responsabilità precise:

* **ice-runtime** → esecuzione, processi, sessioni, I/O
* **ice-api** → contratti pubblici, IPC, UI, IDE
* **ice-engine** → coordinamento operativo (agent runtime, orchestrator)
* **ice-conscious** → *consapevolezza, conoscenza, memoria, semantica*

ICE Conscious è il punto in cui:

* la conoscenza prende forma
* il contesto viene costruito
* la rilevanza viene valutata
* la memoria viene strutturata
* la RAG diventa intenzionale

---

## Struttura concettuale

### Knowledge

Rappresenta ciò che il sistema **sa**:

* entità
* relazioni
* grafi
* query semantiche
* scoring di rilevanza e fiducia

### Memory

Rappresenta ciò che il sistema **ricorda**:

* memoria episodica (eventi)
* memoria semantica (concetti stabili)
* memoria di lavoro (contesto attivo)

### Embeddings

Definisce **come il significato diventa vettore**, senza dipendere da backend concreti.

### RAG

Orchestra:

* intento
* costruzione del contesto
* sessioni RAG
* prompt semantici

Senza orchestrator, senza agenti, senza I/O.

### ML

Contiene **logica cognitiva**, non pipeline operative:

* anomaly detection
* clustering concettuale
* scoring e confidence

### Storage

Solo **contratti astratti**.
Nessun database concreto vive qui.

---

## Principi fondamentali

* Domain-first, non infrastructure-first
* Nessuna dipendenza diretta da runtime o backend
* Tutto è testabile in isolamento
* Tutto è pensato per essere orchestrato da altri layer

ICE Conscious è ciò che rende ICE **più di un sistema esecutivo**:
è ciò che lo rende **consapevole**.
