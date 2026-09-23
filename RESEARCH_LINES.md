# Marco de investigación — preguntas y hipótesis de error sobre una ejecución SAS, en local

Estado: propuesta, 2026-09. Sustituye el planteamiento anterior (diff entre dos versiones).
Documento autocontenido: recoge el diagnóstico del repo, la literatura, las decisiones
tomadas, el diseño, las líneas de investigación y las condiciones de éxito.

---

## 1. Objetivo

Una herramienta **local (≤ 32 GB RAM)** a la que el analista entrega **una ejecución**
de un proyecto SAS Enterprise Guide 8.6 (`.egp`) y sus tablas, y a la que puede
preguntar en lenguaje natural:

- "¿Por qué el ciclo X tiene LGD = 0,9?" → explicación del valor.
- "¿Dónde se pierden / duplican filas?" → paso responsable.
- "¿Qué ciclos incumplen la regla R?" → conjunto + patrón.

La salida son **hipótesis de error**, cada una respaldada por una **consulta ejecutada**
y su resultado. Si falta información, la herramienta lo dice y pide lo concreto que falta.

**Límite por construcción.** Con una sola ejecución se puede explicar *qué pasó*, no
afirmar que *está mal*, salvo que la pregunta, una regla o una fuente externa aporte la
expectativa. La herramienta debe distinguir "coherente con el código" de "correcto".

---

## 2. Diagnóstico del repo actual (verificado en código)

| Afirmación | Realidad | Consecuencia |
|---|---|---|
| Éxito 0,70 en test | La SQL gold es `defect.oracle_sql` literal (`slm/episodes.py:gold_trace`); test = mismas 67 clases, otras semillas | Mide recuerdo de plantillas, no diagnóstico |
| Best-of-N 0,90 | Selecciona con el oráculo del defecto plantado (`slm/evaluate.py:run_best_of_n`) | No existe en producción; es pass@6 |
| Lee `.egp` | Zip falso con `manifest.json`; esquema y linaje hardcodeados (`slm/lineage.py:_LAYERS`, `examples/app/model_agent.py`) | El código SAS real nunca llega al modelo |
| Pregunta en lenguaje natural | Entrada = una fila sospechosa | No hay comprensión de preguntas ni razonamiento sobre conjuntos |
| Modelo 0,5B | Una sola llamada que produce todo el diagnóstico | Demasiado pequeño para un agente con herramientas |

**Reutilizable:** patrón de verificación por ejecución (`slm/oracle.py`), backend
llama.cpp (`slm/backends.py`), catálogo de reglas intra-tabla (`vendor/defect_catalog.py`,
que encaja bien en el alcance de una ejecución), demo SSE, ruta de exportación GGUF.

---

## 3. Qué dice la literatura (resumen)

- **Diagnóstico a partir de quejas es un problema de BD antes que de LLM.** QFix
  localiza la consulta que introdujo un error dados el historial y las quejas; Rain /
  Reptile generalizan las quejas; DIFF explica qué atributos separan dos conjuntos.
  La why/why-not provenance (por qué aparece o falta un dato) está muy estudiada.
- **Depurar SQL con LLMs es difícil incluso a 14B.** BIRD-CRITIC: Bird-Fixer
  (Qwen2.5-Coder-14B, agente entrenado) resuelve ~38 % de incidencias reales.
- **RL con recompensa ejecutable ayuda a modelos pequeños**, con ganancias modestas a ~4B
  (Reward-SQL, ReToolSQL, SERL-SQL, AGRO-SQL, 2025–26).
- **Agente ↔ SAS ya está resuelto:** SASPy + agentes (blog SAS jul-2026, LangChain sobre
  Viya) y servidores MCP (`sas-mcp`, `sastool-mcp`, `MCP4SAS`, el oficial de Viya).
- **Más contexto no implica mejor abstención:** con RAG los modelos se abstienen menos y
  fallan con más confianza (*Sufficient Context*, ICLR'25).
- **Hueco:** no hay trabajo publicado de agentes con modelos pequeños que diagnostiquen
  ejecuciones SAS. Confianza media (arXiv no accesible a texto completo desde aquí).

**Conclusión:** la arquitectura que funciona es *núcleo determinista (linaje, logs,
ejecución) + modelo en los bordes (entender la pregunta, plantear hipótesis, explicar)*.

---

## 4. Decisiones tomadas

| Tema | Decisión | Motivo |
|---|---|---|
| Alcance | Una ejecución, no diff entre versiones | Caso de uso real del analista |
| Entrada | `.egp` (zip: `project.xml` + nodos de código; logs si se guardaron) | Contiene código, flujo y, opcionalmente, logs; **nunca datos** |
| Acceso a datos | SASPy (usuario solo lectura) → snapshot local en DuckDB | Consultas fuera de producción, auditables |
| Tablas intermedias | Persistir `WORK` en librería permanente o reejecutar con esa opción | Sin intermedios no hay trazado hacia atrás |
| Excel | Solo como intercambio, no como almacén | Límite 1.048.576 filas, 15 dígitos, IDs y fechas SAS, missing `.` |
| Automatización | 1) SASPy, 2) API de automatización de EG, 3) batch `sas -sysin`, 4) RPA/ratón solo como último recurso | El ratón es frágil y lento, y un modelo local no lo maneja bien |
| Modelo | `Qwen3.5-9B` Q4_K_M (~6 GB) en llama.cpp CPU; `Qwen3.5-4B` alternativa | El 35B MoE no deja margen en 32 GB compartidos |
| Alternativa si se vetan modelos chinos | Gemma 4 / IBM Granite | Política habitual en banca; algo peor en agentes |
| Entrenamiento | Ninguno hasta que lo justifique RL3; las GPUs (2×5060) solo para QLoRA → GGUF | Afinar cambia comportamiento, no memoria ni velocidad |

**Sobre acumular gradientes en RAM (descartado por aritmética):** el fine-tune completo de
35B necesita ~70 GB solo de gradientes bf16 y ~560 GB con Adam. Con LoRA los gradientes
son < 1 % y ya caben en GPU; el cuello de botella son los pesos congelados (la versión
sensata es offload de pesos, ZeRO-Offload, pagando ~20 GB de PCIe por pasada).

**Presupuesto de memoria objetivo:** modelo 6–7 GB · DuckDB ≤ 8 GB (`memory_limit`) ·
Python/SASPy 1–2 GB · SO 4–6 GB → **~20–23 GB**.

---

## 5. Arquitectura

```
pregunta
 → anclas            (tablas, columnas, claves, pasos, periodos, valores)
 → recuperación      grafo de linaje desde las anclas; índice de log por paso;
                     BM25/denso solo sobre documentación en prosa
 → recorte           a presupuesto de tokens (RL1)
 → razonamiento      hipótesis en memoria de trabajo
 → ¿suficiente?      (RL2) ── no → herramienta o pregunta al usuario ─┐
 → consulta de confirmación (DuckDB, solo lectura) → veredicto        │
 → respuesta | abstención con "qué falta"  ←──────────────────────────┘
```

**Herramientas:** `schema()`, `lineage(campo)`, `log(paso)` (observaciones, avisos
`MERGE … repeats of BY values`, `Missing values were generated`, `uninitialized`,
`truncated`, `converted`), `query(sql)` (solo lectura, límite de filas),
`profile(tabla, col)`.

**Qué va en cada tipo de recuperación:**
- Código y linaje → recorrido del grafo (la similitud por embeddings recupera texto
  parecido, no causalmente relacionado).
- Log → índice estructurado por paso.
- Metodología / diccionario / reglas → BM25 o denso (el único "RAG clásico").
- Datos → nunca en contexto; solo resultados agregados de consultas.

**Memoria de contexto:**
- Memoria de trabajo = un JSON reescrito cada turno:
  `{pregunta, anclas, pasos_visitados, hipotesis[{afirmacion, paso, cols, sql, veredicto}], falta[]}`.
  El historial en bruto se descarta.
- Presupuesto por turno: ~1k sistema/herramientas + 0,5k memoria + 2k contexto + 0,5k
  último resultado (truncado).
- Resúmenes solo de evidencia verificada, nunca del razonamiento del modelo.
- Memoria entre sesiones (opcional): casos (pregunta, camino, causa) **validados por un
  analista**; lo no validado no se guarda.

---

## 6. Evaluación (antes que cualquier modelo)

**Bench-real (principal).** 30 → 100 preguntas reales ya resueltas. Por ítem: pregunta,
respuesta/hipótesis correcta, **conjunto de evidencia gold** (nodos de código, líneas de
log, tablas/columnas, consulta de confirmación) y si hacía falta más información.

**Bench-mut (secundario, escalable).** Mutar un paso SAS (clave de join, `<`/`<=`,
semántica de missing, clave de `NODUPKEY`, periodo desfasado, `LENGTH`), ejecutar y generar
la pregunta a partir del síntoma. Verdad = paso mutado. Partición por operador y por
proyecto, nunca solo por semilla.

**Métricas.**
- Respuesta: hipótesis correcta (paso + columna) **confirmada por consulta ejecutada**.
- Contexto: recall de evidencia gold a 1k/2k/4k tokens; tokens por turno.
- Suficiencia: curva exactitud–cobertura, tasa de información omitida, llamadas innecesarias.
- Coste: tiempo por pregunta en la CPU objetivo; pico de RSS.
- Negocio: minutos ahorrados frente al analista en las mismas preguntas.

---

## 7. Líneas de investigación

### RL1 — Construcción y recorte de contexto (GLiNER2.5 como candidato)

No existe "GLiNER 5.2"; la versión actual es **GLiNER2.5** (Fastino, Apache-2.0: 74M /
0,2B / 0,3B; NER + clasificación + extracción estructurada guiada por esquema; CPU).

- **Encaja bien:** extraer anclas de la pregunta y de la documentación → arranque del
  recorrido por el grafo.
- **Encaja mal:** relevancia condicionada a la pregunta sobre un top-100. GLiNER clasifica
  contra etiquetas; la relevancia pregunta–fragmento es tarea de pares (cross-encoder /
  podador tipo Provence).
- **Desconocido:** rendimiento con identificadores SAS (`LGD_FINAL`, `WORK.T_L3`).

Variantes (mismo top-100, mismo presupuesto):
1. Diccionario de nombres del esquema + grafo (sin ML) — referencia.
2. Anclas GLiNER2.5 + grafo.
3. (2) + reranker cross-encoder (~0,6B) → top-k.
4. (2) + podador por frases (Provence/XProvence).
5. (2) + clasificación GLiNER2.5 sobre `pregunta+fragmento`.

**Abandono:** si (1) empata con la mejor, fuera el recorte con ML.

### RL2 — ¿Necesitamos más información? (control de suficiencia)

1. **Checklist determinista** (filtro obligatorio): ¿camino de linaje completo cargado?
   ¿cada hipótesis tiene consulta ejecutada? ¿hay expectativa para juzgar "mal"?
2. **Autoevaluación del modelo:** acción `need_more_info{que, por_que}` antes de responder.
3. **Clasificador pequeño de suficiencia** sobre `(pregunta, contexto)` — puede ser
   GLiNER2.5 o un encoder afinado.
4. Combinación: (1) como filtro duro, (2)/(3) como señal.

Si falta información: buscar (vecino en el grafo, paso del log, perfil) o **una** pregunta
concreta al usuario (normalmente: el valor esperado).
**Abandono:** si el checklist solo iguala a (4), se queda el checklist.

### RL3 — ¿Merece la pena afinar?

Variantes: 9B base; 4B base; 9B QLoRA destilado de trayectorias verificadas (profesor
mayor en las GPUs; solo se guardan trayectorias cuya consulta confirma). Exportar a GGUF y
medir en CPU.
**Umbral:** entrenar solo si el 9B base < 60–70 % en Bench-real **y** los fallos son de
comportamiento (formato de herramientas, calidad de hipótesis), no de contexto (→ RL1) ni
de expectativa ausente (→ RL2).

---

## 8. Cómo conseguir que tenga éxito

### 8.1 Definición de éxito (acordarla antes de empezar)

- ≥ 70 % de las preguntas de Bench-real con hipótesis correcta **y** confirmada por consulta.
- ≤ 5 % de respuestas afirmativas incorrectas (mejor abstenerse que inventar).
- ≤ 2–3 min por pregunta en la máquina de 32 GB.
- Ahorro medible de tiempo frente al analista en esas mismas preguntas.
- Aprobado por IT / seguridad para uso con datos reales.

Umbrales orientativos: el equipo debe fijarlos antes de medir, no después.

### 8.2 Principio rector: atacar primero la mayor incertidumbre

El riesgo no está en el modelo sino en el acceso y en los datos. Orden:

| Fase | Entregable | Decisión (go / no-go) |
|---|---|---|
| 1. Acceso | SASPy conecta al servidor de EG 8.6 con usuario de solo lectura; `PROC CONTENTS` + log devueltos. Comprobar si los `.egp` guardan logs | Sin acceso → rediseñar sobre exportaciones manuales, alcance menor |
| 2. Parser | `.egp` → grafo de linaje + índice de log; **cobertura** (% pasos/columnas resueltos; macros, `%include`, código dinámico) | Cobertura < ~80 % en proyectos reales → invertir en el parser antes que en nada más |
| 3. Benchmark | Bench-real v0 (30 preguntas) + generador Bench-mut | Sin preguntas reales no hay forma de saber si funciona |
| 4. Referencia sin LLM | Herramientas deterministas + plantillas de informe sobre Bench-real | Fija el listón que el modelo debe superar |
| 5. Bucle con modelo | 9B base + herramientas + memoria + checklist | ¿Supera claramente a la fase 4? |
| 6. Experimentos | RL1 y RL2 en paralelo | Quedarse con lo más simple que no pierda |
| 7. Afinado | RL3 solo si se cumple el umbral | En la mayoría de escenarios, no hará falta |

Duración estimada: 8–12 semanas a tiempo parcial. Es una conjetura: depende sobre todo de
la fase 1 (IT) y de la calidad del código SAS real.

### 8.3 Factores críticos

- **Acotar tipos de pregunta.** Empezar con los tres de §1; añadir más solo con datos.
- **Toda hipótesis se confirma ejecutando.** Es la defensa principal contra las alucinaciones.
- **Evaluación desde el primer día.** Cada cambio se mide contra Bench-real.
- **Humano en el bucle.** La herramienta propone y el analista decide; nunca corrige datos.
- **Gobernanza desde el diseño:** local, solo lectura, auditoría de cada consulta,
  minimización (agregados por defecto, filas solo por extracción explícita y enmascarada).
  Así se queda fuera del perímetro de cambios de modelos regulatorios.
- **Usuarios reales pronto.** Dos o tres analistas usando la fase 5 aportan más que
  cualquier experimento.

### 8.4 Modos de fallo y mitigación

| Fallo | Señal temprana | Mitigación |
|---|---|---|
| IT no permite SASPy | Fase 1 bloqueada | Exportación programada por macro (`PROC EXPORT` de pasos y claves) |
| Macros / código dinámico rompen el linaje | Cobertura del parser baja | Resolver macros con la ejecución real (`MPRINT`/`MLOGIC` en el log) en lugar de análisis estático |
| `WORK` no persistido | Falta de intermedios | Reejecución con libname permanente en entorno sandbox |
| Latencia en CPU | > 3 min por pregunta | Contexto más corto, 4B, caché de prompt en llama.cpp |
| Hipótesis inventadas | Afirmaciones sin consulta | Checklist obligatorio; sin consulta, no hay afirmación |
| Veto a modelos de origen chino | Revisión de compliance | Gemma 4 / Granite; la arquitectura no cambia |
| Ampliar el alcance | Nuevos casos de uso antes de la fase 5 | Congelar el alcance hasta superar Bench-real v0 |

---

## 9. Preguntas abiertas

- ¿Los `.egp` guardan los logs? Si no, ¿se puede reejecutar para capturarlos?
- Volumen típico de las tablas (10^4 / 10^6 / 10^8 filas).
- ¿Hay metodología o reglas de negocio escritas, o el código es la única especificación?
- ¿La máquina de 32 GB es la misma para el modelo y para el cliente SAS?
- Configuración exacta de las GPUs (16+8 o 12+12) si se llega a RL3.

---

## Fuentes

- QFix — https://arxiv.org/abs/1601.07539 ; Rain — https://arxiv.org/pdf/2004.05722 ;
  DIFF — http://www.vldb.org/pvldb/vol12/p419-abuzaid.pdf
- SWE-SQL / BIRD-CRITIC — https://arxiv.org/abs/2506.18951
- Reward-SQL — https://arxiv.org/html/2505.04671 ; ReToolSQL — https://arxiv.org/pdf/2608.27796 ;
  SERL-SQL — https://arxiv.org/html/2608.00485 ; AGRO-SQL — https://arxiv.org/pdf/2512.23366
- Sufficient Context (ICLR'25) — https://arxiv.org/abs/2411.06037
- XProvence — https://arxiv.org/pdf/2601.18886 ; Information Gain Pruning — https://arxiv.org/abs/2601.17532
- GLiNER2.5 — https://fastino.ai/blog/gliner2-5-span-free-information-extraction ;
  GLiNER2 — https://arxiv.org/html/2507.18546v1
- SASPy — https://sassoftware.github.io/saspy/ ; agentes con SAS —
  https://blogs.sas.com/content/sgf/2026/07/23/the-three-components-you-need-to-use-agentic-ai-with-sas/ ;
  MCP — https://github.com/sassoftware/sas-mcp-server , https://pypi.org/project/sas-mcp/0.1.0/ ,
  https://github.com/samiulhq/sastool-mcp , https://github.com/chengzhongshan/MCP4SAS
- Qwen3.5 — https://artificialanalysis.ai/articles/qwen3-5-small-models ;
  Qwen3.6-35B-A3B — https://qwen.ai/blog?id=qwen3.6-35b-a3b
- Spider 2.0 — https://arxiv.org/pdf/2411.07763 ; ELT-Bench — https://arxiv.org/pdf/2504.04808
