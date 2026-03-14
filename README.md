# Jugador Autónomo para HEX — Estrategia Técnica

## Descripción General

Motor de IA competitivo para el juego HEX implementado en `solution.py`. La clase `SmartPlayer` hereda de `Player` y utiliza una combinación de búsqueda en árbol de juego con heurísticas posicionales específicas del HEX para seleccionar jugadas óptimas dentro del límite de 5 segundos por turno.

## Estrategia: Negamax con Heurística de Conexión Bidireccional

### 1. Motor de Búsqueda

- **Negamax con Poda Alfa-Beta**: Versión simétrica del Minimax que explora el árbol de juego descartando ramas que no pueden mejorar el resultado.
- **Principal Variation Search (PVS)**: Optimiza la poda asumiendo que el primer movimiento evaluado (el mejor de la iteración anterior) es el mejor. Los movimientos restantes se buscan con ventana nula y solo se re-evalúan con ventana completa si superan la expectativa.
- **Late Move Reductions (LMR)**: Los movimientos tardíos en el ordenamiento (posición ≥ 4) que no son jugadas críticas se buscan a profundidad reducida, ahorrando tiempo sin sacrificar calidad.
- **Profundización Iterativa con Ventanas de Aspiración**: La búsqueda comienza a profundidad 1 y aumenta progresivamente. A partir de profundidad 3 se usa una ventana estrecha (±300) alrededor del score anterior. Esto garantiza que siempre haya un mejor movimiento disponible al agotarse el tiempo (budget de 4s).

### 2. Evaluación Heurística

- **Distancia de Dijkstra Bidireccional (Two-Distance)**: Calcula el camino más corto desde cada borde del jugador hasta cada celda. Costo: piedra propia = 0, vacía = 1, rival = ∞. La diferencia de distancias es el componente principal del score.
- **Conteo de Caminos (Path Count)**: En tableros ≤ 11×11, se cuenta cuántas celdas pertenecen al camino óptimo. Mayor redundancia = posición más robusta.
- **Puentes Virtuales**: Dos piedras propias separadas por dos celdas vacías compartidas forman una conexión virtual. Se detectan y puntúan en evaluación y ordenamiento de movimientos.

### 3. Optimizaciones de Rendimiento

- **Tabla de Transposición con Zobrist Hashing**: Cada estado del tablero tiene un hash de 64 bits calculado incrementalmente (XOR). Estados ya evaluados se almacenan con política de reemplazo por profundidad preferida (límite: 500,000 entradas).
- **Union-Find (DSU) Incremental**: Detección de victoria en O(α(N)) mediante conjuntos disjuntos con 4 nodos virtuales (LEFT, RIGHT, TOP, BOTTOM). Al colocar una piedra se une con vecinos del mismo color y con el borde correspondiente.
- **Killer Moves y History Heuristic**: Se guardan los 2 movimientos que causaron cortes beta en cada profundidad y se acumulan scores históricos por celda para mejorar el ordenamiento de movimientos.
- **Make/Undo In-Place**: Los movimientos se realizan y deshacen directamente sobre el tablero sin copias, guardando solo snapshots del DSU.

### 4. Adaptación por Tamaño de Tablero

- **Tableros pequeños (≤ 5)**: Branching factor alto (35), evaluación completa con puentes.
- **Tableros medianos (6–11)**: Branching factor moderado (22–28), conteo de caminos activo.
- **Tableros grandes (> 100)**: Fast-path disperso que evalúa solo celdas cercanas a piedras existentes usando scoring de conectividad, sin búsqueda en árbol.

### 5. Libro de Apertura

- **Primer movimiento**: Siempre juega en el centro del tablero.
- **Respuesta al centro rival**: Selecciona la mejor celda adyacente al centro, priorizando la alineación con el eje de conexión propio.

## Dependencias

Solo utiliza librerías de la biblioteca estándar de Python: `heapq`, `random`, `time`, `typing`, `collections`.