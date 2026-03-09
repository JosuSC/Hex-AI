"""
solution.py - Jugador Autónomo para el juego HEX

Implementa un agente inteligente para el juego de estrategia HEX usando:
  - Búsqueda Minimax con Poda Alfa-Beta
  - Profundización Iterativa con control de tiempo (< 5 segundos)
  - Evaluación heurística basada en distancia Dijkstra
  - Ordenamiento inteligente de movimientos para maximizar podas

Arquitectura:
  SmartPlayer ──► SearchEngine ──► evaluate_board (Dijkstra)
                                ──► order_moves
                                ──► check_connection (BFS)

Módulos:
  1. Adyacencia: Vecinos hexagonales en coordenadas even-r offset
  2. Evaluación: Heurística Dijkstra (distancia mínima de conexión)
  3. Búsqueda: Minimax + Poda Alfa-Beta + Profundización Iterativa
  4. SmartPlayer: Clase principal que hereda de Player
"""

import time
import heapq
from math import inf
from collections import deque

from board import HexBoard
from player import Player


# ================================================================
# CONSTANTES
# ================================================================

WIN_SCORE = 1_000_000  # Puntuación para victoria confirmada

# Direcciones de vecinos hexagonales en coordenadas even-r offset.
#
# En el sistema even-r, las filas PARES están desplazadas +0.5 en el eje x
# (visualmente a la derecha). Esto afecta qué columnas son "diagonalmente
# adyacentes" dependiendo de la paridad de la fila.
#
# Fila PAR  (r%2==0): NW(-1, 0) NE(-1,+1) W(0,-1) E(0,+1) SW(+1, 0) SE(+1,+1)
# Fila IMPAR(r%2==1): NW(-1,-1) NE(-1, 0) W(0,-1) E(0,+1) SW(+1,-1) SE(+1, 0)
EVEN_ROW_DIRS = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1))
ODD_ROW_DIRS = ((-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0))


# ================================================================
# MÓDULO DE ADYACENCIA (Even-R Hexagonal Layout)
# ================================================================

def get_neighbors(row, col, size):
    """Retorna los vecinos válidos de (row, col) en tablero hexagonal even-r."""
    dirs = EVEN_ROW_DIRS if row % 2 == 0 else ODD_ROW_DIRS
    result = []
    for dr, dc in dirs:
        nr, nc = row + dr, col + dc
        if 0 <= nr < size and 0 <= nc < size:
            result.append((nr, nc))
    return result


def build_neighbor_table(size):
    """
    Pre-calcula la tabla completa de adyacencia para un tablero de tamaño N×N.
    Evita recomputar vecinos en cada llamada durante la búsqueda,
    convirtiendo get_neighbors de O(6) con branches a O(1) lookup.

    Retorna: lista 2D donde table[r][c] = tupla de vecinos válidos de (r,c).
    """
    table = [[None] * size for _ in range(size)]
    for r in range(size):
        dirs = EVEN_ROW_DIRS if r % 2 == 0 else ODD_ROW_DIRS
        for c in range(size):
            neighbors = []
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    neighbors.append((nr, nc))
            table[r][c] = tuple(neighbors)
    return table


# ================================================================
# UTILIDADES DE TABLERO
# ================================================================

def get_empty_cells(board_matrix, size):
    """Retorna lista de celdas vacías (valor == 0) como tuplas (row, col)."""
    return [
        (r, c)
        for r in range(size)
        for c in range(size)
        if board_matrix[r][c] == 0
    ]


def apply_move(board_matrix, move, player):
    """
    Genera un nuevo estado de tablero con el movimiento aplicado.
    Copia solo las filas necesarias (shallow copy por fila).
    No modifica el tablero original (inmutabilidad para el árbol de búsqueda).
    """
    new_board = [row[:] for row in board_matrix]
    new_board[move[0]][move[1]] = player
    return new_board


# ================================================================
# DETECCIÓN DE VICTORIA (BFS)
# ================================================================

def check_connection(board_matrix, size, player, ntable):
    """
    Verifica si el jugador ha completado su conexión entre sus dos bordes
    usando BFS sobre celdas propias.

    Conexiones:
      - Jugador 1: izquierda (col=0) → derecha  (col=N-1)  [horizontal]
      - Jugador 2: arriba   (row=0) → abajo   (row=N-1)  [vertical]

    Incluye pre-check rápido: si el jugador no tiene piezas en AMBOS bordes,
    la conexión es imposible y evitamos el BFS completo.
    """
    if player == 1:
        # Pre-check: ¿hay piezas en ambos bordes (izquierdo y derecho)?
        has_start = False
        starts = []
        for r in range(size):
            if board_matrix[r][0] == player:
                has_start = True
                starts.append((r, 0))
        if not has_start:
            return False
        if not any(board_matrix[r][size - 1] == player for r in range(size)):
            return False

        # BFS desde el borde izquierdo
        visited = set(starts)
        queue = deque(starts)
        while queue:
            r, c = queue.popleft()
            if c == size - 1:
                return True
            for nr, nc in ntable[r][c]:
                if (nr, nc) not in visited and board_matrix[nr][nc] == player:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    else:  # player == 2
        # Pre-check: ¿hay piezas en ambos bordes (superior e inferior)?
        has_start = False
        starts = []
        for c in range(size):
            if board_matrix[0][c] == player:
                has_start = True
                starts.append((0, c))
        if not has_start:
            return False
        if not any(board_matrix[size - 1][c] == player for c in range(size)):
            return False

        # BFS desde el borde superior
        visited = set(starts)
        queue = deque(starts)
        while queue:
            r, c = queue.popleft()
            if r == size - 1:
                return True
            for nr, nc in ntable[r][c]:
                if (nr, nc) not in visited and board_matrix[nr][nc] == player:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    return False


# ================================================================
# MÓDULO DE EVALUACIÓN HEURÍSTICA (Dijkstra)
# ================================================================

def dijkstra_distance(board_matrix, size, player, ntable):
    """
    Calcula el costo mínimo para que un jugador complete su conexión
    usando el algoritmo de Dijkstra sobre el grafo hexagonal.

    Modelo de costos por celda:
      - Celda propia (player):   0 (ya colocada, tránsito libre)
      - Celda vacía (0):         1 (requiere una jugada futura)
      - Celda del oponente:      ∞ (bloqueada, intransitable)

    Fuentes y destinos:
      - Jugador 1: fuentes = col 0, destino = col N-1  (horizontal)
      - Jugador 2: fuentes = row 0, destino = row N-1  (vertical)

    Retorna: costo mínimo del camino (int), o inf si no existe.
             Menor costo = más cerca de completar la conexión.
    """
    opponent = 3 - player
    dist = [[inf] * size for _ in range(size)]
    heap = []  # Min-heap: (costo, fila, columna)

    if player == 1:
        # Inicializar fuentes desde el borde izquierdo (col=0)
        for r in range(size):
            cell = board_matrix[r][0]
            if cell == opponent:
                continue
            cost = 0 if cell == player else 1
            if cost < dist[r][0]:
                dist[r][0] = cost
                heapq.heappush(heap, (cost, r, 0))

        # Expansión Dijkstra hacia el borde derecho (col=N-1)
        while heap:
            d, r, c = heapq.heappop(heap)
            if d > dist[r][c]:
                continue  # Nodo ya procesado con un costo menor
            if c == size - 1:
                return d  # Alcanzamos el destino
            for nr, nc in ntable[r][c]:
                cell = board_matrix[nr][nc]
                if cell == opponent:
                    continue
                w = 0 if cell == player else 1
                nd = d + w
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))

    else:  # player == 2
        # Inicializar fuentes desde el borde superior (row=0)
        for c in range(size):
            cell = board_matrix[0][c]
            if cell == opponent:
                continue
            cost = 0 if cell == player else 1
            if cost < dist[0][c]:
                dist[0][c] = cost
                heapq.heappush(heap, (cost, 0, c))

        # Expansión Dijkstra hacia el borde inferior (row=N-1)
        while heap:
            d, r, c = heapq.heappop(heap)
            if d > dist[r][c]:
                continue
            if r == size - 1:
                return d
            for nr, nc in ntable[r][c]:
                cell = board_matrix[nr][nc]
                if cell == opponent:
                    continue
                w = 0 if cell == player else 1
                nd = d + w
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))

    return inf  # No existe camino (completamente bloqueado)


def evaluate_board(board_matrix, size, my_player, opponent, ntable):
    """
    Función heurística de evaluación del tablero para estados no terminales.

    Fórmula:
      h(n) = (dist_oponente - dist_propia) × 100 + bonus_centro

    Componentes:
      1. Diferencia de distancias Dijkstra: mide qué tan cerca está cada
         jugador de completar su conexión. Un valor positivo indica que
         estamos más cerca de ganar que el oponente.

      2. Bonus de centro: las celdas centrales del tablero tienen mayor
         valor estratégico en HEX (mayor conectividad y flexibilidad).
         Se premia el control del centro y se penaliza el del oponente.

    Retorna: float donde positivo = ventaja nuestra, negativo = ventaja rival.
    """
    my_dist = dijkstra_distance(board_matrix, size, my_player, ntable)
    opp_dist = dijkstra_distance(board_matrix, size, opponent, ntable)

    # Casos extremos: un jugador completamente bloqueado
    if my_dist == inf and opp_dist == inf:
        return 0
    if opp_dist == inf:
        return WIN_SCORE // 2
    if my_dist == inf:
        return -(WIN_SCORE // 2)

    # Componente principal: diferencia de distancias de conexión
    score = (opp_dist - my_dist) * 100

    # Componente secundaria: bonus por control del centro del tablero
    center = (size - 1) / 2.0
    for r in range(size):
        for c in range(size):
            cell = board_matrix[r][c]
            if cell == 0:
                continue
            # Distancia Manhattan al centro, convertida en peso positivo
            cdist = abs(r - center) + abs(c - center)
            weight = max(0, size - cdist)
            if cell == my_player:
                score += weight
            else:
                score -= weight

    return score


# ================================================================
# ORDENAMIENTO DE MOVIMIENTOS
# ================================================================

def order_moves(moves, board_matrix, size, player, ntable):
    """
    Ordena los movimientos candidatos para maximizar la eficiencia
    de la poda alfa-beta. Un buen orden permite podar más ramas
    y explorar el árbol a mayor profundidad.

    Heurística de ordenamiento (mayor prioridad primero):
      1. Centralidad: celdas cercanas al centro del tablero
         (mayor conectividad y valor estratégico)
      2. Conectividad: celdas adyacentes a piezas propias
         (extensión de cadenas existentes)
      3. Bloqueo: celdas adyacentes a piezas del oponente
         (impedir conexiones rivales)
      4. Eje: alineación con la dirección de conexión del jugador
         (avance hacia el objetivo)
    """
    center = (size - 1) / 2.0
    opponent = 3 - player
    scored = []

    for r, c in moves:
        score = 0.0

        # 1. Centralidad: penalizar distancia al centro
        dist_c = abs(r - center) + abs(c - center)
        score += (size - dist_c) * 3

        # 2-3. Conectividad y bloqueo
        for nr, nc in ntable[r][c]:
            cell = board_matrix[nr][nc]
            if cell == player:
                score += 6  # Extensión de cadena propia
            elif cell == opponent:
                score += 4  # Bloqueo de cadena rival

        # 4. Alineación con el eje de conexión
        if player == 1:  # Horizontal: premiar avance en columnas
            score += (size - abs(c - center)) * 1.5
        else:  # Vertical: premiar avance en filas
            score += (size - abs(r - center)) * 1.5

        scored.append((-score, r, c))  # Negativo para orden descendente

    scored.sort()
    return [(r, c) for _, r, c in scored]


# ================================================================
# MOTOR DE BÚSQUEDA: MINIMAX CON PODA ALFA-BETA
# ================================================================

class SearchEngine:
    """
    Motor de búsqueda adversarial que combina:
      - Minimax recursivo para modelar la interacción MAX (nosotros) vs MIN (rival)
      - Poda Alfa-Beta para eliminar ramas que no pueden influir en la decisión
      - Profundización Iterativa para gestión adaptativa del tiempo
      - Límite de ramificación para control de complejidad en tableros grandes
    """

    def __init__(self, my_player, size, ntable, time_limit):
        """
        Args:
            my_player: Nuestro número de jugador (1 o 2)
            size: Tamaño N del tablero N×N
            ntable: Tabla pre-calculada de vecinos
            time_limit: Tiempo máximo en segundos para la búsqueda
        """
        self.my_player = my_player
        self.opponent = 3 - my_player
        self.size = size
        self.ntable = ntable
        self.time_limit = time_limit
        self.start_time = time.time()
        self.nodes = 0
        self.timed_out = False

        # Límite de ramificación adaptado al tamaño del tablero.
        # Tableros más grandes necesitan mayor poda para mantenerse
        # dentro del presupuesto de tiempo.
        if size <= 5:
            self.max_branch = 30
        elif size <= 7:
            self.max_branch = 25
        elif size <= 11:
            self.max_branch = 20
        else:
            self.max_branch = 15

    def _check_time(self):
        """Lanza TimeoutError si se agotó el presupuesto de tiempo."""
        if self.timed_out:
            raise TimeoutError
        if time.time() - self.start_time >= self.time_limit:
            self.timed_out = True
            raise TimeoutError

    def search(self, board_matrix):
        """
        Profundización Iterativa: ejecuta Minimax a profundidad 1, 2, 3, ...
        hasta agotar el tiempo. Siempre retorna el mejor resultado de la
        profundidad completada más alta.

        Ventajas de la profundización iterativa:
          - Garantiza una respuesta válida incluso con poco tiempo
          - La información de iteraciones previas mejora el ordenamiento
          - Detecta victorias forzadas a profundidades bajas rápidamente

        Retorna: tupla (row, col) con la mejor jugada encontrada.
        """
        empty = get_empty_cells(board_matrix, self.size)
        if not empty:
            return None

        # Ordenar movimientos candidatos en la raíz
        moves = order_moves(
            empty, board_matrix, self.size, self.my_player, self.ntable
        )

        # Limitar ramificación en la raíz
        root_moves = moves[:self.max_branch] if len(moves) > self.max_branch else moves

        best_move = root_moves[0]
        depth = 1
        max_depth = len(empty)

        while depth <= max_depth:
            try:
                move, score = self._search_root(board_matrix, root_moves, depth)
                best_move = move

                # Victoria o derrota forzada: no buscar más profundo
                if abs(score) >= WIN_SCORE - 10000:
                    break

                depth += 1
            except TimeoutError:
                break

        return best_move

    def _search_root(self, board_matrix, moves, depth):
        """
        Búsqueda en el nivel raíz: evalúa cada movimiento candidato
        y retorna el mejor según Minimax con poda alfa-beta.

        Se separa del _minimax general porque:
          - Necesitamos rastrear cuál movimiento tiene la mejor puntuación
          - Podemos verificar victoria inmediata sin recursión
        """
        best_score = -inf
        best_move = moves[0]
        alpha = -inf
        beta = inf

        for move in moves:
            self._check_time()

            new_board = apply_move(board_matrix, move, self.my_player)

            # Victoria inmediata: retornar sin buscar más profundo
            if check_connection(new_board, self.size, self.my_player, self.ntable):
                return move, WIN_SCORE

            # Recursión: turno del oponente (MIN)
            score = self._minimax(new_board, depth - 1, alpha, beta, False)

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, best_score)

        return best_move, best_score

    def _minimax(self, board_matrix, depth, alpha, beta, is_max):
        """
        Algoritmo Minimax con Poda Alfa-Beta.

        Parámetros:
          board_matrix: estado actual del tablero (después del último movimiento)
          depth: profundidad restante de búsqueda
          alpha: mejor valor garantizado para MAX en el camino actual (límite inferior)
          beta:  mejor valor garantizado para MIN en el camino actual (límite superior)
          is_max: True si es turno de MAX (nuestro jugador), False si es turno de MIN

        Poda Alfa-Beta:
          La poda ocurre cuando alpha >= beta, indicando que la rama actual
          no puede producir un resultado mejor que una alternativa ya encontrada:
            - En nodo MAX (is_max=True): si encontramos valor >= beta,
              MIN (el padre) nunca elegirá esta rama → poda beta
            - En nodo MIN (is_max=False): si encontramos valor <= alpha,
              MAX (el padre) nunca elegirá esta rama → poda alpha

        Retorna: puntuación heurística del estado.
        """
        self.nodes += 1

        # Control de tiempo periódico (cada ~512 nodos para minimizar overhead)
        if self.nodes & 511 == 0:
            self._check_time()

        player_now = self.my_player if is_max else self.opponent

        # Corte por profundidad: evaluar heurísticamente
        if depth <= 0:
            return evaluate_board(
                board_matrix, self.size,
                self.my_player, self.opponent, self.ntable
            )

        # Generar movimientos sucesores
        empty = get_empty_cells(board_matrix, self.size)
        if not empty:
            return 0  # Tablero lleno sin ganador (imposible en HEX teórico)

        moves = order_moves(
            empty, board_matrix, self.size, player_now, self.ntable
        )

        # Limitar ramificación con profundidad adaptativa:
        # Cerca de las hojas usamos menos movimientos para controlar
        # el crecimiento exponencial del árbol de búsqueda.
        if depth >= 3:
            branch_limit = self.max_branch
        elif depth == 2:
            branch_limit = min(self.max_branch, 15)
        else:
            branch_limit = min(self.max_branch, 10)
        if len(moves) > branch_limit:
            moves = moves[:branch_limit]

        if is_max:
            # Nodo MAX: buscar el movimiento que maximice la puntuación
            value = -inf
            for move in moves:
                new_board = apply_move(board_matrix, move, player_now)

                # Verificar si este movimiento gana la partida
                if check_connection(new_board, self.size, player_now, self.ntable):
                    return WIN_SCORE + depth  # Preferir ganar antes (más profundidad restante)

                score = self._minimax(new_board, depth - 1, alpha, beta, False)
                if score > value:
                    value = score
                if value > alpha:
                    alpha = value
                if alpha >= beta:
                    break  # Poda Beta: MIN no elegirá esta rama
            return value
        else:
            # Nodo MIN: buscar el movimiento que minimice la puntuación
            value = inf
            for move in moves:
                new_board = apply_move(board_matrix, move, player_now)

                # Verificar si el oponente gana con este movimiento
                if check_connection(new_board, self.size, player_now, self.ntable):
                    return -(WIN_SCORE + depth)  # Preferir perder más tarde

                score = self._minimax(new_board, depth - 1, alpha, beta, True)
                if score < value:
                    value = score
                if value < beta:
                    beta = value
                if alpha >= beta:
                    break  # Poda Alpha: MAX no elegirá esta rama
            return value


# ================================================================
# CLASE PRINCIPAL: SmartPlayer
# ================================================================

class SmartPlayer(Player):
    """
    Jugador autónomo para HEX basado en búsqueda adversarial.

    Hereda de Player e implementa play() para decidir la mejor jugada.
    No mantiene estado entre partidas (idempotente): toda la información
    se calcula localmente en cada llamada a play().

    Estrategia:
      1. Apertura: ocupar el centro del tablero (máxima conectividad)
      2. Medio juego: Minimax con poda alfa-beta y heurística Dijkstra
      3. Gestión de tiempo: profundización iterativa con límite de 4.5s
    """

    def play(self, board: HexBoard) -> tuple:
        """
        Decide la mejor jugada para el estado actual del tablero.

        Proceso:
          1. Identifica nuestro número de jugador (1 o 2)
          2. Pre-calcula la tabla de adyacencia hexagonal
          3. Verifica caso especial de apertura (centro del tablero)
          4. Ejecuta búsqueda adversarial con profundización iterativa

        Args:
            board: Estado actual del tablero HexBoard

        Returns:
            Tupla (row, col) con la posición de la celda seleccionada.
            Siempre será una celda vacía (valor 0).
        """
        size = board.size
        board_matrix = board.board

        # Identificar nuestro número de jugador
        my_player = self._identify_player(board_matrix, size)

        # Pre-calcular tabla de vecinos (determinista, solo depende de N)
        ntable = build_neighbor_table(size)

        # Estrategia de apertura: ocupar el centro del tablero.
        # El centro es la posición más fuerte en HEX por su máxima
        # conectividad hacia ambos bordes.
        empty_count = sum(
            1 for r in range(size) for c in range(size)
            if board_matrix[r][c] == 0
        )
        if empty_count >= size * size - 1:
            center = size // 2
            if board_matrix[center][center] == 0:
                return (center, center)
            # Si el centro está ocupado, elegir la mejor celda adyacente
            for nr, nc in ntable[center][center]:
                if board_matrix[nr][nc] == 0:
                    return (nr, nc)

        # Búsqueda adversarial con control de tiempo.
        # Se usa 4.0s como límite para dejar 1.0s de margen para overhead
        # del framework, construcción de la tabla de vecinos, etc.
        engine = SearchEngine(my_player, size, ntable, time_limit=4.0)
        best_move = engine.search(board_matrix)

        return best_move

    def _identify_player(self, board_matrix, size):
        """
        Determina si somos Jugador 1 o Jugador 2 de forma robusta.

        Estrategia de detección (en orden de prioridad):
          1. Buscar atributos comunes de la clase base Player
          2. Fallback: inferir del estado del tablero
             (Jugador 1 mueve primero → si #piezas iguales, somos J1)
        """
        # Intentar atributos comunes del framework
        for attr in ('player_id', 'player', 'id', 'color', 'number'):
            val = getattr(self, attr, None)
            if val in (1, 2):
                return val

        # Inferencia por conteo de piezas
        p1 = sum(row.count(1) for row in board_matrix)
        p2 = sum(row.count(2) for row in board_matrix)
        return 1 if p1 <= p2 else 2
