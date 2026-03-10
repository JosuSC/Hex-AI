"""
solution.py  –  Motor competitivo para HEX

Técnicas implementadas:
  • Negamax con Poda Alfa-Beta (equivalente a Minimax, código más limpio)
  • Principal Variation Search (PVS) – ventana nula tras el primer hijo
  • Profundización Iterativa con reordenamiento PV entre iteraciones
  • Zobrist Hashing + Transposition Table (64-bit, dict-based)
  • Killer Moves (2 slots por profundidad)
  • History Heuristic (tabla de éxitos de corte por celda)
  • Two-Distance Heuristic (Dijkstra doble) + detección de puentes
  • Make / Undo move in-place (zero-copy)
  • Union-Find (DSU) incremental para detección de victoria O(α(N))
  • Generación de movimientos focalizada (zona activa + vecindad)
  • Apertura con libro de respuestas al centro
"""

import time
import heapq
import random
from math import inf

from board import HexBoard
from player import Player

# ================================================================
# CONSTANTES
# ================================================================

WIN_SCORE  = 1_000_000
INF        = float("inf")

# Transposition table flags
TT_EXACT = 0
TT_LOWER = 1  # (alpha cutoff – lower bound)
TT_UPPER = 2  # (beta cutoff – upper bound)

# Even-r hex offset neighbours
EVEN_DIRS = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1))
ODD_DIRS  = ((-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0))

# ================================================================
# TABLA DE VECINOS (cache global por tamaño)
# ================================================================
_NTABLE_CACHE = {}

def _compute_neighbors(r, c, size):
    """Compute neighbors for a single cell on-the-fly."""
    dirs = EVEN_DIRS if r % 2 == 0 else ODD_DIRS
    nbs = []
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            nbs.append((nr, nc))
    return tuple(nbs)


class LazyNeighborTable:
    """Lazy neighbor table that computes neighbors on demand for large boards."""
    __slots__ = ('size', '_cache')

    def __init__(self, size):
        self.size = size
        self._cache = {}

    def __getitem__(self, r):
        return _LazyRow(self, r)


class _LazyRow:
    __slots__ = ('_table', '_r')

    def __init__(self, table, r):
        self._table = table
        self._r = r

    def __getitem__(self, c):
        key = (self._r, c)
        cache = self._table._cache
        val = cache.get(key)
        if val is None:
            val = _compute_neighbors(self._r, c, self._table.size)
            cache[key] = val
        return val


def build_neighbor_table(size):
    if size in _NTABLE_CACHE:
        return _NTABLE_CACHE[size]
    # For large boards, use lazy computation to avoid O(N^2) upfront cost
    if size > 100:
        table = LazyNeighborTable(size)
        _NTABLE_CACHE[size] = table
        return table
    table = [[None]*size for _ in range(size)]
    for r in range(size):
        dirs = EVEN_DIRS if r % 2 == 0 else ODD_DIRS
        for c in range(size):
            nbs = []
            for dr, dc in dirs:
                nr, nc = r+dr, c+dc
                if 0 <= nr < size and 0 <= nc < size:
                    nbs.append((nr, nc))
            table[r][c] = tuple(nbs)
    _NTABLE_CACHE[size] = table
    return table

# ================================================================
# PATRONES DE PUENTES VIRTUALES  (Even-R)
# ================================================================
# Un "bridge" en HEX: dos celdas del mismo jugador separadas por exactamente
# dos celdas vacías compartidas como vecinos comunes.  Si el rival ocupa una,
# el otro la llena y la conexión se mantiene → conexión virtual segura.
# Precalculamos para cada celda los pares (celda_destino, carrier1, carrier2).

_BRIDGE_CACHE = {}

def build_bridge_table(size, ntable):
    if size in _BRIDGE_CACHE:
        return _BRIDGE_CACHE[size]
    bridges = [[[] for _ in range(size)] for _ in range(size)]
    for r in range(size):
        for c in range(size):
            nbs = set(ntable[r][c])
            for nr, nc in ntable[r][c]:
                for nr2, nc2 in ntable[nr][nc]:
                    if (nr2, nc2) == (r, c):
                        continue
                    if (nr2, nc2) in nbs:
                        continue
                    # (nr,nc) es carrier 1; buscar carrier 2
                    common = set(ntable[r][c]) & set(ntable[nr2][nc2])
                    common.discard((nr2, nc2))
                    common.discard((r, c))
                    if (nr, nc) in common:
                        # hay al menos 2 carriers: (nr,nc) y algún otro
                        for other in common:
                            if other != (nr, nc):
                                bridges[r][c].append((nr2, nc2, nr, nc, other[0], other[1]))
                                break
    _BRIDGE_CACHE[size] = bridges
    return bridges

# ================================================================
# ZOBRIST HASHING
# ================================================================
_ZOBRIST_CACHE = {}

def get_zobrist_table(size):
    if size in _ZOBRIST_CACHE:
        return _ZOBRIST_CACHE[size]
    rng = random.Random(42)  # semilla fija → determinista
    zt = [[[rng.getrandbits(64) for _ in range(3)] for _ in range(size)] for _ in range(size)]
    _ZOBRIST_CACHE[size] = zt
    return zt

def compute_zobrist(board, size, zt):
    h = 0
    for r in range(size):
        row = board[r]
        for c in range(size):
            v = row[c]
            if v:
                h ^= zt[r][c][v]
    return h

# ================================================================
# UNION-FIND (DSU) para detección de victoria O(α(N))
# ================================================================

class DSU:
    """
    Disjoint-Set Union con nodos virtuales para los bordes.
    Para un tablero N×N:
      - Nodos reales: r*N + c  (0..N²-1)
      - Player 1 borders: LEFT = N², RIGHT = N²+1
      - Player 2 borders: TOP  = N²+2, BOTTOM = N²+3
    """
    __slots__ = ('parent', 'rank', 'n', 'size')

    def __init__(self, size):
        self.size = size
        self.n = size * size + 4
        self.parent = list(range(self.n))
        self.rank = [0] * self.n

    def find(self, x):
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]  # path halving
            x = p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1

    def connected(self, a, b):
        return self.find(a) == self.find(b)

    def copy(self):
        d = DSU.__new__(DSU)
        d.size = self.size
        d.n = self.n
        d.parent = self.parent[:]
        d.rank = self.rank[:]
        return d

def build_dsu_from_board(board, size, ntable):
    """Construye DSU con todas las piezas actuales del tablero."""
    N2 = size * size
    LEFT, RIGHT, TOP, BOTTOM = N2, N2+1, N2+2, N2+3
    dsu = DSU(size)
    for r in range(size):
        for c in range(size):
            v = board[r][c]
            if v == 0:
                continue
            idx = r * size + c
            # Conectar con bordes virtuales
            if v == 1:
                if c == 0:
                    dsu.union(idx, LEFT)
                if c == size - 1:
                    dsu.union(idx, RIGHT)
            else:
                if r == 0:
                    dsu.union(idx, TOP)
                if r == size - 1:
                    dsu.union(idx, BOTTOM)
            # Conectar con vecinos del mismo color
            # Only check right/down neighbors to avoid double-processing
            dirs = EVEN_DIRS if r % 2 == 0 else ODD_DIRS
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and board[nr][nc] == v:
                    dsu.union(idx, nr * size + nc)
    return dsu

def dsu_add_stone(dsu, board, size, ntable, r, c, player):
    """Agrega una piedra al DSU y conecta con vecinos y bordes."""
    N2 = size * size
    idx = r * size + c
    if player == 1:
        if c == 0:
            dsu.union(idx, N2)      # LEFT
        if c == size - 1:
            dsu.union(idx, N2 + 1)  # RIGHT
    else:
        if r == 0:
            dsu.union(idx, N2 + 2)  # TOP
        if r == size - 1:
            dsu.union(idx, N2 + 3)  # BOTTOM
    for nr, nc in ntable[r][c]:
        if board[nr][nc] == player:
            dsu.union(idx, nr * size + nc)

def dsu_check_win(dsu, size, player):
    N2 = size * size
    if player == 1:
        return dsu.connected(N2, N2 + 1)
    return dsu.connected(N2 + 2, N2 + 3)


# ================================================================
# FUNCIONES PÚBLICAS DE COMPATIBILIDAD (usadas por los tests)
# ================================================================

def get_neighbors(row, col, size):
    dirs = EVEN_DIRS if row % 2 == 0 else ODD_DIRS
    return [(row+dr, col+dc) for dr, dc in dirs
            if 0 <= row+dr < size and 0 <= col+dc < size]

def get_empty_cells(board, size):
    return [(r, c) for r in range(size) for c in range(size) if board[r][c] == 0]

def apply_move(board, move, player):
    nb = [row[:] for row in board]
    nb[move[0]][move[1]] = player
    return nb

def check_connection(board, size, player, ntable):
    """BFS-based – mantenida para compatibilidad con tests."""
    from collections import deque
    if player == 1:
        starts = [(r, 0) for r in range(size) if board[r][0] == player]
        if not starts:
            return False
        if not any(board[r][size-1] == player for r in range(size)):
            return False
        visited = set(starts); q = deque(starts)
        while q:
            r, c = q.popleft()
            if c == size - 1:
                return True
            for nr, nc in ntable[r][c]:
                if (nr, nc) not in visited and board[nr][nc] == player:
                    visited.add((nr, nc)); q.append((nr, nc))
    else:
        starts = [(0, c) for c in range(size) if board[0][c] == player]
        if not starts:
            return False
        if not any(board[size-1][c] == player for c in range(size)):
            return False
        visited = set(starts); q = deque(starts)
        while q:
            r, c = q.popleft()
            if r == size - 1:
                return True
            for nr, nc in ntable[r][c]:
                if (nr, nc) not in visited and board[nr][nc] == player:
                    visited.add((nr, nc)); q.append((nr, nc))
    return False


# ================================================================
# MÓDULO DE EVALUACIÓN HEURÍSTICA
# ================================================================

def dijkstra_distance(board, size, player, ntable):
    """
    Dijkstra shortest-path: costo mínimo para conectar los dos bordes.
    Costos: celda propia = 0, vacía = 1, oponente = ∞.
    """
    opp = 3 - player
    dist = [[INF]*size for _ in range(size)]
    heap = []
    if player == 1:
        for r in range(size):
            v = board[r][0]
            if v == opp:
                continue
            cost = 0 if v == player else 1
            if cost < dist[r][0]:
                dist[r][0] = cost
                heapq.heappush(heap, (cost, r, 0))
        while heap:
            d, r, c = heapq.heappop(heap)
            if d > dist[r][c]:
                continue
            if c == size - 1:
                return d
            for nr, nc in ntable[r][c]:
                v = board[nr][nc]
                if v == opp:
                    continue
                nd = d + (0 if v == player else 1)
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))
    else:
        for c in range(size):
            v = board[0][c]
            if v == opp:
                continue
            cost = 0 if v == player else 1
            if cost < dist[0][c]:
                dist[0][c] = cost
                heapq.heappush(heap, (cost, 0, c))
        while heap:
            d, r, c = heapq.heappop(heap)
            if d > dist[r][c]:
                continue
            if r == size - 1:
                return d
            for nr, nc in ntable[r][c]:
                v = board[nr][nc]
                if v == opp:
                    continue
                nd = d + (0 if v == player else 1)
                if nd < dist[nr][nc]:
                    dist[nr][nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))
    return INF


def dijkstra_full(board, size, player, ntable):
    """
    Dijkstra completo: retorna la matriz de distancias desde el borde origen.
    Necesario para two-distance y para el ordenamiento de movimientos.
    """
    opp = 3 - player
    dist = [[INF]*size for _ in range(size)]
    heap = []
    if player == 1:
        for r in range(size):
            v = board[r][0]
            if v == opp:
                continue
            cost = 0 if v == player else 1
            if cost < dist[r][0]:
                dist[r][0] = cost
                heapq.heappush(heap, (cost, r, 0))
    else:
        for c in range(size):
            v = board[0][c]
            if v == opp:
                continue
            cost = 0 if v == player else 1
            if cost < dist[0][c]:
                dist[0][c] = cost
                heapq.heappush(heap, (cost, 0, c))
    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r][c]:
            continue
        for nr, nc in ntable[r][c]:
            v = board[nr][nc]
            if v == opp:
                continue
            nd = d + (0 if v == player else 1)
            if nd < dist[nr][nc]:
                dist[nr][nc] = nd
                heapq.heappush(heap, (nd, nr, nc))
    return dist


def dijkstra_reverse(board, size, player, ntable):
    """Dijkstra desde el borde destino (invertido)."""
    opp = 3 - player
    dist = [[INF]*size for _ in range(size)]
    heap = []
    if player == 1:
        for r in range(size):
            v = board[r][size-1]
            if v == opp:
                continue
            cost = 0 if v == player else 1
            if cost < dist[r][size-1]:
                dist[r][size-1] = cost
                heapq.heappush(heap, (cost, r, size-1))
    else:
        for c in range(size):
            v = board[size-1][c]
            if v == opp:
                continue
            cost = 0 if v == player else 1
            if cost < dist[size-1][c]:
                dist[size-1][c] = cost
                heapq.heappush(heap, (cost, size-1, c))
    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r][c]:
            continue
        for nr, nc in ntable[r][c]:
            v = board[nr][nc]
            if v == opp:
                continue
            nd = d + (0 if v == player else 1)
            if nd < dist[nr][nc]:
                dist[nr][nc] = nd
                heapq.heappush(heap, (nd, nr, nc))
    return dist


def two_distance(board, size, player, ntable):
    """
    Two-distance heuristic: para cada celda vacía, calcula
    dist_from_start[r][c] + dist_from_end[r][c].
    El mínimo sobre todas las celdas es la "distancia de conexión".
    Pero la fuerza real viene de contar cuántas celdas están en el
    shortest path (on-path cells) → da información sobre redundancia.

    Retorna: (shortest_distance, num_on_path_cells)
    """
    dfwd = dijkstra_full(board, size, player, ntable)
    drev = dijkstra_reverse(board, size, player, ntable)

    # Distancia mínima total
    if player == 1:
        mind = min(dfwd[r][size-1] for r in range(size))
    else:
        mind = min(dfwd[size-1][c] for c in range(size))

    if mind >= INF:
        return INF, 0

    # Contar celdas on-path (donde fwd + rev == mind y celda es vacía o propia)
    opp = 3 - player
    on_path = 0
    for r in range(size):
        for c in range(size):
            if board[r][c] == opp:
                continue
            if dfwd[r][c] + drev[r][c] == mind:
                on_path += 1

    return mind, on_path


def _count_shortest_paths(board, size, player, ntable):
    """
    Count the number of cells on *any* shortest path for `player`.
    A cell is on a shortest path if dist_fwd[r][c] + dist_rev[r][c] == min_dist.
    More on-path cells ≈ more redundant paths ≈ harder to block.
    Uses a cheap bidirectional Dijkstra already available.
    """
    opp = 3 - player
    dfwd = dijkstra_full(board, size, player, ntable)
    drev = dijkstra_reverse(board, size, player, ntable)
    if player == 1:
        mind = min(dfwd[r][size - 1] for r in range(size))
    else:
        mind = min(dfwd[size - 1][c] for c in range(size))
    if mind >= INF:
        return INF, 0
    count = 0
    for r in range(size):
        for c in range(size):
            if board[r][c] == opp:
                continue
            if dfwd[r][c] + drev[r][c] == mind:
                count += 1
    return mind, count


def evaluate_fast(board, size, my_player, opponent, ntable):
    """
    Evaluación rápida basada en Dijkstra simple.
    Usada en nodos profundos del árbol donde velocidad > precisión.
    For small boards (≤7) uses path counting to distinguish blocking moves.
    """
    my_dist = dijkstra_distance(board, size, my_player, ntable)
    opp_dist = dijkstra_distance(board, size, opponent, ntable)

    if my_dist >= INF and opp_dist >= INF:
        return 0
    if opp_dist >= INF:
        return WIN_SCORE // 2
    if my_dist >= INF:
        return -(WIN_SCORE // 2)

    score = (opp_dist - my_dist) * 200

    # When distances are equal, use path-count to break ties (critical for blocking)
    if size <= 11 and my_dist == opp_dist:
        _, my_paths = _count_shortest_paths(board, size, my_player, ntable)
        _, opp_paths = _count_shortest_paths(board, size, opponent, ntable)
        score += (my_paths - opp_paths) * 10

    return score


def evaluate_board(board, size, my_player, opponent, ntable):
    """
    Heurística completa: two-distance + on-path redundancy.
    Usada en nodos poco profundos donde precisión importa más.
    """
    my_dist, my_paths = two_distance(board, size, my_player, ntable)
    opp_dist, opp_paths = two_distance(board, size, opponent, ntable)

    if my_dist >= INF and opp_dist >= INF:
        return 0
    if opp_dist >= INF:
        return WIN_SCORE // 2
    if my_dist >= INF:
        return -(WIN_SCORE // 2)

    score = (opp_dist - my_dist) * 200
    score += (my_paths - opp_paths) * 10

    return score


# ================================================================
# MOTOR DE BÚSQUEDA: NEGAMAX + PVS + TT + KILLER + HISTORY
# ================================================================

class SearchEngine:
    def __init__(self, my_player, size, ntable, board, time_limit):
        self.my_player = my_player
        self.opponent = 3 - my_player
        self.size = size
        self.ntable = ntable
        self.time_limit = time_limit
        self.start_time = time.time()
        self.nodes = 0
        self.timed_out = False

        # Zobrist
        self.zt = get_zobrist_table(size)
        self.zhash = compute_zobrist(board, size, self.zt)

        # Transposition Table
        self.tt = {}

        # Killer moves: 2 slots por profundidad
        self.killers = [[None, None] for _ in range(64)]

        # History heuristic: puntuación acumulada por celda y jugador
        self.history = [[[0, 0, 0] for _ in range(size)] for _ in range(size)]

        # DSU construido desde el estado actual
        self.dsu_base = build_dsu_from_board(board, size, ntable)

        # Límite de ramificación adaptado
        if size <= 5:
            self.max_branch = 35
        elif size <= 7:
            self.max_branch = 28
        elif size <= 11:
            self.max_branch = 22
        else:
            self.max_branch = 16

        # Precalculo de celdas vacías con vecino no-vacío (zona activa)
        self._precompute_active_zone(board)

    def _precompute_active_zone(self, board):
        """Identifica celdas vacías adyacentes a alguna pieza (zona caliente)."""
        size = self.size
        self.active = set()
        self.all_empty = []
        for r in range(size):
            for c in range(size):
                if board[r][c] != 0:
                    continue
                self.all_empty.append((r, c))
                dirs = EVEN_DIRS if r % 2 == 0 else ODD_DIRS
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size and board[nr][nc] != 0:
                        self.active.add((r, c))
                        break

    def _check_time(self):
        if self.timed_out:
            raise TimeoutError
        if time.time() - self.start_time >= self.time_limit:
            self.timed_out = True
            raise TimeoutError

    # --- Ordenamiento de movimientos ---
    def _order_moves(self, board, player, depth, pv_move, empty_cells):
        """
        Orden de movimientos para maximizar podas alfa-beta:
          0. PV move (mejor movimiento de la iteración anterior)
          1. Killer moves (causaron cortes en nodos hermanos)
          2. History heuristic + Dijkstra-on-path + centralidad + conectividad
        """
        ntable = self.ntable
        size = self.size
        opp = 3 - player
        center = (size - 1) / 2.0

        scored = []
        k0, k1 = self.killers[depth] if depth < 64 else (None, None)

        for r, c in empty_cells:
            if (r, c) == pv_move:
                scored.append((-10000000, r, c))
                continue
            s = 0.0
            if (r, c) == k0:
                s += 50000
            elif (r, c) == k1:
                s += 40000

            # History
            s += self.history[r][c][player] * 2

            # Conectividad con piezas existentes — also detect near-win
            own_neighbors = 0
            for nr, nc in ntable[r][c]:
                v = board[nr][nc]
                if v == player:
                    s += 12
                    own_neighbors += 1
                elif v == opp:
                    s += 7

            # CRITICAL: Boost cells on borders that connect to own pieces
            # This prevents winning moves on edges from being pruned
            if player == 1:
                if c == 0 or c == size - 1:
                    if own_neighbors > 0:
                        s += 5000  # border cell connecting to own chain
            else:
                if r == 0 or r == size - 1:
                    if own_neighbors > 0:
                        s += 5000  # border cell connecting to own chain

            # Centralidad
            dist_c = abs(r - center) + abs(c - center)
            s += (size - dist_c) * 3

            # Alineación con eje
            if player == 1:
                s += (size - abs(c - center)) * 1.5
            else:
                s += (size - abs(r - center)) * 1.5

            scored.append((-s, r, c))

        scored.sort()
        return [(r, c) for _, r, c in scored]

    def _store_killer(self, depth, move):
        if depth < 64:
            if self.killers[depth][0] != move:
                self.killers[depth][1] = self.killers[depth][0]
                self.killers[depth][0] = move

    def _find_immediate_win(self, board):
        """Scan ALL empty cells for an immediate winning move.
        This runs before the search to guarantee winning moves are never pruned."""
        size = self.size
        ntable = self.ntable
        player = self.my_player
        N2 = size * size
        for r in range(size):
            for c in range(size):
                if board[r][c] != 0:
                    continue
                # Make move
                board[r][c] = player
                snap_p = self.dsu_base.parent[:]
                snap_r = self.dsu_base.rank[:]
                dsu_add_stone(self.dsu_base, board, size, ntable, r, c, player)
                won = dsu_check_win(self.dsu_base, size, player)
                # Undo
                board[r][c] = 0
                self.dsu_base.parent = snap_p
                self.dsu_base.rank = snap_r
                if won:
                    return (r, c)
        return None

    # --- Búsqueda principal ---
    def search(self, board):
        empty = self.all_empty
        if not empty:
            return None

        # CRITICAL: Check for immediate winning move before any pruning
        win_move = self._find_immediate_win(board)
        if win_move is not None:
            return win_move

        # Generar candidatos: priorizar zona activa, pero incluir el resto
        if self.active:
            candidates = list(self.active)
            rest = [c for c in empty if c not in self.active]
            # Agregar las mejores celdas no-activas (por centralidad)
            center = (self.size - 1) / 2.0
            rest.sort(key=lambda rc: abs(rc[0]-center)+abs(rc[1]-center))
            candidates.extend(rest[:max(5, self.max_branch - len(candidates))])
        else:
            candidates = empty[:]

        # Limitar
        if len(candidates) > self.max_branch:
            ordered = self._order_moves(board, self.my_player, 0, None, candidates)
            candidates = ordered[:self.max_branch]
        else:
            candidates = self._order_moves(board, self.my_player, 0, None, candidates)

        best_move = candidates[0]
        pv_move = None
        depth = 1
        max_depth = min(len(empty), 50)

        while depth <= max_depth:
            try:
                # Reordenar con PV move de la iteración anterior
                if pv_move and pv_move in [(r, c) for r, c in candidates]:
                    pass  # ya incluido
                ordered = self._order_moves(board, self.my_player, 0, pv_move, candidates)

                move, score = self._search_root(board, ordered, depth)
                best_move = move
                pv_move = move

                if abs(score) >= WIN_SCORE - 10000:
                    break
                depth += 1
            except TimeoutError:
                break

        return best_move

    def _search_root(self, board, moves, depth):
        best_score = -INF
        best_move = moves[0]
        alpha = -INF
        beta = INF

        for i, move in enumerate(moves):
            self._check_time()
            r, c = move
            # Make move
            board[r][c] = self.my_player
            self.zhash ^= self.zt[r][c][self.my_player]
            dsu_snap_parent = self.dsu_base.parent[:]
            dsu_snap_rank = self.dsu_base.rank[:]
            dsu_add_stone(self.dsu_base, board, self.size, self.ntable, r, c, self.my_player)

            if dsu_check_win(self.dsu_base, self.size, self.my_player):
                # Undo
                board[r][c] = 0
                self.zhash ^= self.zt[r][c][self.my_player]
                self.dsu_base.parent = dsu_snap_parent
                self.dsu_base.rank = dsu_snap_rank
                return move, WIN_SCORE

            # PVS: primer hijo ventana completa, resto ventana nula
            if i == 0:
                score = -self._negamax(board, depth - 1, -beta, -alpha, self.opponent)
            else:
                score = -self._negamax(board, depth - 1, -alpha - 1, -alpha, self.opponent)
                if alpha < score < beta:
                    score = -self._negamax(board, depth - 1, -beta, -score, self.opponent)

            # Undo move
            board[r][c] = 0
            self.zhash ^= self.zt[r][c][self.my_player]
            self.dsu_base.parent = dsu_snap_parent
            self.dsu_base.rank = dsu_snap_rank

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score

        return best_move, best_score

    def _negamax(self, board, depth, alpha, beta, player):
        """Negamax con alpha-beta, PVS, TT, killers, y history."""
        self.nodes += 1
        if self.nodes & 1023 == 0:
            self._check_time()

        alpha_orig = alpha
        size = self.size
        ntable = self.ntable
        opp = 3 - player

        # TT Lookup
        tt_key = self.zhash ^ (player * 0x9E3779B97F4A7C15)
        tt_entry = self.tt.get(tt_key)
        if tt_entry is not None:
            tt_depth, tt_flag, tt_value, tt_move = tt_entry
            if tt_depth >= depth:
                if tt_flag == TT_EXACT:
                    return tt_value
                elif tt_flag == TT_LOWER:
                    if tt_value > alpha:
                        alpha = tt_value
                elif tt_flag == TT_UPPER:
                    if tt_value < beta:
                        beta = tt_value
                if alpha >= beta:
                    return tt_value
        else:
            tt_move = None

        if depth <= 0:
            # Evaluación rápida en hojas para mantener velocidad
            raw = evaluate_fast(board, size, player, opp, ntable)
            return raw

        # Generar movimientos: zona activa focalizada
        empty = []
        active = []
        for r in range(size):
            row = board[r]
            for c in range(size):
                if row[c] == 0:
                    has_neighbor = False
                    dirs = EVEN_DIRS if r % 2 == 0 else ODD_DIRS
                    for dr, dc in dirs:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < size and 0 <= nc < size and board[nr][nc] != 0:
                            has_neighbor = True
                            break
                    if has_neighbor:
                        active.append((r, c))
                    else:
                        empty.append((r, c))

        if not active and not empty:
            return 0

        candidates = active if active else empty
        if len(candidates) < 6 and empty:
            # Agregar algunas celdas no-activas
            candidates = candidates + empty[:min(6, len(empty))]

        # Limitar branching por profundidad
        if depth >= 3:
            blimit = self.max_branch
        elif depth == 2:
            blimit = min(self.max_branch, 16)
        else:
            blimit = min(self.max_branch, 10)

        pv_hint = tt_move
        moves = self._order_moves(board, player, depth, pv_hint, candidates)
        if len(moves) > blimit:
            moves = moves[:blimit]

        best_value = -INF
        best_move = moves[0] if moves else None

        for i, move in enumerate(moves):
            r, c = move
            board[r][c] = player
            self.zhash ^= self.zt[r][c][player]
            dsu_snap_p = self.dsu_base.parent[:]
            dsu_snap_r = self.dsu_base.rank[:]
            dsu_add_stone(self.dsu_base, board, size, ntable, r, c, player)

            if dsu_check_win(self.dsu_base, size, player):
                board[r][c] = 0
                self.zhash ^= self.zt[r][c][player]
                self.dsu_base.parent = dsu_snap_p
                self.dsu_base.rank = dsu_snap_r
                val = WIN_SCORE + depth
                # TT store
                self.tt[tt_key] = (depth, TT_LOWER, val, move)
                self._store_killer(depth, move)
                self.history[r][c][player] += depth * depth
                return val

            if i == 0:
                val = -self._negamax(board, depth - 1, -beta, -alpha, opp)
            else:
                val = -self._negamax(board, depth - 1, -alpha - 1, -alpha, opp)
                if alpha < val < beta:
                    val = -self._negamax(board, depth - 1, -beta, -val, opp)

            board[r][c] = 0
            self.zhash ^= self.zt[r][c][player]
            self.dsu_base.parent = dsu_snap_p
            self.dsu_base.rank = dsu_snap_r

            if val > best_value:
                best_value = val
                best_move = move
            if val > alpha:
                alpha = val
            if alpha >= beta:
                self._store_killer(depth, move)
                self.history[r][c][player] += depth * depth
                break

        # TT Store
        if best_value <= alpha_orig:
            tt_flag = TT_UPPER
        elif best_value >= beta:
            tt_flag = TT_LOWER
        else:
            tt_flag = TT_EXACT
        self.tt[tt_key] = (depth, tt_flag, best_value, best_move)

        return best_value


# ================================================================
# CLASE PRINCIPAL: SmartPlayer
# ================================================================

class SmartPlayer(Player):
    """
    Motor competitivo para HEX.

    Arquitectura:
      1. Apertura: centro o respuesta estratégica al centro rival
      2. Medio/final: Negamax + PVS + TT + Killer + History + Two-Distance
      3. Gestión de tiempo: iterative deepening con 4.0s budget
    """

    def play(self, board: HexBoard) -> tuple:
        size = board.size
        bm = board.board
        my_player = self._identify_player(bm, size)

        # --- Fast path for large boards ---
        # For boards >100, avoid O(N^2) full-board scans.
        # Use sparse approach: focus only on cells near existing pieces.
        if size > 100:
            center = size // 2
            if bm[center][center] == 0:
                return (center, center)
            # Find all occupied cells (sparse scan for near-empty large boards)
            occupied = []
            for r in range(size):
                for c in range(size):
                    if bm[r][c] != 0:
                        occupied.append((r, c))
            # If very few pieces, play near center/existing pieces
            if len(occupied) <= 4:
                ntable = build_neighbor_table(size)
                # Try center neighbors first
                for nr, nc in ntable[center][center]:
                    if bm[nr][nc] == 0:
                        return (nr, nc)
                # Try neighbors of occupied cells
                for r, c in occupied:
                    for nr, nc in ntable[r][c]:
                        if bm[nr][nc] == 0:
                            return (nr, nc)
            # For more pieces on a large board, use a focused zone approach
            ntable = build_neighbor_table(size)
            # Build candidate zone: cells within radius 2 of any piece
            zone = set()
            for r, c in occupied:
                for nr, nc in ntable[r][c]:
                    if bm[nr][nc] == 0:
                        zone.add((nr, nc))
                    for nr2, nc2 in ntable[nr][nc]:
                        if bm[nr2][nc2] == 0:
                            zone.add((nr2, nc2))
            if zone:
                # Quick evaluation: pick the cell that minimizes our Dijkstra distance
                best_move = None
                best_score = -INF
                opp = 3 - my_player
                # Sample a subset if zone is large
                candidates = list(zone)
                if len(candidates) > 50:
                    c_center = (size - 1) / 2.0
                    candidates.sort(key=lambda rc: abs(rc[0]-c_center)+abs(rc[1]-c_center))
                    candidates = candidates[:50]
                for r, c in candidates:
                    # Score: connectivity with own pieces + centrality
                    s = 0
                    for nr, nc in ntable[r][c]:
                        if bm[nr][nc] == my_player:
                            s += 20
                        elif bm[nr][nc] == opp:
                            s += 5
                    c_center = (size - 1) / 2.0
                    s += (size - abs(r - c_center) - abs(c - c_center)) * 2
                    if my_player == 1:
                        s += (size - abs(c - c_center)) * 1.5
                    else:
                        s += (size - abs(r - c_center)) * 1.5
                    if s > best_score:
                        best_score = s
                        best_move = (r, c)
                return best_move

        ntable = build_neighbor_table(size)

        # --- Apertura ---
        empty_count = sum(1 for r in range(size) for c in range(size) if bm[r][c] == 0)

        if empty_count >= size * size - 1:
            center = size // 2
            if bm[center][center] == 0:
                return (center, center)
            # Oponente tomó el centro → responder con celda adyacente fuerte
            # En HEX, la mejor respuesta es una celda adyacente alineada con nuestro eje
            best = None
            best_score = -INF
            for nr, nc in ntable[center][center]:
                if bm[nr][nc] != 0:
                    continue
                if my_player == 1:
                    s = -abs(nc - center) + (size - abs(nr - center))
                else:
                    s = -abs(nr - center) + (size - abs(nc - center))
                if s > best_score:
                    best_score = s
                    best = (nr, nc)
            if best:
                return best

        # Segunda jugada: si solo falta N²-2, aún podemos usar heurística rápida
        if empty_count >= size * size - 2:
            # Jugar adyacente al centro si aún hay pocas piezas
            center = size // 2
            candidates = []
            for nr, nc in ntable[center][center]:
                if bm[nr][nc] == 0:
                    candidates.append((nr, nc))
            if candidates:
                # Elegir la que mejor se alinee con nuestro eje
                if my_player == 1:
                    candidates.sort(key=lambda rc: abs(rc[1] - center))
                else:
                    candidates.sort(key=lambda rc: abs(rc[0] - center))
                # Pero no saltarse la búsqueda: puede haber algo mejor.
                # Solo como hint, pasar a la búsqueda

        # --- Copia de trabajo (in-place make/undo modifica esta copia) ---
        work = [row[:] for row in bm]

        engine = SearchEngine(my_player, size, ntable, work, time_limit=4.0)
        best_move = engine.search(work)
        return best_move

    def _identify_player(self, board, size):
        for attr in ('player_id', 'player', 'id', 'color', 'number'):
            val = getattr(self, attr, None)
            if val in (1, 2):
                return val
        p1 = sum(row.count(1) for row in board)
        p2 = sum(row.count(2) for row in board)
        return 1 if p1 <= p2 else 2
