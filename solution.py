"""
solution.py – Competitive HEX Game Engine

High-performance AI engine for the board game HEX, implementing modern
game-tree search techniques for optimal move selection.

Techniques:
  * Negamax with Alpha-Beta Pruning
  * Principal Variation Search (PVS) with Late Move Reductions (LMR)
  * Iterative Deepening with Aspiration Windows
  * Zobrist Hashing + Bounded Transposition Table (depth-preferred)
  * Killer Moves (2 slots per depth)
  * History Heuristic (cumulative cutoff scores per cell)
  * Two-Distance Heuristic (bidirectional Dijkstra) + virtual bridge detection
  * In-place Make/Undo move (zero-copy)
  * Incremental Union-Find (DSU) for O(α(N)) win detection
  * Focused move generation (active zone + neighborhood)
  * Opening book with center-response strategy
"""

import heapq
import random
import time
from typing import Dict, List, Optional, Set, Tuple

from board import HexBoard
from player import Player


# =====================================================================
#  CONSTANTS
# =====================================================================

# -- Scoring --
WIN_SCORE: int = 1_000_000
HALF_WIN: int = WIN_SCORE // 2
INF: float = float("inf")

# -- Transposition Table --
TT_EXACT: int = 0
TT_LOWER: int = 1   # alpha cutoff  → lower bound
TT_UPPER: int = 2   # beta  cutoff  → upper bound
TT_PLAYER_MIX: int = 0x9E3779B97F4A7C15   # golden-ratio constant
TT_MAX_ENTRIES: int = 500_000              # memory safety cap (~50 MB)

# -- Zobrist Hashing --
ZOBRIST_SEED: int = 42   # fixed seed → deterministic hashes
ZOBRIST_BITS: int = 64

# -- Even-R Hex Offset Neighbor Directions --
EVEN_DIRS: Tuple[Tuple[int, int], ...] = (
    (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1),
)
ODD_DIRS: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0),
)

# -- Move Ordering Priorities --
PV_MOVE_PRIORITY: int = 10_000_000
KILLER_PRIMARY_BONUS: int = 50_000
KILLER_SECONDARY_BONUS: int = 40_000
BORDER_CONNECT_BONUS: int = 5_000
BRIDGE_CONNECT_BONUS: int = 20    # virtual bridge with own stone
OWN_NEIGHBOR_BONUS: int = 12
OPP_NEIGHBOR_BONUS: int = 7
CENTRALITY_WEIGHT: float = 3.0
AXIS_ALIGNMENT_WEIGHT: float = 1.5
HISTORY_MULTIPLIER: int = 2

# -- Evaluation Weights --
DISTANCE_WEIGHT: int = 200
PATH_COUNT_WEIGHT: int = 10
PATH_COUNT_MAX_SIZE: int = 11   # path counting only for boards ≤ this size

# -- Search Parameters --
TIME_BUDGET: float = 4.0
TIME_CHECK_MASK: int = 1023     # check clock every (N & mask == 0)
MAX_SEARCH_DEPTH: int = 50
MAX_KILLER_DEPTH: int = 64
WIN_DECISIVE_MARGIN: int = 10_000

# -- Late Move Reductions (LMR) --
LMR_DEPTH_THRESHOLD: int = 4   # only reduce at depth ≥ this
LMR_MOVE_THRESHOLD: int = 4    # only reduce move index ≥ this
LMR_REDUCTION: int = 1         # reduce by this many plies

# -- Aspiration Windows --
ASPIRATION_WINDOW: int = 300    # initial half-window around previous score
ASPIRATION_MIN_DEPTH: int = 3   # first depth where aspiration is used

# -- Board-Size Thresholds --
LAZY_NEIGHBOR_THRESHOLD: int = 100   # lazy neighbor table above this size
LARGE_BOARD_THRESHOLD: int = 100     # sparse fast-path above this size
BRIDGE_MAX_SIZE: int = 15            # precompute bridges only up to this

# -- Large-Board Fast-Path --
LARGE_BOARD_FEW_PIECES: int = 4
LARGE_BOARD_MAX_CANDIDATES: int = 50
LARGE_BOARD_OWN_WEIGHT: int = 20
LARGE_BOARD_OPP_WEIGHT: int = 5
LARGE_BOARD_CENTRALITY_WEIGHT: float = 2.0

# -- Player Identification Attributes --
_PLAYER_ATTRIBUTES: Tuple[str, ...] = (
    "player_id", "player", "id", "color", "number",
)


# =====================================================================
#  NEIGHBOR TABLE (cached per board size)
# =====================================================================

_NEIGHBOR_TABLE_CACHE: Dict[int, object] = {}


def _compute_neighbors(
    row: int, col: int, size: int,
) -> Tuple[Tuple[int, int], ...]:
    """Compute hex neighbors for a single cell using even-r offset coordinates."""
    dirs = EVEN_DIRS if row % 2 == 0 else ODD_DIRS
    neighbors: List[Tuple[int, int]] = []
    for dr, dc in dirs:
        nr, nc = row + dr, col + dc
        if 0 <= nr < size and 0 <= nc < size:
            neighbors.append((nr, nc))
    return tuple(neighbors)


class LazyNeighborTable:
    """On-demand neighbor table for large boards.

    Avoids O(N²) upfront precomputation by computing and caching
    neighbors lazily on first access.
    """

    __slots__ = ("size", "_cache")

    def __init__(self, size: int) -> None:
        self.size: int = size
        self._cache: Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]] = {}

    def __getitem__(self, row: int) -> "_LazyRow":
        return _LazyRow(self, row)


class _LazyRow:
    """Row proxy for ``LazyNeighborTable``."""

    __slots__ = ("_table", "_row")

    def __init__(self, table: LazyNeighborTable, row: int) -> None:
        self._table = table
        self._row = row

    def __getitem__(self, col: int) -> Tuple[Tuple[int, int], ...]:
        key = (self._row, col)
        cache = self._table._cache
        result = cache.get(key)
        if result is None:
            result = _compute_neighbors(self._row, col, self._table.size)
            cache[key] = result
        return result


def build_neighbor_table(size: int):
    """Build or retrieve the cached neighbor lookup table for *size*.

    Returns ``LazyNeighborTable`` for boards > ``LAZY_NEIGHBOR_THRESHOLD``,
    or a fully precomputed 2-D list otherwise.
    """
    if size in _NEIGHBOR_TABLE_CACHE:
        return _NEIGHBOR_TABLE_CACHE[size]

    if size > LAZY_NEIGHBOR_THRESHOLD:
        table = LazyNeighborTable(size)
        _NEIGHBOR_TABLE_CACHE[size] = table
        return table

    table: List[List[Optional[Tuple]]] = [[None] * size for _ in range(size)]
    for r in range(size):
        dirs = EVEN_DIRS if r % 2 == 0 else ODD_DIRS
        for c in range(size):
            neighbors: List[Tuple[int, int]] = []
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    neighbors.append((nr, nc))
            table[r][c] = tuple(neighbors)
    _NEIGHBOR_TABLE_CACHE[size] = table
    return table


# =====================================================================
#  VIRTUAL BRIDGE TABLE (Even-R hex)
# =====================================================================
# A "bridge" in HEX: two cells of the same player separated by exactly
# two shared empty neighbor cells (carriers).  If the opponent takes one
# carrier, the player fills the other → the virtual connection holds.
# Each entry: (dest_r, dest_c, carrier1_r, carrier1_c, carrier2_r, carrier2_c).

_BRIDGE_CACHE: Dict[int, List[List[List]]] = {}


def build_bridge_table(size: int, ntable) -> List[List[List]]:
    """Precompute virtual bridge patterns for every cell on the board."""
    if size in _BRIDGE_CACHE:
        return _BRIDGE_CACHE[size]

    bridges: List[List[List]] = [[[] for _ in range(size)] for _ in range(size)]
    for r in range(size):
        for c in range(size):
            own_neighbors = set(ntable[r][c])
            for nr, nc in ntable[r][c]:
                for nr2, nc2 in ntable[nr][nc]:
                    if (nr2, nc2) == (r, c):
                        continue
                    if (nr2, nc2) in own_neighbors:
                        continue
                    common = set(ntable[r][c]) & set(ntable[nr2][nc2])
                    common.discard((nr2, nc2))
                    common.discard((r, c))
                    if (nr, nc) in common:
                        for other in common:
                            if other != (nr, nc):
                                bridges[r][c].append(
                                    (nr2, nc2, nr, nc, other[0], other[1])
                                )
                                break
    _BRIDGE_CACHE[size] = bridges
    return bridges


# =====================================================================
#  ZOBRIST HASHING
# =====================================================================

_ZOBRIST_CACHE: Dict[int, List[List[List[int]]]] = {}


def get_zobrist_table(size: int) -> List[List[List[int]]]:
    """Get or create a deterministic Zobrist hash table for *size*."""
    if size in _ZOBRIST_CACHE:
        return _ZOBRIST_CACHE[size]

    rng = random.Random(ZOBRIST_SEED)
    zt: List[List[List[int]]] = [
        [[rng.getrandbits(ZOBRIST_BITS) for _ in range(3)] for _ in range(size)]
        for _ in range(size)
    ]
    _ZOBRIST_CACHE[size] = zt
    return zt


def compute_zobrist(
    board: List[List[int]], size: int, zt: List[List[List[int]]],
) -> int:
    """Compute the full Zobrist hash for the current board state."""
    h: int = 0
    for r in range(size):
        row = board[r]
        for c in range(size):
            cell = row[c]
            if cell:
                h ^= zt[r][c][cell]
    return h


# =====================================================================
#  UNION-FIND (DSU) — O(α(N)) Win Detection
# =====================================================================
# Virtual border nodes for N×N board:
#   LEFT=N², RIGHT=N²+1, TOP=N²+2, BOTTOM=N²+3


class DSU:
    """Disjoint-Set Union with path halving and union by rank.

    Four virtual border nodes allow win detection via a single
    ``connected()`` query (e.g. LEFT↔RIGHT for player 1).
    """

    __slots__ = ("parent", "rank", "n", "size")

    def __init__(self, size: int) -> None:
        self.size: int = size
        self.n: int = size * size + 4
        self.parent: List[int] = list(range(self.n))
        self.rank: List[int] = [0] * self.n

    def find(self, x: int) -> int:
        """Find representative with path halving."""
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        """Union by rank."""
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1

    def connected(self, a: int, b: int) -> bool:
        """Check whether two nodes share a component."""
        return self.find(a) == self.find(b)

    def copy(self) -> "DSU":
        """Snapshot the DSU state (shallow copy of internal arrays)."""
        d = DSU.__new__(DSU)
        d.size = self.size
        d.n = self.n
        d.parent = self.parent[:]
        d.rank = self.rank[:]
        return d


def build_dsu_from_board(
    board: List[List[int]], size: int, ntable,
) -> DSU:
    """Construct a DSU reflecting all stones currently on the board."""
    n2 = size * size
    left, right, top, bottom = n2, n2 + 1, n2 + 2, n2 + 3
    dsu = DSU(size)

    for r in range(size):
        for c in range(size):
            player = board[r][c]
            if player == 0:
                continue
            idx = r * size + c

            if player == 1:
                if c == 0:
                    dsu.union(idx, left)
                if c == size - 1:
                    dsu.union(idx, right)
            else:
                if r == 0:
                    dsu.union(idx, top)
                if r == size - 1:
                    dsu.union(idx, bottom)

            # Inline direction lookup avoids ntable dependency during build
            dirs = EVEN_DIRS if r % 2 == 0 else ODD_DIRS
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and board[nr][nc] == player:
                    dsu.union(idx, nr * size + nc)

    return dsu


def dsu_add_stone(
    dsu: DSU,
    board: List[List[int]],
    size: int,
    ntable,
    row: int,
    col: int,
    player: int,
) -> None:
    """Incrementally add a stone to the DSU and connect to borders/neighbors."""
    n2 = size * size
    idx = row * size + col

    if player == 1:
        if col == 0:
            dsu.union(idx, n2)          # LEFT
        if col == size - 1:
            dsu.union(idx, n2 + 1)      # RIGHT
    else:
        if row == 0:
            dsu.union(idx, n2 + 2)      # TOP
        if row == size - 1:
            dsu.union(idx, n2 + 3)      # BOTTOM

    for nr, nc in ntable[row][col]:
        if board[nr][nc] == player:
            dsu.union(idx, nr * size + nc)


def dsu_check_win(dsu: DSU, size: int, player: int) -> bool:
    """Check if *player* has connected their two borders."""
    n2 = size * size
    if player == 1:
        return dsu.connected(n2, n2 + 1)       # LEFT ↔ RIGHT
    return dsu.connected(n2 + 2, n2 + 3)       # TOP  ↔ BOTTOM


# =====================================================================
#  PUBLIC COMPATIBILITY FUNCTIONS (used by external tests)
# =====================================================================


def get_neighbors(
    row: int, col: int, size: int,
) -> List[Tuple[int, int]]:
    """Return valid hex neighbors for (*row*, *col*) on a board of *size*."""
    dirs = EVEN_DIRS if row % 2 == 0 else ODD_DIRS
    return [
        (row + dr, col + dc)
        for dr, dc in dirs
        if 0 <= row + dr < size and 0 <= col + dc < size
    ]


def get_empty_cells(
    board: List[List[int]], size: int,
) -> List[Tuple[int, int]]:
    """Return all empty cells on the board."""
    return [
        (r, c)
        for r in range(size)
        for c in range(size)
        if board[r][c] == 0
    ]


def apply_move(
    board: List[List[int]], move: Tuple[int, int], player: int,
) -> List[List[int]]:
    """Return a new board with *player*'s stone placed at *move*."""
    new_board = [row[:] for row in board]
    new_board[move[0]][move[1]] = player
    return new_board


def check_connection(
    board: List[List[int]],
    size: int,
    player: int,
    ntable,
) -> bool:
    """BFS-based connectivity check — kept for test compatibility."""
    from collections import deque

    if player == 1:
        starts = [(r, 0) for r in range(size) if board[r][0] == player]
        if not starts:
            return False
        if not any(board[r][size - 1] == player for r in range(size)):
            return False
        visited: Set[Tuple[int, int]] = set(starts)
        queue = deque(starts)
        while queue:
            r, c = queue.popleft()
            if c == size - 1:
                return True
            for nr, nc in ntable[r][c]:
                if (nr, nc) not in visited and board[nr][nc] == player:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    else:
        starts = [(0, c) for c in range(size) if board[0][c] == player]
        if not starts:
            return False
        if not any(board[size - 1][c] == player for c in range(size)):
            return False
        visited = set(starts)
        queue = deque(starts)
        while queue:
            r, c = queue.popleft()
            if r == size - 1:
                return True
            for nr, nc in ntable[r][c]:
                if (nr, nc) not in visited and board[nr][nc] == player:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return False


# =====================================================================
#  EVALUATION HEURISTICS
# =====================================================================


def dijkstra_distance(
    board: List[List[int]], size: int, player: int, ntable,
) -> float:
    """Shortest-path distance to connect *player*'s two borders.

    Cell costs: own stone = 0, empty = 1, opponent = ∞ (impassable).
    Returns ``INF`` if no path exists.
    """
    opp = 3 - player
    dist = [[INF] * size for _ in range(size)]
    heap: List[Tuple[float, int, int]] = []

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


def dijkstra_full(
    board: List[List[int]], size: int, player: int, ntable,
) -> List[List[float]]:
    """Full Dijkstra from *player*'s start border.

    Returns the complete distance matrix needed for two-distance
    evaluation and path-count tie-breaking.
    """
    opp = 3 - player
    dist = [[INF] * size for _ in range(size)]
    heap: List[Tuple[float, int, int]] = []

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


def dijkstra_reverse(
    board: List[List[int]], size: int, player: int, ntable,
) -> List[List[float]]:
    """Dijkstra from *player*'s goal border (reverse direction)."""
    opp = 3 - player
    dist = [[INF] * size for _ in range(size)]
    heap: List[Tuple[float, int, int]] = []

    if player == 1:
        for r in range(size):
            v = board[r][size - 1]
            if v == opp:
                continue
            cost = 0 if v == player else 1
            if cost < dist[r][size - 1]:
                dist[r][size - 1] = cost
                heapq.heappush(heap, (cost, r, size - 1))
    else:
        for c in range(size):
            v = board[size - 1][c]
            if v == opp:
                continue
            cost = 0 if v == player else 1
            if cost < dist[size - 1][c]:
                dist[size - 1][c] = cost
                heapq.heappush(heap, (cost, size - 1, c))

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


def two_distance(
    board: List[List[int]], size: int, player: int, ntable,
) -> Tuple[float, int]:
    """Two-distance heuristic: bidirectional Dijkstra + on-path cell count.

    Computes ``dist_fwd + dist_rev`` for every non-opponent cell.
    The minimum is the connection distance; the count of cells achieving
    that minimum measures path redundancy (more paths → harder to block).

    Returns:
        (shortest_distance, on_path_cell_count)
    """
    dfwd = dijkstra_full(board, size, player, ntable)
    drev = dijkstra_reverse(board, size, player, ntable)

    if player == 1:
        min_dist = min(dfwd[r][size - 1] for r in range(size))
    else:
        min_dist = min(dfwd[size - 1][c] for c in range(size))

    if min_dist >= INF:
        return INF, 0

    opp = 3 - player
    on_path = 0
    for r in range(size):
        for c in range(size):
            if board[r][c] == opp:
                continue
            if dfwd[r][c] + drev[r][c] == min_dist:
                on_path += 1

    return min_dist, on_path


def evaluate_fast(
    board: List[List[int]],
    size: int,
    my_player: int,
    opponent: int,
    ntable,
) -> int:
    """Fast Dijkstra-based evaluation for deep tree nodes.

    For small boards (≤ ``PATH_COUNT_MAX_SIZE``), augments the distance
    difference with path-count tie-breaking to distinguish blocking moves.
    """
    my_dist = dijkstra_distance(board, size, my_player, ntable)
    opp_dist = dijkstra_distance(board, size, opponent, ntable)

    if my_dist >= INF and opp_dist >= INF:
        return 0
    if opp_dist >= INF:
        return HALF_WIN
    if my_dist >= INF:
        return -HALF_WIN

    score = (opp_dist - my_dist) * DISTANCE_WEIGHT

    # Path-count tie-breaking uses two_distance (bidirectional Dijkstra)
    # to count on-path cells — more paths = stronger position
    if size <= PATH_COUNT_MAX_SIZE and my_dist == opp_dist:
        _, my_paths = two_distance(board, size, my_player, ntable)
        _, opp_paths = two_distance(board, size, opponent, ntable)
        score += (my_paths - opp_paths) * PATH_COUNT_WEIGHT

    return score


def evaluate_board(
    board: List[List[int]],
    size: int,
    my_player: int,
    opponent: int,
    ntable,
) -> int:
    """Full two-distance evaluation for shallow tree nodes.

    Combines connection distance difference with on-path redundancy.
    """
    my_dist, my_paths = two_distance(board, size, my_player, ntable)
    opp_dist, opp_paths = two_distance(board, size, opponent, ntable)

    if my_dist >= INF and opp_dist >= INF:
        return 0
    if opp_dist >= INF:
        return HALF_WIN
    if my_dist >= INF:
        return -HALF_WIN

    score = (opp_dist - my_dist) * DISTANCE_WEIGHT
    score += (my_paths - opp_paths) * PATH_COUNT_WEIGHT

    return score


# =====================================================================
#  SEARCH ENGINE: Negamax + PVS + LMR + TT + Killer + History
# =====================================================================


class SearchEngine:
    """Iterative-deepening Negamax search with alpha-beta pruning.

    Features:
      - Principal Variation Search (PVS)
      - Late Move Reductions (LMR) for low-priority late moves
      - Aspiration Windows for tighter search bounds
      - Bounded Transposition Table with depth-preferred replacement
      - Killer move heuristic (2 slots per depth)
      - History heuristic for move ordering
      - Virtual bridge patterns in move ordering
      - Adaptive branching limits by board size and depth
      - Immediate win detection before search
    """

    def __init__(
        self,
        my_player: int,
        size: int,
        ntable,
        board: List[List[int]],
        time_limit: float,
    ) -> None:
        self.my_player: int = my_player
        self.opponent: int = 3 - my_player
        self.size: int = size
        self.ntable = ntable
        self.time_limit: float = time_limit
        self.start_time: float = time.time()
        self.nodes: int = 0
        self.timed_out: bool = False

        # Zobrist hashing
        self.zt = get_zobrist_table(size)
        self.zhash: int = compute_zobrist(board, size, self.zt)

        # Bounded transposition table
        self.tt: Dict[int, Tuple[int, int, float, Optional[Tuple[int, int]]]] = {}

        # Killer moves: 2 slots per depth
        self.killers: List[List[Optional[Tuple[int, int]]]] = [
            [None, None] for _ in range(MAX_KILLER_DEPTH)
        ]

        # History heuristic: cumulative cutoff score per cell per player
        self.history: List[List[List[int]]] = [
            [[0, 0, 0] for _ in range(size)] for _ in range(size)
        ]

        # DSU built from current board state
        self.dsu_base: DSU = build_dsu_from_board(board, size, ntable)

        # Bridge table for move ordering (only worthwhile for small boards)
        if size <= BRIDGE_MAX_SIZE:
            self.bridge_table = build_bridge_table(size, ntable)
        else:
            self.bridge_table = None

        # Adaptive branching limit by board size
        if size <= 5:
            self.max_branch: int = 35
        elif size <= 7:
            self.max_branch = 28
        elif size <= 11:
            self.max_branch = 22
        else:
            self.max_branch = 16

        # Precompute hot zone (empty cells adjacent to existing stones)
        self._precompute_active_zone(board)

    # -----------------------------------------------------------------
    #  Active Zone
    # -----------------------------------------------------------------

    def _precompute_active_zone(self, board: List[List[int]]) -> None:
        """Identify empty cells adjacent to any piece (hot zone).

        Moves in the active zone are prioritised because they interact
        with the existing stone structure.
        """
        size = self.size
        self.active: Set[Tuple[int, int]] = set()
        self.all_empty: List[Tuple[int, int]] = []

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

    # -----------------------------------------------------------------
    #  Time Management
    # -----------------------------------------------------------------

    def _check_time(self) -> None:
        """Raise ``TimeoutError`` if the time budget has been exhausted."""
        if self.timed_out:
            raise TimeoutError
        if time.time() - self.start_time >= self.time_limit:
            self.timed_out = True
            raise TimeoutError

    # -----------------------------------------------------------------
    #  Transposition Table (bounded, depth-preferred)
    # -----------------------------------------------------------------

    def _tt_store(
        self,
        key: int,
        depth: int,
        flag: int,
        value: float,
        move: Optional[Tuple[int, int]],
    ) -> None:
        """Store entry with depth-preferred replacement policy.

        Existing deeper entries are never overwritten by shallower ones.
        When the table exceeds ``TT_MAX_ENTRIES``, all entries are cleared
        (generation-based eviction) to guarantee bounded memory.
        """
        tt = self.tt
        existing = tt.get(key)
        if existing is not None:
            # Depth-preferred: keep the deeper analysis
            if depth < existing[0]:
                return
        elif len(tt) >= TT_MAX_ENTRIES:
            # Safety cap reached — start a fresh generation
            tt.clear()
        tt[key] = (depth, flag, value, move)

    # -----------------------------------------------------------------
    #  Move Ordering
    # -----------------------------------------------------------------

    def _order_moves(
        self,
        board: List[List[int]],
        player: int,
        depth: int,
        pv_move: Optional[Tuple[int, int]],
        empty_cells: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Score and sort candidate moves for maximum alpha-beta pruning.

        Priority tiers:
          0. PV move (best from previous iteration)
          1. Killer moves (caused cutoffs at sibling nodes)
          2. Bridge connections, history, connectivity, centrality, axis
        """
        ntable = self.ntable
        size = self.size
        opp = 3 - player
        center = (size - 1) / 2.0
        bridge_table = self.bridge_table

        scored: List[Tuple[float, int, int]] = []
        k0, k1 = (
            self.killers[depth] if depth < MAX_KILLER_DEPTH else (None, None)
        )

        for r, c in empty_cells:
            # PV move gets absolute priority
            if (r, c) == pv_move:
                scored.append((-PV_MOVE_PRIORITY, r, c))
                continue

            score = 0.0

            # Killer bonus
            if (r, c) == k0:
                score += KILLER_PRIMARY_BONUS
            elif (r, c) == k1:
                score += KILLER_SECONDARY_BONUS

            # History heuristic
            score += self.history[r][c][player] * HISTORY_MULTIPLIER

            # Connectivity with existing pieces
            own_neighbors = 0
            for nr, nc in ntable[r][c]:
                v = board[nr][nc]
                if v == player:
                    score += OWN_NEIGHBOR_BONUS
                    own_neighbors += 1
                elif v == opp:
                    score += OPP_NEIGHBOR_BONUS

            # Border-connecting bonus — prevents winning edge
            # moves from being pruned due to low centrality
            if player == 1:
                if (c == 0 or c == size - 1) and own_neighbors > 0:
                    score += BORDER_CONNECT_BONUS
            else:
                if (r == 0 or r == size - 1) and own_neighbors > 0:
                    score += BORDER_CONNECT_BONUS

            # Virtual bridge bonus — detects virtual connections
            # through carrier cells, improving connectivity awareness
            if bridge_table is not None:
                for dest_r, dest_c, c1r, c1c, c2r, c2c in bridge_table[r][c]:
                    if board[dest_r][dest_c] == player:
                        # Both carriers must be empty for bridge to hold
                        if board[c1r][c1c] == 0 and board[c2r][c2c] == 0:
                            score += BRIDGE_CONNECT_BONUS

            # Centrality
            dist_center = abs(r - center) + abs(c - center)
            score += (size - dist_center) * CENTRALITY_WEIGHT

            # Axis alignment — prefer cells along our connection axis
            if player == 1:
                score += (size - abs(c - center)) * AXIS_ALIGNMENT_WEIGHT
            else:
                score += (size - abs(r - center)) * AXIS_ALIGNMENT_WEIGHT

            scored.append((-score, r, c))

        scored.sort()
        return [(r, c) for _, r, c in scored]

    def _store_killer(self, depth: int, move: Tuple[int, int]) -> None:
        """Record a killer move at the given depth."""
        if depth < MAX_KILLER_DEPTH:
            if self.killers[depth][0] != move:
                self.killers[depth][1] = self.killers[depth][0]
                self.killers[depth][0] = move

    # -----------------------------------------------------------------
    #  Immediate Win Detection
    # -----------------------------------------------------------------

    def _find_immediate_win(
        self, board: List[List[int]],
    ) -> Optional[Tuple[int, int]]:
        """Scan all empty cells for a single-move win.

        Runs before the full search to guarantee winning moves are never
        missed due to pruning or move-ordering limitations.
        """
        size = self.size
        ntable = self.ntable
        player = self.my_player

        for r in range(size):
            for c in range(size):
                if board[r][c] != 0:
                    continue
                # Make move
                board[r][c] = player
                snap_parent = self.dsu_base.parent[:]
                snap_rank = self.dsu_base.rank[:]
                dsu_add_stone(self.dsu_base, board, size, ntable, r, c, player)
                won = dsu_check_win(self.dsu_base, size, player)
                # Undo move
                board[r][c] = 0
                self.dsu_base.parent = snap_parent
                self.dsu_base.rank = snap_rank
                if won:
                    return (r, c)

        return None

    # -----------------------------------------------------------------
    #  Iterative Deepening with Aspiration Windows
    # -----------------------------------------------------------------

    def search(self, board: List[List[int]]) -> Optional[Tuple[int, int]]:
        """Run iterative-deepening search and return the best move found.

        Uses aspiration windows starting at ``ASPIRATION_MIN_DEPTH`` to
        search with a narrow alpha-beta window around the previous
        iteration's score, falling back to full window on failure.
        """
        empty = self.all_empty
        if not empty:
            return None

        # Immediate win detection — bypasses the entire search tree
        win_move = self._find_immediate_win(board)
        if win_move is not None:
            return win_move

        # Build candidate set: prioritise active zone, pad with central cells
        if self.active:
            candidates = list(self.active)
            rest = [cell for cell in empty if cell not in self.active]
            center = (self.size - 1) / 2.0
            rest.sort(key=lambda rc: abs(rc[0] - center) + abs(rc[1] - center))
            candidates.extend(rest[:max(5, self.max_branch - len(candidates))])
        else:
            candidates = empty[:]

        # Trim to branching limit
        if len(candidates) > self.max_branch:
            ordered = self._order_moves(
                board, self.my_player, 0, None, candidates,
            )
            candidates = ordered[: self.max_branch]
        else:
            candidates = self._order_moves(
                board, self.my_player, 0, None, candidates,
            )

        best_move = candidates[0]
        pv_move: Optional[Tuple[int, int]] = None
        depth = 1
        max_depth = min(len(empty), MAX_SEARCH_DEPTH)
        last_score: Optional[float] = None

        while depth <= max_depth:
            try:
                ordered = self._order_moves(
                    board, self.my_player, 0, pv_move, candidates,
                )

                # Aspiration windows: search with narrow bounds around
                # the previous iteration's score to improve cutoff rates
                if last_score is not None and depth >= ASPIRATION_MIN_DEPTH:
                    asp_alpha = last_score - ASPIRATION_WINDOW
                    asp_beta = last_score + ASPIRATION_WINDOW
                    move, score = self._search_root(
                        board, ordered, depth, asp_alpha, asp_beta,
                    )
                    # Fall back to full window on aspiration failure
                    if score <= asp_alpha or score >= asp_beta:
                        move, score = self._search_root(
                            board, ordered, depth,
                        )
                else:
                    move, score = self._search_root(board, ordered, depth)

                best_move = move
                pv_move = move
                last_score = score

                # Early exit on decisive score
                if abs(score) >= WIN_SCORE - WIN_DECISIVE_MARGIN:
                    break
                depth += 1
            except TimeoutError:
                break

        return best_move

    # -----------------------------------------------------------------
    #  Root-Level Search
    # -----------------------------------------------------------------

    def _search_root(
        self,
        board: List[List[int]],
        moves: List[Tuple[int, int]],
        depth: int,
        alpha: float = -INF,
        beta: float = INF,
    ) -> Tuple[Tuple[int, int], float]:
        """Search from root with PVS.  Accepts custom alpha/beta for aspiration."""
        best_score = -INF
        best_move = moves[0]

        for i, move in enumerate(moves):
            self._check_time()
            r, c = move

            # ---- Make move ----
            board[r][c] = self.my_player
            self.zhash ^= self.zt[r][c][self.my_player]
            dsu_snap_parent = self.dsu_base.parent[:]
            dsu_snap_rank = self.dsu_base.rank[:]
            dsu_add_stone(
                self.dsu_base, board, self.size, self.ntable,
                r, c, self.my_player,
            )

            if dsu_check_win(self.dsu_base, self.size, self.my_player):
                board[r][c] = 0
                self.zhash ^= self.zt[r][c][self.my_player]
                self.dsu_base.parent = dsu_snap_parent
                self.dsu_base.rank = dsu_snap_rank
                return move, WIN_SCORE

            # PVS: full window for first child, null window for rest
            if i == 0:
                score = -self._negamax(
                    board, depth - 1, -beta, -alpha, self.opponent,
                )
            else:
                score = -self._negamax(
                    board, depth - 1, -alpha - 1, -alpha, self.opponent,
                )
                if alpha < score < beta:
                    score = -self._negamax(
                        board, depth - 1, -beta, -score, self.opponent,
                    )

            # ---- Undo move ----
            board[r][c] = 0
            self.zhash ^= self.zt[r][c][self.my_player]
            self.dsu_base.parent = dsu_snap_parent
            self.dsu_base.rank = dsu_snap_rank

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            # Beta cutoff — needed for aspiration windows
            if alpha >= beta:
                break

        return best_move, best_score

    # -----------------------------------------------------------------
    #  Negamax with Alpha-Beta + PVS + LMR + TT + Killers + History
    # -----------------------------------------------------------------

    def _negamax(
        self,
        board: List[List[int]],
        depth: int,
        alpha: float,
        beta: float,
        player: int,
    ) -> float:
        """Recursive negamax search with all pruning enhancements.

        Late Move Reductions (LMR) reduce search depth for low-priority
        moves that appear late in the move ordering, re-searching at
        full depth only if the reduced search raises alpha.
        """
        self.nodes += 1
        if self.nodes & TIME_CHECK_MASK == 0:
            self._check_time()

        alpha_orig = alpha
        size = self.size
        ntable = self.ntable
        opp = 3 - player

        # ---- Transposition Table Lookup ----
        tt_key = self.zhash ^ (player * TT_PLAYER_MIX)
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

        # ---- Leaf Evaluation ----
        if depth <= 0:
            return evaluate_fast(board, size, player, opp, ntable)

        # ---- Move Generation: active zone focus ----
        empty: List[Tuple[int, int]] = []
        active: List[Tuple[int, int]] = []
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
            candidates = candidates + empty[: min(6, len(empty))]

        # Depth-adaptive branching limit
        if depth >= 3:
            branch_limit = self.max_branch
        elif depth == 2:
            branch_limit = min(self.max_branch, 16)
        else:
            branch_limit = min(self.max_branch, 10)

        pv_hint = tt_move
        moves = self._order_moves(board, player, depth, pv_hint, candidates)
        if len(moves) > branch_limit:
            moves = moves[:branch_limit]

        best_value = -INF
        best_move = moves[0] if moves else None

        # Cache killers for LMR eligibility check
        k0, k1 = (
            self.killers[depth] if depth < MAX_KILLER_DEPTH else (None, None)
        )

        for i, move in enumerate(moves):
            r, c = move

            # ---- Make move ----
            board[r][c] = player
            self.zhash ^= self.zt[r][c][player]
            dsu_snap_parent = self.dsu_base.parent[:]
            dsu_snap_rank = self.dsu_base.rank[:]
            dsu_add_stone(self.dsu_base, board, size, ntable, r, c, player)

            # Check for win
            if dsu_check_win(self.dsu_base, size, player):
                board[r][c] = 0
                self.zhash ^= self.zt[r][c][player]
                self.dsu_base.parent = dsu_snap_parent
                self.dsu_base.rank = dsu_snap_rank
                val = WIN_SCORE + depth
                self._tt_store(tt_key, depth, TT_LOWER, val, move)
                self._store_killer(depth, move)
                self.history[r][c][player] += depth * depth
                return val

            # ---- PVS + LMR ----
            if i == 0:
                # First move: full window, no reduction
                val = -self._negamax(board, depth - 1, -beta, -alpha, opp)
            else:
                # Determine Late Move Reduction
                # LMR is safe for "quiet" late moves: not PV hint,
                # not a killer, and sufficiently deep in the tree
                reduction = 0
                if (depth >= LMR_DEPTH_THRESHOLD
                        and i >= LMR_MOVE_THRESHOLD
                        and move != pv_hint
                        and move != k0
                        and move != k1):
                    reduction = LMR_REDUCTION

                # Null-window search (possibly reduced)
                val = -self._negamax(
                    board, depth - 1 - reduction, -alpha - 1, -alpha, opp,
                )

                # Re-search at full depth if reduced search raised alpha
                if reduction > 0 and val > alpha:
                    val = -self._negamax(
                        board, depth - 1, -alpha - 1, -alpha, opp,
                    )

                # Re-search with full window if null window raised alpha
                if alpha < val < beta:
                    val = -self._negamax(
                        board, depth - 1, -beta, -val, opp,
                    )

            # ---- Undo move ----
            board[r][c] = 0
            self.zhash ^= self.zt[r][c][player]
            self.dsu_base.parent = dsu_snap_parent
            self.dsu_base.rank = dsu_snap_rank

            if val > best_value:
                best_value = val
                best_move = move
            if val > alpha:
                alpha = val
            if alpha >= beta:
                self._store_killer(depth, move)
                self.history[r][c][player] += depth * depth
                break

        # ---- Transposition Table Store (bounded) ----
        if best_value <= alpha_orig:
            tt_flag = TT_UPPER
        elif best_value >= beta:
            tt_flag = TT_LOWER
        else:
            tt_flag = TT_EXACT
        self._tt_store(tt_key, depth, tt_flag, best_value, best_move)

        return best_value


# =====================================================================
#  MAIN PLAYER CLASS
# =====================================================================


class SmartPlayer(Player):
    """Competitive HEX engine exposed as a ``Player`` subclass.

    Architecture:
      1. **Opening**: center or strategic center-response
      2. **Mid/endgame**: Negamax + PVS + LMR + TT + Killer + History
      3. **Time management**: iterative deepening with aspiration windows

    For large boards (> ``LARGE_BOARD_THRESHOLD``), a lightweight sparse
    fast-path avoids full-board scans and delivers moves in sub-second time.
    """

    def play(self, board: HexBoard) -> Tuple[int, int]:
        """Select and return the best move for the current position."""
        size: int = board.size
        board_matrix: List[List[int]] = board.board
        my_player: int = self._identify_player(board_matrix, size)

        # ---- Large-board fast-path ----
        if size > LARGE_BOARD_THRESHOLD:
            return self._play_large_board(board_matrix, size, my_player)

        ntable = build_neighbor_table(size)

        # ---- Opening book ----
        empty_count = sum(
            1 for r in range(size) for c in range(size)
            if board_matrix[r][c] == 0
        )

        if empty_count >= size * size - 1:
            move = self._opening_move(board_matrix, size, my_player, ntable)
            if move is not None:
                return move

        # Second-move hint: sort center-adjacent cells (search will refine)
        if empty_count >= size * size - 2:
            center = size // 2
            candidates = [
                (nr, nc)
                for nr, nc in ntable[center][center]
                if board_matrix[nr][nc] == 0
            ]
            if candidates:
                if my_player == 1:
                    candidates.sort(key=lambda rc: abs(rc[1] - center))
                else:
                    candidates.sort(key=lambda rc: abs(rc[0] - center))

        # ---- Full search ----
        work = [row[:] for row in board_matrix]
        engine = SearchEngine(
            my_player, size, ntable, work, time_limit=TIME_BUDGET,
        )
        return engine.search(work)

    # -----------------------------------------------------------------
    #  Opening Book
    # -----------------------------------------------------------------

    @staticmethod
    def _opening_move(
        board_matrix: List[List[int]],
        size: int,
        my_player: int,
        ntable,
    ) -> Optional[Tuple[int, int]]:
        """Return an opening move, or ``None`` to fall through to search."""
        center = size // 2
        if board_matrix[center][center] == 0:
            return (center, center)

        # Opponent took center → respond with a strong adjacent cell
        best: Optional[Tuple[int, int]] = None
        best_score = -INF
        for nr, nc in ntable[center][center]:
            if board_matrix[nr][nc] != 0:
                continue
            if my_player == 1:
                score = -abs(nc - center) + (size - abs(nr - center))
            else:
                score = -abs(nr - center) + (size - abs(nc - center))
            if score > best_score:
                best_score = score
                best = (nr, nc)
        return best

    # -----------------------------------------------------------------
    #  Large-Board Sparse Fast-Path
    # -----------------------------------------------------------------

    def _play_large_board(
        self,
        board_matrix: List[List[int]],
        size: int,
        my_player: int,
    ) -> Tuple[int, int]:
        """Lightweight move selection for boards > ``LARGE_BOARD_THRESHOLD``.

        Avoids full O(N²) search by focusing only on cells near existing
        stones, using connectivity scoring rather than tree search.
        """
        center = size // 2
        if board_matrix[center][center] == 0:
            return (center, center)

        occupied: List[Tuple[int, int]] = []
        for r in range(size):
            for c in range(size):
                if board_matrix[r][c] != 0:
                    occupied.append((r, c))

        ntable = build_neighbor_table(size)

        # Very few pieces: play near center and existing stones
        if len(occupied) <= LARGE_BOARD_FEW_PIECES:
            for nr, nc in ntable[center][center]:
                if board_matrix[nr][nc] == 0:
                    return (nr, nc)
            for r, c in occupied:
                for nr, nc in ntable[r][c]:
                    if board_matrix[nr][nc] == 0:
                        return (nr, nc)

        # Build candidate zone: cells within radius 2 of any piece
        zone: Set[Tuple[int, int]] = set()
        for r, c in occupied:
            for nr, nc in ntable[r][c]:
                if board_matrix[nr][nc] == 0:
                    zone.add((nr, nc))
                for nr2, nc2 in ntable[nr][nc]:
                    if board_matrix[nr2][nc2] == 0:
                        zone.add((nr2, nc2))

        if zone:
            opp = 3 - my_player
            candidates = list(zone)
            if len(candidates) > LARGE_BOARD_MAX_CANDIDATES:
                c_center = (size - 1) / 2.0
                candidates.sort(
                    key=lambda rc: abs(rc[0] - c_center) + abs(rc[1] - c_center)
                )
                candidates = candidates[:LARGE_BOARD_MAX_CANDIDATES]

            best_move: Optional[Tuple[int, int]] = None
            best_score = -INF
            for r, c in candidates:
                score = 0.0
                for nr, nc in ntable[r][c]:
                    if board_matrix[nr][nc] == my_player:
                        score += LARGE_BOARD_OWN_WEIGHT
                    elif board_matrix[nr][nc] == opp:
                        score += LARGE_BOARD_OPP_WEIGHT
                c_center = (size - 1) / 2.0
                score += (
                    (size - abs(r - c_center) - abs(c - c_center))
                    * LARGE_BOARD_CENTRALITY_WEIGHT
                )
                if my_player == 1:
                    score += (size - abs(c - c_center)) * AXIS_ALIGNMENT_WEIGHT
                else:
                    score += (size - abs(r - c_center)) * AXIS_ALIGNMENT_WEIGHT
                if score > best_score:
                    best_score = score
                    best_move = (r, c)
            return best_move

    # -----------------------------------------------------------------
    #  Player Identification
    # -----------------------------------------------------------------

    def _identify_player(
        self, board_matrix: List[List[int]], size: int,
    ) -> int:
        """Determine which player we are.

        Checks common player-attribute names on ``self``, then falls back
        to piece-count parity (Player 1 moves first, so equal counts →
        we are Player 1).
        """
        for attr in _PLAYER_ATTRIBUTES:
            val = getattr(self, attr, None)
            if val in (1, 2):
                return val
        count_1 = sum(row.count(1) for row in board_matrix)
        count_2 = sum(row.count(2) for row in board_matrix)
        return 1 if count_1 <= count_2 else 2


