import math
from heapq import heappush, heappop

# ---------------------------------------------------
# 1. Calcular matriz de distancias
# ---------------------------------------------------

def distancia(p1, p2):
    """Distancia Euclidiana entre dos puntos."""
    return math.dist(p1, p2)

def matriz_distancias(houses, corners):
    """Genera la matriz NxM con distancias d_ij."""
    N = len(houses)
    M = len(corners)
    D = [[0]*M for _ in range(N)]
    for i in range(N):
        for j in range(M):
            D[i][j] = distancia(houses[i], corners[j]) #type:ignore
    return D

# ---------------------------------------------------
# 2. Asignación Greedy con capacidad
# ---------------------------------------------------

def asignar_greedy(houses, corners, D, Cmax):
    """Asigna cada casa a la esquina más cercana respetando capacidad."""
    N = len(houses)
    M = len(corners)

    capacidad = [0]*M       # cuántas casas tiene cada esquina
    asignacion = [-1]*N     # asignación de cada casa

    for i in range(N):
        # ordenar esquinas por distancia d_ij
        orden = sorted(range(M), key=lambda j: D[i][j])
        for j in orden:
            if capacidad[j] < Cmax:
                asignacion[i] = j
                capacidad[j] += 1
                break

    return asignacion

# ---------------------------------------------------
# 3. Algoritmo de Prim para MST
# ---------------------------------------------------

def prim_mst(corners):
    """Construye el MST entre esquinas (completo) usando Prim."""
    M = len(corners)
    visited = [False]*M
    mst_edges = []
    total_weight = 0

    pq = []
    visited[0] = True

    # insertar aristas desde nodo 0
    for v in range(1, M):
        w = distancia(corners[0], corners[v])
        heappush(pq, (w, 0, v))

    while pq and len(mst_edges) < M-1:
        w, u, v = heappop(pq)
        if not visited[v]:
            visited[v] = True
            mst_edges.append((u, v, w))
            total_weight += w

            # añadir aristas desde v a los demás
            for k in range(M):
                if not visited[k]:
                    d = distancia(corners[v], corners[k])
                    heappush(pq, (d, v, k))

    return mst_edges, total_weight

# ---------------------------------------------------
# 4. Función objetivo
# ---------------------------------------------------

def costo_total(asignacion, D, mst_cost):
    """Costo = sum(distancias asignación) + costo del MST."""
    costo_asignacion = 0
    for i, j in enumerate(asignacion):
        costo_asignacion += D[i][j]
    return costo_asignacion + mst_cost

# ---------------------------------------------------
# 5. Algoritmo iterativo completo
# ---------------------------------------------------

def optimizar(houses, corners, Cmin=1, Cmax_final=5):
    D = matriz_distancias(houses, corners)
    mejor_costo = float("inf")
    mejor_asignacion = None
    mejor_mst = None

    for C in range(Cmin, Cmax_final+1):
        asign = asignar_greedy(houses, corners, D, C)
        mst_edges, mst_cost = prim_mst(corners)
        F = costo_total(asign, D, mst_cost)

        print(f"Capacidad={C}  Costo total={F:.2f}")

        if F < mejor_costo:
            mejor_costo = F
            mejor_asignacion = asign
            mejor_mst = mst_edges

    return mejor_asignacion, mejor_mst, mejor_costo

# ---------------------------------------------------
# 6. Ejemplo de uso 
# ---------------------------------------------------

if __name__ == "__main__":
    houses = [(1,1), (2,1), (4,2), (6,1)]
    corners = [(1,2), (4,1), (7,2)]

    asign, mst, costo = optimizar(houses, corners, Cmin=1, Cmax_final=2)

    print("\nMejor asignación:", asign)
    print("MST:", mst)
    print("Costo total:", costo)
