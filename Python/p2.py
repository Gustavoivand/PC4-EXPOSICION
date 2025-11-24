"""
Ejemplo simplificado del enfoque:
- Se genera un grafo geométrico aleatorio que modela una red IoT.
- Se elige un nodo sumidero (sink).
- Cada nodo aprende, mediante Q-learning distribuido, el mejor vecino
  al que reenviar paquetes para llegar al sumidero.
- A partir de los Q-valores se construye un Árbol de Caminos Más Cortos (SPT)
  aproximado y se imprime el resultado.
"""

import random
import math
from collections import deque, defaultdict
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Parámetros del ejemplo
# ---------------------------------------------------------------------

N_NODOS = 15          # número de nodos en la red
RADIO = 0.45          # radio de comunicación (para el grafo geométrico)
SINK = 0              # índice del nodo sumidero
NUM_EPISODIOS = 4000  # episodios de entrenamiento Q-learning
MAX_PASOS = 2 * N_NODOS
ALPHA = 0.5           # tasa de aprendizaje
GAMMA = 0.9           # factor de descuento
EPSILON = 0.2         # prob. de exploración

RANDOM_SEED = 42      # para reproducibilidad
random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------
# Generación de grafo geométrico aleatorio
# ---------------------------------------------------------------------

def distancia(p1, p2):
    return math.dist(p1, p2)

def generar_grafo_geometrico(n, radio):
    """
    Genera:
    - posiciones: dict {nodo: (x, y)}
    - grafo: dict {nodo: [vecinos]}
    """
    posiciones = {i: (random.random(), random.random()) for i in range(n)}
    grafo = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            if distancia(posiciones[i], posiciones[j]) <= radio:
                grafo[i].append(j)
                grafo[j].append(i)

    return posiciones, grafo

def nodos_alcanzables(grafo, sink):
    """Devuelve el conjunto de nodos alcanzables desde sink (BFS)."""
    visitados = set()
    cola = deque([sink])
    visitados.add(sink)
    while cola:
        v = cola.popleft()
        for u in grafo[v]:
            if u not in visitados:
                visitados.add(u)
                cola.append(u)
    return visitados

# Generamos un grafo que, idealmente, sea conexo con respecto al sink
while True:
    posiciones, grafo = generar_grafo_geometrico(N_NODOS, RADIO)
    alcanzables = nodos_alcanzables(grafo, SINK)
    if len(alcanzables) == N_NODOS:
        break  # grafo conectado
    # Si no es conexo, se regenera

print("Grafo generado. Cada nodo tiene como promedio",
      sum(len(v) for v in grafo.values()) / N_NODOS, "vecinos.")

# ---------------------------------------------------------------------
# Q-learning distribuido para enrutamiento hacia el sink
# ---------------------------------------------------------------------

# Inicializar Q-valores: Q[v][u] para (v,u) vecinos
Q = {v: {u: 0.0 for u in vecinos} for v, vecinos in grafo.items()}

def elegir_accion_epsilon_greedy(v, epsilon):
    """Elige un vecino de v usando política epsilon-greedy."""
    vecinos = list(grafo[v])
    if not vecinos:
        return None
    if random.random() < epsilon:
        # Exploración
        return random.choice(vecinos)
    else:
        # Explotación: elegir vecino con mayor Q
        max_q = max(Q[v][u] for u in vecinos)
        mejores = [u for u in vecinos if Q[v][u] == max_q]
        return random.choice(mejores)

def recompensa(v, u):
    """Recompensa inmediata al ir de v a u."""
    if u == SINK:
        return 1.0  # recompensa positiva al llegar al sumidero
    return 0.0

def paso_q_learning(v):
    """
    Ejecuta un solo paso de Q-learning desde el nodo actual v.
    Devuelve el nuevo estado (nodo) tras la transición.
    """
    if v == SINK:
        return v  # ya estamos en el sumidero

    u = elegir_accion_epsilon_greedy(v, EPSILON)
    if u is None:
        return v  # sin vecinos, no se mueve (caso extremo)

    r = recompensa(v, u)

    # Q_max del siguiente estado
    if grafo[u]:
        max_q_sig = max(Q[u][w] for w in grafo[u])
    else:
        max_q_sig = 0.0

    # Actualización de Bellman
    Q[v][u] = (1 - ALPHA) * Q[v][u] + ALPHA * (r + GAMMA * max_q_sig)

    return u

# Entrenamiento
for episodio in range(NUM_EPISODIOS):
    # Elegimos un nodo de inicio aleatorio que no sea el sink
    v = random.choice([n for n in grafo.keys() if n != SINK])
    pasos = 0
    while v != SINK and pasos < MAX_PASOS:
        v = paso_q_learning(v)
        pasos += 1

# ---------------------------------------------------------------------
# Construcción del SPT aproximado a partir de los Q-valores
# ---------------------------------------------------------------------

def construir_politica_desde_Q():
    """
    A partir de Q se define un padre para cada nodo (excepto el sink):
    parent[v] = argmax_u Q[v][u].
    """
    parent = {}
    for v in grafo:
        if v == SINK or not grafo[v]:
            continue
        vecinos = grafo[v]
        max_q = max(Q[v][u] for u in vecinos)
        mejores = [u for u in vecinos if Q[v][u] == max_q]
        parent[v] = random.choice(mejores)
    return parent

parent = construir_politica_desde_Q()

def extraer_camino(parent, origen, sink):
    """
    Sigue parent[v] hasta el sink o hasta detectar un ciclo.
    Devuelve (camino, llega_a_sink: bool).
    """
    camino = [origen]
    visitados = {origen}
    actual = origen

    while actual != sink and actual in parent:
        actual = parent[actual]
        if actual in visitados:
            # Se detectó un ciclo
            camino.append(actual)
            return camino, False
        camino.append(actual)
        visitados.add(actual)

    return camino, (actual == sink)

# ---------------------------------------------------------------------
# Cálculo de distancias reales de camino más corto (para comparación)
# ---------------------------------------------------------------------

def distancias_minimas_desde_sink(grafo, sink):
    """
    Distancias (en número de saltos) desde sink al resto de nodos usando BFS.
    """
    dist = {n: math.inf for n in grafo}
    dist[sink] = 0
    cola = deque([sink])
    while cola:
        v = cola.popleft()
        for u in grafo[v]:
            if dist[u] == math.inf:
                dist[u] = dist[v] + 1
                cola.append(u)
    return dist

dist_real = distancias_minimas_desde_sink(grafo, SINK)

# ---------------------------------------------------------------------
# Mostrar resultados
# ---------------------------------------------------------------------

print("\n=== POSICIONES DE NODOS (x, y) ===")
for v, (x, y) in posiciones.items():
    print(f"Nodo {v}: ({x:.3f}, {y:.3f})")

print("\n=== VECINDARIO DE CADA NODO ===")
for v, vecinos in grafo.items():
    print(f"{v}: vecinos -> {sorted(vecinos)}")

print("\n=== PADRES SEGÚN POLÍTICA Q-LEARNING (SPT aproximado) ===")
for v in sorted(grafo.keys()):
    if v == SINK:
        print(f"{v}: es el SUMIDERO (raíz del árbol)")
    elif v in parent:
        print(f"{v}: padre -> {parent[v]}")
    else:
        print(f"{v}: SIN padre definido (posible problema de aprendizaje o aislamiento)")

print("\n=== CAMINOS INDUCIDOS POR LA POLÍTICA HACIA EL SINK ===")
for v in sorted(grafo.keys()):
    if v == SINK:
        continue
    camino, ok = extraer_camino(parent, v, SINK)
    estado = "OK" if ok else "ciclo / truncado"
    print(f"Nodo {v}: camino -> {camino}  [{estado}]")

print("\n=== COMPARACIÓN CON DISTANCIAS REALES (BFS) ===")
for v in sorted(grafo.keys()):
    if v == SINK:
        continue
    camino, ok = extraer_camino(parent, v, SINK)
    if ok:
        long_camino = len(camino) - 1  # número de saltos
    else:
        long_camino = None

    print(f"Nodo {v}: dist_real={dist_real[v]:2}, "
          f"dist_politica={long_camino if long_camino is not None else 'N/A'}")

# ---------------------------------------------------------------------
# Visualización del grafo con SPT resaltado
# ---------------------------------------------------------------------

def visualizar_grafo_con_spt(posiciones, grafo, parent, sink):
    """
    Visualiza el grafo geométrico con:
    - Nodos en azul, excepto el sink en rojo
    - Aristas del grafo en gris claro
    - Aristas del SPT (parent) en verde más grueso
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Dibujar todas las aristas del grafo en gris claro
    for v in grafo:
        for u in grafo[v]:
            if v < u:  # evitar dibujar dos veces
                x1, y1 = posiciones[v]
                x2, y2 = posiciones[u]
                ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.3, linewidth=1, zorder=1)
    
    # Dibujar aristas del SPT (parent) en verde más grueso
    for v, padre in parent.items():
        x1, y1 = posiciones[v]
        x2, y2 = posiciones[padre]
        ax.plot([x1, x2], [y1, y2], 'green', linewidth=2.5, zorder=2, label='SPT' if v == list(parent.keys())[0] else '')
    
    # Dibujar nodos
    for v in grafo:
        x, y = posiciones[v]
        if v == sink:
            # Nodo sumidero en rojo
            ax.scatter(x, y, s=300, c='red', marker='s', zorder=3, edgecolors='darkred', linewidth=2)
            ax.text(x, y, str(v), ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        else:
            # Nodos normales en azul
            ax.scatter(x, y, s=300, c='lightblue', marker='o', zorder=3, edgecolors='darkblue', linewidth=2)
            ax.text(x, y, str(v), ha='center', va='center', fontsize=10, fontweight='bold', color='darkblue')
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title(f'Red IoT: Grafo Geométrico con SPT\n(Sink={sink} en rojo, SPT en verde)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', edgecolor='darkred', label='Nodo Sumidero (Sink)'),
        Patch(facecolor='lightblue', edgecolor='darkblue', label='Nodos Normales'),
        plt.Line2D([0], [0], color='green', linewidth=2.5, label='Aristas SPT'),
        plt.Line2D([0], [0], color='gray', linewidth=1, alpha=0.3, label='Aristas del Grafo')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig, ax

# Generar y mostrar la visualización
fig, ax = visualizar_grafo_con_spt(posiciones, grafo, parent, SINK)
plt.show()
