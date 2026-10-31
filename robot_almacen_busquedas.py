"""
Proyecto integrador: búsqueda no informada e informada para un robot de almacén.
Implementación original en Python (BFS, DFS y A*).

Estado: (fila, columna, mascara)
- La máscara usa 3 bits: 1, 2 y 4 indican paquetes recogidos.
- Cada movimiento cuesta 1.
- El robot recoge automáticamente un paquete al entrar en su celda.
"""

from collections import deque
from heapq import heappush, heappop
from itertools import count
from time import perf_counter

N = 10
INICIO = (0, 0)
PAQUETES = ((2, 8), (7, 2), (9, 9))

# Celdas bloqueadas; se mantienen iguales para los tres algoritmos.
OBSTACULOS = {
    (0, 3), (1, 3), (2, 3), (3, 3),
    (4, 6), (5, 6), (6, 6),
    (6, 1), (6, 2), (6, 3),
    (8, 7), (8, 8)
}

ACCIONES = (
    (-1, 0, "ARRIBA"),
    (1, 0, "ABAJO"),
    (0, -1, "IZQUIERDA"),
    (0, 1, "DERECHA"),
)
META_MASCARA = (1 << len(PAQUETES)) - 1


def actualizar_mascara(fila, columna, mascara):
    """Marca como recogidos los paquetes ubicados en la nueva celda."""
    for i, posicion in enumerate(PAQUETES):
        if (fila, columna) == posicion:
            mascara |= (1 << i)
    return mascara


def sucesores(estado):
    """Genera estados alcanzables mediante un movimiento válido."""
    fila, columna, mascara = estado
    for df, dc, nombre in ACCIONES:
        nf, nc = fila + df, columna + dc
        if 0 <= nf < N and 0 <= nc < N and (nf, nc) not in OBSTACULOS:
            nueva_mascara = actualizar_mascara(nf, nc, mascara)
            yield (nf, nc, nueva_mascara), nombre


def es_meta(estado):
    return estado[2] == META_MASCARA


def distancia_manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def heuristica(estado):
    """
    Cota inferior admisible:
    distancia al paquete pendiente más cercano + árbol de expansión
    mínima (MST) entre los paquetes pendientes, usando Manhattan.
    No sobreestima porque ignora obstáculos y permite una cota inferior.
    """
    fila, columna, mascara = estado
    pendientes = [
        p for i, p in enumerate(PAQUETES)
        if not (mascara & (1 << i))
    ]
    if not pendientes:
        return 0

    costo = min(distancia_manhattan((fila, columna), p) for p in pendientes)

    # Prim sobre los paquetes que faltan.
    conectados = {pendientes[0]}
    no_conectados = set(pendientes[1:])
    while no_conectados:
        arista = min(
            (distancia_manhattan(a, b), b)
            for a in conectados for b in no_conectados
        )
        costo += arista[0]
        conectados.add(arista[1])
        no_conectados.remove(arista[1])
    return costo


def reconstruir_camino(padres, estado_final):
    estados = [estado_final]
    while estados[-1] in padres:
        estados.append(padres[estados[-1]][0])
    estados.reverse()
    acciones = []
    for estado in estados[1:]:
        acciones.append(padres[estado][1])
    return estados, acciones


def bfs(inicial):
    """Búsqueda en anchura. Óptima cuando todos los pasos cuestan lo mismo."""
    frontera = deque([inicial])
    visitados = {inicial}
    padres = {}
    expandidos = 0

    while frontera:
        actual = frontera.popleft()
        if es_meta(actual):
            estados, acciones = reconstruir_camino(padres, actual)
            return estados, acciones, expandidos

        expandidos += 1
        for siguiente, accion in sucesores(actual):
            if siguiente not in visitados:
                visitados.add(siguiente)
                padres[siguiente] = (actual, accion)
                frontera.append(siguiente)
    return None, None, expandidos


def dfs(inicial):
    """Búsqueda en profundidad con control de estados visitados."""
    frontera = [inicial]
    visitados = {inicial}
    padres = {}
    expandidos = 0

    while frontera:
        actual = frontera.pop()
        if es_meta(actual):
            estados, acciones = reconstruir_camino(padres, actual)
            return estados, acciones, expandidos

        expandidos += 1
        # Se invierte para mantener un orden de exploración reproducible.
        for siguiente, accion in reversed(list(sucesores(actual))):
            if siguiente not in visitados:
                visitados.add(siguiente)
                padres[siguiente] = (actual, accion)
                frontera.append(siguiente)
    return None, None, expandidos


def astar(inicial):
    """A* con f(n)=g(n)+h(n), usando una cola de prioridad."""
    contador = count()
    frontera = []
    heappush(frontera, (heuristica(inicial), 0, next(contador), inicial))
    mejor_g = {inicial: 0}
    padres = {}
    expandidos = 0

    while frontera:
        _, g_actual, _, actual = heappop(frontera)
        if g_actual != mejor_g.get(actual):
            continue  # Entrada obsoleta en la cola.

        if es_meta(actual):
            estados, acciones = reconstruir_camino(padres, actual)
            return estados, acciones, expandidos

        expandidos += 1
        for siguiente, accion in sucesores(actual):
            nuevo_g = g_actual + 1
            if nuevo_g < mejor_g.get(siguiente, float("inf")):
                mejor_g[siguiente] = nuevo_g
                padres[siguiente] = (actual, accion)
                f = nuevo_g + heuristica(siguiente)
                heappush(frontera, (f, nuevo_g, next(contador), siguiente))
    return None, None, expandidos


def ejecutar(nombre, algoritmo, inicial):
    inicio = perf_counter()
    estados, acciones, expandidos = algoritmo(inicial)
    tiempo = perf_counter() - inicio
    if estados is None:
        return {
            "algoritmo": nombre, "encontrada": False,
            "nodos_expandidos": expandidos, "pasos": None,
            "costo": None, "tiempo_segundos": tiempo
        }
    return {
        "algoritmo": nombre, "encontrada": True,
        "nodos_expandidos": expandidos, "pasos": len(acciones),
        "costo": len(acciones), "tiempo_segundos": tiempo,
        "acciones": acciones, "estados": estados
    }


def mostrar_mapa():
    """Imprime S=inicio, #=obstáculo, 1-3=paquetes, .=libre."""
    simbolos = {INICIO: "S"}
    for i, p in enumerate(PAQUETES, start=1):
        simbolos[p] = str(i)
    print("Mapa del almacén (10x10):")
    for f in range(N):
        print(" ".join(
            "#" if (f, c) in OBSTACULOS else simbolos.get((f, c), ".")
            for c in range(N)
        ))


def main():
    inicial = (INICIO[0], INICIO[1],
               actualizar_mascara(INICIO[0], INICIO[1], 0))
    mostrar_mapa()
    resultados = [
        ejecutar("BFS", bfs, inicial),
        ejecutar("DFS", dfs, inicial),
        ejecutar("A*", astar, inicial),
    ]

    print("\\nBENCHMARK")
    print(f"{'Algoritmo':<12}{'Expandidos':>13}{'Pasos':>10}{'Costo':>10}{'Tiempo(s)':>14}")
    for r in resultados:
        print(f"{r['algoritmo']:<12}{r['nodos_expandidos']:>13}"
              f"{str(r['pasos']):>10}{str(r['costo']):>10}"
              f"{r['tiempo_segundos']:>14.6f}")
        if r["encontrada"]:
            print("  Ruta:", " -> ".join(r["acciones"]))
    print("\\nNota: el tiempo depende del equipo y de la carga del sistema.")


if __name__ == "__main__":
    main()
