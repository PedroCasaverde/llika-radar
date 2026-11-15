"""PASCO (Lasalle et al.) adaptado a la red nacional de LLIKA (HALLAZGOS 117). Parte del cuaderno del proyecto de
Machine Learning de Pedro (Ciclo_III/ml/laboratorios/project1/project2/pasco_graph_clustering.ipynb), con tres cambios:
- arreglos de numpy en vez de diccionarios, para grafos de cientos de miles de nodos;
- Louvain (python-louvain) sobre el grafo reducido en vez de espectral, porque no se sabe cuántas comunidades hay;
- el grafo es la red bipartita entidad-proveedor de un año (aristas = casos, ponderadas por el log del monto).
Se mantienen el coarsening aleatorio multinivel (algoritmos 1 y 2 del paper), el lift, la alineación húngara y el voto
mayoritario entre R repeticiones (ecuación 3).

    from pasco import comunidades
    com = comunidades(casos_del_anio, R=5, rho=10)      # Serie: nodo ("E:ruc" o "P:ruc") -> comunidad
"""
import time
from concurrent.futures import ThreadPoolExecutor

import community as community_louvain
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment


def grafo_bipartito(casos):
    """casos: ruc_entidad, ruc, c_monto de un año → matriz de adyacencia simétrica y nombres de nodos."""
    ent = "E:" + casos["ruc_entidad"].astype(str)
    pro = "P:" + casos["ruc"].astype(str)
    nodos = pd.Index(pd.unique(pd.concat([ent, pro], ignore_index=True)))
    i, j = nodos.get_indexer(ent), nodos.get_indexer(pro)
    w = np.log1p(casos["c_monto"].astype(float).to_numpy())
    n = len(nodos)
    A = sp.coo_matrix((np.r_[w, w], (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    A.sum_duplicates()
    return A, nodos


def coarsen_un_nivel(A, rng):
    """Algoritmo 2: emparejamiento aleatorio de cada nodo libre con un vecino libre al azar."""
    n = A.shape[0]
    emparejado = np.zeros(n, dtype=bool)
    mapa = np.full(n, -1, dtype=np.int64)
    k = 0
    indptr, indices = A.indptr, A.indices
    for u in rng.permutation(n):
        if emparejado[u]:
            continue
        vec = indices[indptr[u]:indptr[u + 1]]
        libres = vec[~emparejado[vec]]
        libres = libres[libres != u]
        mapa[u] = k
        emparejado[u] = True
        if len(libres):
            v = libres[rng.integers(len(libres))]
            mapa[v] = k
            emparejado[v] = True
        k += 1
    H = sp.csr_matrix((np.ones(n), (np.arange(n), mapa)), shape=(n, k))
    Ac = (H.T @ A @ H).tocsr()
    Ac.setdiag(0)
    Ac.eliminate_zeros()
    return Ac, mapa


def coarsen_global(A, rho, rng):
    """Algoritmo 1: coarsening multinivel hasta ~N/rho nodos; devuelve el grafo reducido y el mapa nodo → hipernodo."""
    objetivo = max(A.shape[0] // rho, 100)
    mapa = np.arange(A.shape[0])
    while A.shape[0] > objetivo:
        n0 = A.shape[0]
        A, m = coarsen_un_nivel(A, rng)
        mapa = m[mapa]
        if A.shape[0] >= 0.95 * n0:
            break
    return A, mapa


def louvain(A, semilla):
    g = nx.from_scipy_sparse_array(A)
    part = community_louvain.best_partition(g, weight="weight", random_state=semilla)
    return np.array([part[i] for i in range(A.shape[0])])


def una_repeticion(A, rho, semilla):
    rng = np.random.default_rng(semilla)
    Ac, mapa = coarsen_global(A, rho, rng)
    return louvain(Ac, semilla)[mapa]          # lift: cada nodo hereda la comunidad de su hipernodo


def alinear(ref, otra):
    """Alineación húngara de las etiquetas de `otra` a las de `ref` (máximo solapamiento)."""
    cr, ir = np.unique(ref, return_inverse=True)
    co, io = np.unique(otra, return_inverse=True)
    m = max(len(cr), len(co))
    conf = sp.coo_matrix((np.ones(len(ref)), (ir, io)), shape=(m, m)).toarray()
    filas, cols = linear_sum_assignment(-conf)
    remap = np.empty(m, dtype=np.int64)
    remap[cols] = filas
    return remap[io]


def comunidades(casos, R=5, rho=10, semilla=0, hilos=5):
    """PASCO: R repeticiones de coarsening + Louvain en paralelo, alineadas y fusionadas por voto mayoritario."""
    t0 = time.time()
    A, nodos = grafo_bipartito(casos)
    with ThreadPoolExecutor(max_workers=hilos) as ex:
        parts = list(ex.map(lambda r: una_repeticion(A, rho, semilla + 1000 * r), range(R)))
    ref = parts[0]
    alineadas = np.vstack([ref] + [alinear(ref, p) for p in parts[1:]])
    voto = np.array([np.bincount(alineadas[:, k]).argmax() for k in range(alineadas.shape[1])])
    acuerdo = float((alineadas == voto).mean())
    print(f"   PASCO: {A.shape[0]:,} nodos, {A.nnz // 2:,} aristas, {len(np.unique(voto)):,} comunidades, "
          f"acuerdo entre repeticiones {acuerdo:.1%}, {time.time() - t0:.0f} s", flush=True)
    return pd.Series(voto, index=nodos, name="comunidad")
