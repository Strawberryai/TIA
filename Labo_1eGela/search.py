# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from abc import ABC, abstractmethod

import util


class SearchProblem(ABC):
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    @abstractmethod
    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    @abstractmethod
    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    @abstractmethod
    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    @abstractmethod
    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    print([s, s, w, s, w, w, s, w])
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    
    print("---- Depth First Search")
    #print(problem.getStartState())
    #print(problem.getSuccessors(problem.getStartState()))
    #print(problem.getSuccessors((4, 5)))
    #print(problem.isGoalState((1,1)))
    #porVisitar = [(problem.getStartState(), [])]
    porVisitar = util.Stack([(problem.getStartState(), [])]) #Stack -> Inicializado con la posición inicial y la lista de movimientos hasta el estado
    visitados = set() # Set
    nodoFinal = None

    while nodoFinal is None and len(porVisitar) != 0:
        nodoAct = porVisitar.pop() # Visitamos el siguiente nodo

        if nodoAct[0] not in visitados:
            # No habiamos visitado anteriormente este nodo y registramos la posición actual como visitada
            visitados.add(nodoAct[0])

            if problem.isGoalState(nodoAct[0]):
                # Hemos encontrado el camino
                nodoFinal = nodoAct
            else:
                # Aun pueden quedar estados por expandir
                sucesores = problem.getSuccessors(nodoAct[0])

                for s in sucesores:
                    # Copiamos el camino del padre y añadimos la acción para llegar al nuevo estado
                    camino = nodoAct[1].copy()
                    camino.append(s[1])
                    pos = s[0]
                    porVisitar.push((pos, camino))

    if nodoFinal is None:
        # No hemos encontrado un camino que lleve al objetivo
        return []
    
    # Devolvemos el camino hasta el objetivo
    return nodoFinal[1]




def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    # Es igual que el DFS pero en lugar de utilizar un stack usamos una cola
    # Solo cambia el nodo que miramos primero
    porVisitar = util.Queue([(problem.getStartState(), [])]) #Cola -> Inicializado con la posición inicial y la lista de movimientos hasta el estado
    visitados = set() # Set
    nodoFinal = None

    while nodoFinal is None and not porVisitar.isEmpty():
        nodoAct = porVisitar.pop() # Visitamos el siguiente nodo

        if nodoAct[0] not in visitados:
            # No habiamos visitado anteriormente este nodo y registramos la posición actual como visitada
            visitados.add(nodoAct[0])

            if problem.isGoalState(nodoAct[0]):
                # Hemos encontrado el camino
                nodoFinal = nodoAct
            else:
                # Aun pueden quedar estados por expandir
                sucesores = problem.getSuccessors(nodoAct[0])

                for s in sucesores:
                    # Copiamos el camino del padre y añadimos la acción para llegar al nuevo estado
                    camino = nodoAct[1].copy()
                    camino.append(s[1])
                    pos = s[0]
                    porVisitar.push((pos, camino))

    if nodoFinal is None:
        # No hemos encontrado un camino que lleve al objetivo
        return []
    
    # Devolvemos el camino hasta el objetivo
    return nodoFinal[1]


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    porVisitar = util.PriorityQueue() #Cola de prioridad -> Inicializado con la posición inicial y la lista de movimientos hasta el estado
    porVisitar.push((problem.getStartState(), [], 0), 0)
    visitados = set() # Set
    nodoFinal = None

    while nodoFinal is None and not porVisitar.isEmpty():
        nodoAct = porVisitar.pop() # Visitamos el siguiente nodo

        if nodoAct[0] not in visitados:
            # No habiamos visitado anteriormente este nodo y registramos la posición actual como visitada
            visitados.add(nodoAct[0])

            if problem.isGoalState(nodoAct[0]):
                # Hemos encontrado el camino
                nodoFinal = nodoAct
            else:
                # Aun pueden quedar estados por expandir
                sucesores = problem.getSuccessors(nodoAct[0])

                for s in sucesores:
                    # Copiamos el camino del padre y añadimos la acción para llegar al nuevo estado
                    pos = s[0]
                    camino = nodoAct[1].copy()
                    camino.append(s[1])
                    coste = nodoAct[2] + s[2]
                    porVisitar.push((pos, camino, coste), coste)

    if nodoFinal is None:
        # No hemos encontrado un camino que lleve al objetivo
        return []
    
    # Devolvemos el camino hasta el objetivo
    return nodoFinal[1]


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    # INICIALIZACIÓN
    porVisitar = util.PriorityQueue() #Cola de prioridad -> Inicializado con la posición inicial y la lista de movimientos hasta el estado
    posInicial = problem.getStartState()
    costeInicial = heuristic(posInicial, problem)
    porVisitar.push((posInicial, [], costeInicial), costeInicial)
    visitados = set() # Set
    nodoFinal = None

    # ALGORITMO
    while nodoFinal is None and not porVisitar.isEmpty():
        nodoAct = porVisitar.pop() # Visitamos el siguiente nodo

        if nodoAct[0] not in visitados:
            # No habiamos visitado anteriormente este nodo y registramos la posición actual como visitada
            visitados.add(nodoAct[0])
            # El coste de llegar hasta el nodo actual es el coste acumulado menos el heurístico del nodo actual. 
            # Antes se sumó el heurístico, por lo que para saber el coste hay que restarlo
            costeNodoAct = nodoAct[2] - heuristic(nodoAct[0], problem) 

            if problem.isGoalState(nodoAct[0]):
                # Hemos encontrado el camino
                nodoFinal = nodoAct
            else:
                # Aun pueden quedar estados por expandir
                sucesores = problem.getSuccessors(nodoAct[0])

                for s in sucesores:
                    # Copiamos el camino del padre y añadimos la acción para llegar al nuevo estado
                    pos = s[0]
                    camino = nodoAct[1].copy()
                    camino.append(s[1])
                    
                    # El coste del sucesor va a ser el coste acumulado hasta el más el heurístico del sucesor
                    coste =  costeNodoAct + s[2] + heuristic(pos, problem) 
                    porVisitar.push((pos, camino, coste), coste)

    # RETURN
    if nodoFinal is None:
        # No hemos encontrado un camino que lleve al objetivo
        return []
    
    # Devolvemos el camino hasta el objetivo
    return nodoFinal[1]


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
