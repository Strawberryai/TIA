# multiAgents.py
# --------------
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


import random

import util
from game import Agent
from util import manhattanDistance


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        " YOUR CODE HERE "
        curFood=currentGameState.getFood().asList()
        posicionesNewComida = []
        posicionFantasmas = []
        distanciaTotalCom=1
        for comida in curFood:
            posicionesNewComida.append(util.manhattanDistance(newPos, comida))
            distanciaTotalCom+=util.manhattanDistance(newPos, comida)
            #print(distanciaTotalCom)
        distNuevaCom = 1/(min(posicionesNewComida) + 1)
        for fantasmas in successorGameState.getGhostPositions():
            posicionFantasmas.append(util.manhattanDistance(newPos,fantasmas))
        posFant=min(posicionFantasmas)
        fantMasCerc = 1/(posFant+1)
        if(posFant<5):
            fantMasCerc=fantMasCerc*4
        #print(fantMasCerc)
        #print(action + " -->" + str(1 / distNuevaCom - 1 / fantMasCerc))
        if(len(newFood)==0):
            return(99999999)
        resultado=distNuevaCom - fantMasCerc-0.1*distanciaTotalCom/(len(newFood))
        #print(action + " -->"+str(resultado))
        return (10000*resultado) #por decimales


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__()
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def maxvalue(estadoAct,agentIndex, numFant, profundidad ):
          #Si llegamos al final
          if profundidad == 0 or estadoAct.isWin() or estadoAct.isLose():
            return self.evaluationFunction(estadoAct)
          puntos = -9999999
          acciones= estadoAct.getLegalActions(agentIndex)
          for accion in acciones:#cambios a minimizar para los fantasmas
            puntos = max(puntos, minvalue(estadoAct.generateSuccessor(agentIndex, accion), agentIndex+1, numFant, profundidad))
          return puntos

        def minvalue(estadoAct,agentIndex, numFant, profundidad):
          #Si llegamos al final
          if profundidad == 0 or estadoAct.isWin() or estadoAct.isLose():
            return self.evaluationFunction(estadoAct)
          
          acciones = estadoAct.getLegalActions(agentIndex)
          puntos = 999999
          for accion in acciones:
            if  agentIndex == numFant: #si es pacMan cambiamos a maximizar
              puntos = min(puntos, maxvalue(estadoAct.generateSuccessor( agentIndex, accion), 0, numFant, profundidad-1))
            else:   #si es otro fantasma seguimos
              puntos = min(puntos, minvalue(estadoAct.generateSuccessor( agentIndex, accion),agentIndex + 1, numFant, profundidad ))
          return puntos

        puntos, optAction = -99999, None
        numFant = game_state.getNumAgents()- 1
        acciones = game_state.getLegalActions(0)
        for accion in acciones:
          curPunts = minvalue(game_state.generateSuccessor(0, accion), 1, numFant, self.depth)
          if curPunts > puntos:
            puntos, optAction = curPunts, accion
        return optAction
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def min_value(game_state, agentIndex, depth, alpha, beta):
          # Comprobar si es un estado final
          if depth == 0 or game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state), None
          
          # Inicialización
          value = float("inf")
          optAction = None
          
          # Algoritmo
          for action in game_state.getLegalActions(agentIndex):
            if agentIndex == game_state.getNumAgents() -1:
              # Se trata del Pac-Man -> cambiamos a maximizar
              v, a = max_value(game_state.generateSuccessor(agentIndex, action), 0, depth -1, alpha, beta)
              
            else:
              # Se trata de un fantasma -> seguimoms minimizando
              v, a = min_value(game_state.generateSuccessor(agentIndex, action), agentIndex+1, depth, alpha, beta)

            if v < value:
              value, optAction = v, action
          
            if value < alpha:
              return value, optAction
            
            beta = min(beta, value)
          
          return value, optAction
        
        
        def max_value(game_state, agentIndex, depth, alpha, beta):
          # Comprobar si es un estado final
          if depth == 0 or game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state), None
          
          # Inicialización
          value = -float("inf")
          optAction = None
          
          # Algoritmo
          for action in game_state.getLegalActions(agentIndex):
            v, a = min_value(game_state.generateSuccessor(agentIndex, action), agentIndex+1, depth, alpha, beta)
            
            if v > value:
              value, optAction = v, action
            
            if value > beta:
              return value, optAction
            
            alpha = max(alpha, v)
          
          return value, optAction
          
          
          alpha = -float("inf")
          beta = float("inf")
          
          value, optAction = max_value(game_state, 0, self.depth, alpha, beta)
          
          return value, optAction
        
        
        # LLamada principal
        alpha = -float("inf")
        beta = float("inf")
        
        value, optAction = max_value(game_state, 0, self.depth, alpha, beta)
        
        return optAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
      """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
      """
      "*** YOUR CODE HERE ***"
      
      def exp_value(game_state, agentIndex, depth):
        # Comprobar si es un estado final
        if depth == 0 or game_state.isWin() or game_state.isLose():
          return self.evaluationFunction(game_state)

        value = 0
        actions = game_state.getLegalActions(agentIndex)
        
        for action in actions:
          if agentIndex == game_state.getNumAgents() -1:
            # Se trata del Pac-Man -> cambiamos a maximizar
            v, a = max_value(game_state.generateSuccessor(agentIndex, action), 0, depth -1)
      
          else:
            # Se trata de un fantasma -> seguimoms minimizando
            v = exp_value(game_state.generateSuccessor(agentIndex, action), agentIndex+1, depth)
          
          p = 1 / len(actions) # La probabilidad es uniforme
          value += p * v
        
        return  value
        
      def max_value(game_state, agentIndex, depth):
        # Comprobar si es un estado final
        if depth == 0 or game_state.isWin() or game_state.isLose():
          return self.evaluationFunction(game_state), None
        
        # Inicialización
        value = -float("inf")
        optAction = None
        
        # Algoritmo
        for action in game_state.getLegalActions(agentIndex):
          v = exp_value(game_state.generateSuccessor(agentIndex, action), agentIndex+1, depth)
          
          if v > value:
            value, optAction = v, action
        
        return value, optAction
        
      value, optAction = max_value(gameState, 0, self.depth)
      
      return optAction


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    pacman_pos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    capsules = currentGameState.getCapsules()    
    
    "*** YOUR CODE HERE ***"
    # Devolver casos finales
    if currentGameState.isWin():
      return float("inf")
    if currentGameState.isLose():
      return -float("inf")
    if len(newFood) == 0:
      return float("inf")
    
    # Calcular la distancia más corta a una comida    
    min_distancia_comida = min([manhattanDistance(pacman_pos, food) for food in newFood])

    # Calcular la distancia más corta a un fantasma asustado
    fantasmas_austados  = [g for i, g in enumerate(newGhostStates) if newScaredTimes[i] > 0]
    min_dist_asust = 0
    if len(fantasmas_austados) > 0:
      min_dist_asust = min([manhattanDistance(pacman_pos, g.getPosition()) for g in fantasmas_austados])
    
    
    # Calcular la distancia más corta a un fantasma no asustado
    fantasmas_no_aust   = [g for g in newGhostStates if g not in fantasmas_austados]
    min_dist_no_asust = 0
    if len(fantasmas_no_aust) > 0:
      min_dist_asust = min([manhattanDistance(pacman_pos, g.getPosition()) for g in fantasmas_no_aust])
    
    # Consideramos el número de capsulas
    num_capsulas = len(capsules)

    # Consideramos la puntuación actual
    puntuacion = currentGameState.getScore()
    
    # Finalmente, valoramos el estado
    evaluation = puntuacion                         # A mayor puntuación  -> Mejor
    evaluation += 5 / (min_distancia_comida +0.1)   # A menor distancia   -> Mejor    | +0.1 para evitar divisiones por 0
    evaluation += 3 / (min_dist_asust +1)           # A menor distancia   -> Mejor    | +1 para evitar divisiones por 0
    evaluation -= 10 * min_dist_no_asust            # A mayor distancia   -> Mejor
    evaluation -= 2 * len(newFood)                  # A mayor numero      -> Peor
    evaluation -= 4 * num_capsulas                  # A mayor numero      -> Peor (queremos que las coma)
    
    return evaluation


# Abbreviation
better = betterEvaluationFunction