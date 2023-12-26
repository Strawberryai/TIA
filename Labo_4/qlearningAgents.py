# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
import util

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        
        self.q_table = util.Counter()
        

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        # Igual hay que darle una vuelta a esto
        # print(self.q_table)
        
        if (state, action) not in self.q_table:
          self.q_table[(state, action)] = 0.0
        return self.q_table[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        acciones = self.getLegalActions(state)
        if not acciones:
          return 0.0
        
        max_value = float("-inf")
        for accion in acciones:
          q_value = self.getQValue(state, accion)
          
          if q_value > max_value:
            max_value = q_value

        return max_value
          

    def computeActionFromQValues(self, state):
        """
          Obtener la politica
        
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        acciones = self.getLegalActions(state)
        if not acciones:
          return None
        
        # Guarda las acciones de mayor q_value
        # (pueden haber varias acciones con el mismo q_value)
        politica = [] 
        for accion in acciones:
          q_value = self.getQValue(state, accion)
          # Si es la primera iteración
          if len(politica) == 0:
            politica = [(accion, q_value)]
          
          # Si encontramos una accion con mayor q_value
          elif q_value >= politica[0][1]:
            politica = [(accion, q_value)]
          
          # Si encontramos una accion con el mismo 1_value
          elif q_value == politica[0][1]:
            politica.append((accion, q_value))
        
        return random.choice(politica)[0]

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        
        acciones = self.getLegalActions(state)
        if not acciones:
          return None
       
        elif util.flipCoin(self.epsilon):
          return random.choice(acciones)

        else:
          return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        
        # Q(S, A) <- Q(S, A) + alpha * [R + discount * max_a(Q(S', a) - Q(S, A) ) ]
        max_a = self.getValue(nextState)
        q_value = self.q_table[(state, action)]
        self.q_table[(state, action)] = q_value + self.alpha * (reward + self.discount * max_a - q_value)
        

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()
        

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        
        # Q(S, A) <- w1 * f1 + w2 * f2 + ...
        # donde w son los weights y f las evaluaciones de los estados-features
        
        feats = self.featExtractor.getFeatures(state, action)
        
        q_value = 0.0
        for  feature, f in feats.items():
          w = self.weights[feature]
          q_value += w * f
        
        return q_value

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
       
        # w <- w + alpha * [diferencia] * f(S, A) 
        # diferencia <- [R + discount * max_a(Q(S', a'))] - Q(S, A)
        
        q_value = self.getQValue(state, action)
        max_a = self.getValue(nextState)
        diferencia = reward + self.discount * max_a - q_value

        # Actualizamos los pesos
        feats = self.featExtractor.getFeatures(state, action)
        for feature, f in feats.items():
          w = self.weights[feature]
          self.weights[feature] = w + self.alpha * diferencia * f
        

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
