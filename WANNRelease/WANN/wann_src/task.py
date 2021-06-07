import numpy as np
import time
import sys
import random

from domain.make_env import make_env
from .ind import *
from domain.classify_gym import mnist_256, fashion_mnist
import pdb


class Task():
    """Problem domain to be solved by neural network. Uses OpenAI Gym patterns.
    """

    def __init__(self, game, paramOnly=False, nReps=1):
        """Initializes task environment

        Args:
          game - (string) - dict key of task to be solved (see domain/config.py)

        Optional:
          paramOnly - (bool)  - only load parameters instead of launching task?
          nReps     - (nReps) - number of trials to get average fitness
        """
        # Network properties
        self.nInput = game.input_size
        self.nOutput = game.output_size
        self.actRange = game.h_act
        self.absWCap = game.weightCap
        self.layers = game.layers
        self.activations = np.r_[np.full(1, 1), game.i_act, game.o_act]

        # Environment
        self.maxEpisodeLength = game.max_episode_length
        self.actSelect = game.actionSelect

        if not paramOnly:
            self.env = make_env(game.env_name)

        # Special needs...
        self.needsClosed = (game.env_name.startswith("CartPoleSwingUp"))

    def testInd(self, wVec, aVec, view=False, seed=-1):
        """Evaluate individual on task
        Args:
          wVec    - (np_array) - weight matrix as a flattened vector
                    [N**2 X 1]
          aVec    - (np_array) - activation function of each node
                    [N X 1]    - stored as ints (see applyAct in ann.py)

        Optional:
          view    - (bool)     - view trial?
          seed    - (int)      - starting random seed for trials

        Returns:
          fitness - (float)    - reward earned in trial
        """
        if seed >= 0:
            random.seed(seed)
            np.random.seed(seed)
            self.env.seed(seed)

        state = self.env.reset()
        self.env.t = 0

        annOut = act(wVec, aVec, self.nInput, self.nOutput, state)
        action = selectAct(annOut, self.actSelect)

        state, reward, done, info = self.env.step(action)
        if self.maxEpisodeLength == 0:
            return reward
        else:
            totalReward = reward

        for tStep in range(self.maxEpisodeLength):
            annOut = act(wVec, aVec, self.nInput, self.nOutput, state)
            action = selectAct(annOut, self.actSelect)
            state, reward, done, info = self.env.step(action)
            totalReward += reward
            if view:
                # time.sleep(0.01)
                if self.needsClosed:
                    self.env.render(close=done)
                else:
                    self.env.render()
            if done:
                break

        return totalReward

# -- 'Weight Agnostic Network' evaluation -------------------------------- -- #


    def setWeights(self, wVec, wVal):
        """Set single shared weight of network

        Args:
          wVec    - (np_array) - weight matrix as a flattened vector
                    [N**2 X 1]
          wVal    - (float)    - value to assign to all weights

        Returns:
          wMat    - (np_array) - weight matrix with single shared weight
                    [N X N]
        """
        # Create connection matrix
        wVec[np.isnan(wVec)] = 0
        dim = int(np.sqrt(np.shape(wVec)[0]))
        cMat = np.reshape(wVec, (dim, dim))
        cMat[cMat != 0] = 1.0

        # Assign value to all weights
        wMat = np.copy(cMat) * wVal
        return wMat

    def obtain_data(self, mnist=False):
        if mnist:
            # construct state for training data
            x_train, y_train = mnist_256(train=True)
            x_test, y_test = mnist_256(train=False)
        else:
            x_train, y_train = fashion_mnist(train=True)
            x_test, y_test = fashion_mnist(train=False)

        return x_train, y_train, x_test, y_test

    def predict(self, wVec, aVec, x, view=False, seed=-1, mnist=False):

        if seed >= 0:
            random.seed(seed)
            np.random.seed(seed)
            self.env.seed(seed)

        # train accuracy
        state = x
        annOut = act(wVec, aVec, self.nInput, self.nOutput, state)
        # these are the (soft max) outputs
        action = selectAct(annOut, self.actSelect)
        predictions = np.argmax(action, axis=1)
        return predictions

    def evaluateModel(self, wVec, aVec, hyp, mnist=False, seed=-1, nRep=False, nVals=6, view=False, returnVals=False):

        if nRep is False:
            nRep = hyp['alg_nReps']

        # Set weight values to test WANN with
        if (hyp['alg_wDist'] == "standard") and nVals == 6:  # Double, constant, and half signal
            wVals = np.array((-2, -1.0, -0.5, 0.5, 1.0, 2))
        else:
            wVals = np.linspace(-self.absWCap, self.absWCap, nVals)

        x_train, y_train, x_test, y_test = self.obtain_data(mnist)

        train_predictions = np.empty((nVals, y_train.shape[0]), dtype=np.float64)
        test_predictions = np.empty((nVals, y_test.shape[0]), dtype=np.float64)

        train_accuracies = np.empty((nRep, nVals), dtype=np.float64)
        test_accuracies = np.empty((nRep, nVals), dtype=np.float64)
        
        def get_majority_predictions(predictions):
            def _majority(l):
                return max(set(l), key=l.count)
            predictions = [_majority(list(predictions[:, i]))
                           for i in range(predictions.shape[1])]
            return predictions
        def calc_accuracy(predictions, ground_truths):
            n_correct = np.sum(predictions == ground_truths)
            return n_correct / ground_truths.shape[0]
        
        for iVal in range(nVals):
            wMat = self.setWeights(wVec, wVals[iVal])

            print('accuracy testing')

            train_prediction = self.predict(
                wMat, aVec, x_train, seed=seed, view=view)
            test_prediction = self.predict(
                wMat, aVec, x_test, seed=seed, view=view)

            train_predictions[iVal, :] = train_prediction
            test_predictions[iVal, :] = test_prediction

            train_accuracies[0, iVal] = calc_accuracy(train_prediction,y_train) 
            test_accuracies[0, iVal] = calc_accuracy(test_prediction,y_test)

        train_majority_prediction = get_majority_predictions(train_predictions)
        test_majority_prediction = get_majority_predictions(test_predictions)

        ensemble_accuracy_train = calc_accuracy(train_majority_prediction, y_train)
        ensemble_accuracy_test = calc_accuracy(test_majority_prediction, y_test)
        return train_accuracies, test_accuracies, ensemble_accuracy_train, ensemble_accuracy_test, train_predictions, test_predictions, train_majority_prediction, test_majority_prediction, y_train, y_test

    def usedInputs(self, wVec, aVec):
        nr_of_classes = 10 #currently hardcoded since both fashion and mnist consist of 10 classes
        wMat = self.setWeights(wVec, 1)
        outputs = wMat[:,-nr_of_classes:] 
        
        tree_dict = {}

        def usedInputsHelper(nodes_to_visit, visited_nodes):
            if len(nodes_to_visit) == 0:
                return visited_nodes

            head = nodes_to_visit[0]
            tail = nodes_to_visit[1:]

            if head < 256: ## we don't have to  visit input nodes
                visited_nodes.append(head)
                nodes_to_visit = tail 
                return usedInputsHelper(nodes_to_visit, visited_nodes)

            
            visited_nodes.append(head)
            nodes_to_visit = np.append(tail, np.where(wMat[:,head]>0)[0] )

            return usedInputsHelper(nodes_to_visit, visited_nodes)

        for i in reversed(range(1, nr_of_classes+1)):
            all_nodes = []
            nodes_of_interest = np.where(wMat[:,-i] > 0)[0]
            route = usedInputsHelper(nodes_of_interest, [])

            tree_dict[i%11] = sorted(route)

        return tree_dict



    def getDistFitness(self, wVec, aVec, hyp,
                       seed=-1, nRep=False, nVals=6, view=False, returnVals=False, accuracyTest=False):
        """Get fitness of a single individual with distribution of weights

        Args:
          wVec    - (np_array) - weight matrix as a flattened vector
                    [N**2 X 1]
          aVec    - (np_array) - activation function of each node 
                    [N X 1]    - stored as ints (see applyAct in ann.py)
          hyp     - (dict)     - hyperparameters
            ['alg_wDist']        - weight distribution  [standard;fixed;linspace]
            ['alg_absWCap']      - absolute value of highest weight for linspace

        Optional:
          seed    - (int)      - starting random seed for trials
          nReps   - (int)      - number of trials to get average fitness
          nVals   - (int)      - number of weight values to test


        Returns:
          fitness - (float)    - mean reward over all trials
        """
        if nRep is False:
            nRep = hyp['alg_nReps']

        # Set weight values to test WANN with
        if (hyp['alg_wDist'] == "standard") and nVals == 6:  # Double, constant, and half signal
            wVals = np.array((-2, -1.0, -0.5, 0.5, 1.0, 2))
        else:
            wVals = np.linspace(-self.absWCap, self.absWCap, nVals)

        # Get reward from 'reps' rollouts -- test population on same seeds
        reward = np.empty((nRep, nVals))
        train_accuracies = np.empty((nRep, nVals), dtype=np.float64)
        test_accuracies = np.empty((nRep, nVals), dtype=np.float64)
        for iRep in range(nRep):
            for iVal in range(nVals):
                wMat = self.setWeights(wVec, wVals[iVal])

                if accuracyTest:
                    print('accuracy test')
                    train_accuracy, test_accuracy = self.testIndividualAccuracy(
                        wMat, aVec, seed=seed, view=view)
                    train_accuracies[iRep, iVal] = train_accuracy
                    test_accuracies[iRep, iVal] = test_accuracy

                else:
                    if seed == -1:
                        reward[iRep, iVal] = self.testInd(
                            wMat, aVec, seed=seed, view=view)
                    else:
                        reward[iRep, iVal] = self.testInd(
                            wMat, aVec, seed=seed+iRep, view=view)

        if returnVals is True:
            return np.mean(reward, axis=0), wVals
        return np.mean(reward, axis=0)
