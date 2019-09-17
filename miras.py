# mira.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()
        self.weights = {}

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        weights = {}
        def find_best_c(c):

            # Reset the weights
            self.weights = dict((label, util.Counter()) for label in self.legalLabels)

            # the amount of iterations to do, default to 3
            print "Finding the best C value; currently checking: ", c
            for iteration in range(self.max_iterations):
                print "Starting iteration (MIRA)", iteration, "..."
                for features, real in zip(trainingData, trainingLabels):
                    # get prediction
                    prediction = self.classify([features])[0]
                    # if prediction = real, no need to adjust weights.
                    if real == prediction:
                        pass
                    # find the modifier via the formula
                    modifier = min([
                            c,
                            ((self.weights[prediction] - self.weights[real]) * features + 1.)
                            / (2 * (features * features))
                            ])
                    # copy the features
                    copy = features.copy()
                    # get the key and the value from the features
                    for key, value in copy.items():
                        # multiply the value * modifier and set the copy
                        copy[key] = value * modifier
                    # add to the real weight, subtract from the prediction weight
                    self.weights[real] += copy
                    self.weights[prediction] -= copy

            # set the weights for the c score calculated in the for loop
            weights[c] = self.weights
            # sum how many predictions match the real results in the validation data.
            return sum(int(real == prediction) for real, prediction in zip(validationLabels,
                                                       self.classify(validationData)))

        # loop through the definition defined above for the given c scores
        c_scores = [find_best_c(c) for c in Cgrid]

        # Pick out the best value for C, choosing the lower value in the case of ties
        max_c, max_c_score = Cgrid[0], -1

        for c, c_score in zip(Cgrid, c_scores):

            if c_score > max_c_score or \
              (c_score == max_c_score and c < max_c):
              max_c, max_c_score = c, c_score

        # training is over, set the weight to best c score weights and the C value to the max_c
        self.weights = weights[max_c]
        self.C = max_c
        return max_c

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses