import pickle
import math
import random
import lab2


def B(q):
    """
    Calculating entropy
    :param q: probability of an attribute
    :return: Entropy of q
    """
    if q == 0 or q == 1:
        return math.log(1,2)
    return -((q * math.log(q,2)) + ((1-q)*math.log((1-q),2)))


def informationGain(attributeNo, input, p, n):
    """
    Calculating information gain
    :param attributeNo: attrribute number/ feature number
    :param input: input list containing examples with feature True/False values
    :param p: number of examples in english
    :param n: number of examples in dutch
    :return: the information gain, left and right list of the attribute
    """
    #Calculating remainder
    trueEn = 0
    trueNl = 0
    falseEn = 0
    falseNl = 0
    left = list()
    right = list()
    rowLen = len(input[0]) - 1
    for eachRow in input:
        if eachRow[rowLen] == "en" and eachRow[attributeNo] == True:
            trueEn = trueEn + 1
            left.append(eachRow)
        elif eachRow[rowLen] == "en" and eachRow[attributeNo] == False:
            falseEn = falseEn + 1
            right.append(eachRow)
        elif eachRow[rowLen] == "nl" and eachRow[attributeNo] == True:
            trueNl = trueNl + 1
            left.append(eachRow)
        elif eachRow[rowLen] == "nl" and eachRow[attributeNo] == False:
            falseNl = falseNl + 1
            right.append(eachRow)
    if falseEn == 0 or falseNl == 0 or trueEn == 0 or trueNl == 0:
        rem = 0
        if falseEn == 0 and falseNl == 0:
            rem = rem + ((0) * B(0))
        elif falseEn == 0:
            rem = rem + (((falseNl) / (p + n)) * B(0))
        elif falseNl == 0:
            rem = rem + (((falseEn) / (p + n)) * B(1))
        else:
            rem = rem + (((falseEn + falseNl) / (p + n)) * B(falseEn / (falseEn + falseNl)))

        if trueEn == 0 and trueNl == 0:
            rem = rem + ((0) * B(0))
        elif trueEn == 0:
            rem = rem + (((trueNl) / (p + n)) * B(0))
        elif trueNl == 0:
            rem = rem + (((trueEn) / (p + n)) * B(1))
        else:
            rem = rem + (((trueEn + trueNl) / (p + n)) * B(trueEn / (trueEn + trueNl)))
    else:
        rem = (((trueEn + trueNl) / (p + n)) * B(trueEn / (trueEn + trueNl))) + (
                    ((falseEn + falseNl) / (p + n)) * B(falseEn / (falseEn + falseNl)))
    #returning info gain
    return (B(p/(p+n)) - rem), left, right


def pickAttribute(input, p, n):
    """
    Determines with attribute provides maximum information gain
    :param input: input list containing examples with feature True/False values
    :param p: number of examples in english
    :param n: number of examples in dutch
    :return: attribute number that provides maximum information gain,
    list of it's true values. list of it's false values, information gain value
    """
    maxInfoGain = -1
    maxAttributeNo = 0
    maxLeft = list()
    maxRight = list()
    for attributeNo in range(6):
        infoGain, left, right = informationGain(attributeNo, input, p, n)
        if infoGain > maxInfoGain:
            maxInfoGain = infoGain
            maxAttributeNo = attributeNo
            maxLeft = left
            maxRight = right
    return maxAttributeNo, maxLeft, maxRight, maxInfoGain


def calError(input, weights, attributeNo):
    """
    Calculate the sum of weights of the inputs that are incorrectly classified
    :param input: list containing examples
    :param weights: weights of the examples
    :param attributeNo: attribute number to check against
    :return: sum of weights of the inputs that are incorrectly classified
    """
    rowLen = len(input[0]) - 1
    correct = 0
    wrong = 0
    sumWeightsError = 0
    index = 0
    for eachRow in input:
        if eachRow[attributeNo] == True and eachRow[rowLen] == "en":
            correct = correct + 1
        elif eachRow[attributeNo] == False and eachRow[rowLen] == "nl":
            correct = correct + 1
        else:
            wrong = wrong + 1
            sumWeightsError = sumWeightsError + weights[index]
        index = index + 1
    return sumWeightsError


def normalize(weights,sumWeights):
    """
    Normalising the weights
    :param weights: list of weights to normalise
    :param sumWeights: sum of all the weights before normalization
    :return: normalised weights list
    """
    for index in range(len(weights)):
        weights[index] = weights[index]/sumWeights
    return weights


def calculatePN(input):
    """
    Calculates the number of en and nl entries in input
    :param input: list to be traversed
    :return: count of en and nl examples
    """
    p = 0
    n = 0
    rowLen = len(input[0]) - 1
    for eachRow in input:
        if eachRow[rowLen] == "en":
            p = p + 1
        else:
            n = n + 1
    return p, n


def answer(child, input):
    """
    Evaluates if the leaf node should represent en or nl.
    :param child: list to be evaluated.
    :param input: list to be evaluated if child is empty.
    :return: 1 for en or -1 for nl
    """
    if len(child) == 0:
        countEn, countNl = calculatePN(input)
        if countEn == len(input):
            return 1
        elif countNl == len(input):
            return -1
        elif countEn >= countNl:
            return 1
        elif countEn < countNl:
            return -1
    countEn, countNl = calculatePN(child)
    if countEn == len(child):
        return 1
    elif countNl == len(child):
        return -1
    elif countEn >= countNl:
        return 1
    elif countEn < countNl:
        return -1


def make(input, weights):
    """
    Creates the adaboost model
    :param input: list of training examples with True/False feature values
    :param weights: list of weights
    :return: attributeNo, hypothesis weight, list of weights,
    prediction of left node of tree, prediction for right node of tree
    """
    p, n = calculatePN(input)
    attributeNo, left, right, ingoGain = pickAttribute(input, p, n)
    totalError = calError(input, weights, attributeNo)
    rowLen = len(input[0]) - 1
    index = 0
    sumWeights = 0
    update = totalError/(1-totalError)
    if totalError == 0:
        hw = 1
    else:
        hw = math.log((1 - totalError) / totalError)
    for eachRow in input:
        if eachRow[attributeNo] == "True" and eachRow[rowLen] == "en":
            weights[index] = weights[index] * update
        elif eachRow[attributeNo] == "False" and eachRow[rowLen] == "nl":
            weights[index] = weights[index] * update
        sumWeights = sumWeights + weights[index]
        index = index + 1
    weights = normalize(weights, sumWeights)
    return attributeNo, hw, weights, answer(left, input), answer(right, input)


def weightsInit(input):
    """
    Initialising weights list with 1/N value
    :param input:
    :return:
    """
    weights = list()
    N = len(input)
    for i in range(len(input)):
        weights.append(1/N)
    return weights


def update_input(input, weights):
    """
    Updating the input examples list for next hypothesis
    :param input: list of examples
    :param weights: list of weights
    :return: new list of examples where some might be repeated
    """
    new_input = list()
    for _ in range(len(input)):
        randNumber = random.random()
        sum = 0
        prev = 0
        for i in range(len(input)):
            sum = sum + weights[i]
            if randNumber >= prev and randNumber < sum:
                new_input.append(input[i])
                break
            prev = sum
    return new_input


def adaboost_train(input, outputFileName):
    """
    Creating the adaboost model and saving it in file
    :param input: list of input examples
    :param outputFileName: file to save the model to
    :return: None
    """
    weights = weightsInit(input)
    h = list()
    for k in range(15):
        attrNo, hw, weights, true, false = make(input, weights)
        h.append((attrNo, hw, true, false))
        input = update_input(input, weights)
        weights = weightsInit(input)
    file = open(outputFileName, "wb")
    pickle.dump(h, file)
    print("Model created!")


def check_features(list_obj, line):
    """
    Checking each line from the test file and computing it's H(X)
    :param list_obj: adaboost object - list containing attribute number, hypothesis weight,
    value for left node, value for right node
    :param line: line to be evaluated
    :return: en or nl
    """
    sum = 0
    for k in range(len(list_obj)):
        if list_obj[k][0] == 0:
            result = lab2.presence_common_dutch_words(line)
        elif list_obj[k][0] == 1:
            result = lab2.presence_common_english_words(line)
        elif list_obj[k][0] == 2:
            result = lab2.consecutive_repeated_vowels(line)
        elif list_obj[k][0] == 3:
            result = lab2.consecutive_repeated_letters(line)
        elif list_obj[k][0] == 4:
            result = lab2.countij(line)
        elif list_obj[k][0] == 5:
            result = lab2. avg_word_length(line)
        if result == True:
            sum = sum + (list_obj[k][1] * list_obj[k][2])
        else:
            sum = sum + (list_obj[k][1] * list_obj[k][3])
    if sum >= 0:
        return "en"
    else:
        return "nl"


def adaboost_predict(list_obj, testFileName):
    """
    Evaluates each line in the test file and writes the output in the file
    :param list_obj: list object of trained adaboost model
    :param testFileName: file containing test sentences
    :return: None
    """
    ans = ""
    with open(testFileName, encoding='utf-8') as file:
        for line in file:
            ans = ans + check_features(list_obj, line) + "\n"
    f = open("ada_answer.out", "w")
    f.write(ans)

