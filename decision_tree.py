import math
import pickle
import lab2


class Node:
    """
    feature: attribute number the node represents
    left: left node of the current node
    right: right node of the current node
    lvalue: None if left node exists. If left is None, then it has value en/nl
    rvalue: None if right node exists. If right is None, then it has value en/nl
    """
    __slots__ = "left", "right", "feature", "rvalue", "lvalue"

    def __init__(self, feature = None):
        self.feature = feature
        self.lvalue = None
        self.rvalue = None


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


def pickAttribute(input, p, n, attributeList):
    """
    Determines with attribute provides maximum information gain
    :param input: input list containing examples with feature True/False values
    :param p: number of examples in english
    :param n: number of examples in dutch
    :param attributeList: list containing the attributes list
    :return: attribute number that provides maximum information gain,
    list of it's true values. list of it's false values, infomration gain value
    """
    maxInfoGain = -1
    maxAttributeNo = attributeList[0]
    maxLeft = list()
    maxRight = list()
    for attributeNo in attributeList:
        infoGain, left, right = informationGain(attributeNo, input, p, n)
        if infoGain > maxInfoGain:
            maxInfoGain = infoGain
            maxAttributeNo = attributeNo
            maxLeft = left
            maxRight = right
    return maxAttributeNo, maxLeft, maxRight, maxInfoGain


def answer(child, input):
    """
    Evaluates if the leaf node should represent en or nl.
    :param child: list to be evaluated.
    :param input: list to be evaluated if child is empty.
    :return: en or nl
    """
    if len(child) == 0:
        countEn, countNl = calculatePN(input)
        if countEn == len(input):
            return "en"
        elif countNl == len(input):
            return "nl"
        elif countEn >= countNl:
            return "en"
        elif countEn < countNl:
            return "nl"
    countEn, countNl = calculatePN(child)
    if countEn == len(child):
        return "en"
    elif countNl == len(child):
        return "nl"
    elif countEn >= countNl:
        return "en"
    elif countEn < countNl:
        return "nl"


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


def make(input, depth, attributeList, node, parent, side):
    """
    Creates the decision tree.
    :param input: list containing the training examples
    :param depth: number of attributes/features
    :param attributeList: list containing the attributes
    :param node: node currently being processed
    :param parent: parent of node
    :param side: indicates whether node is the left or right child of parent node
    :return: None
    """
    if depth > 0 and len(input) > 0:
        p, n = calculatePN(input)
        if p == 0 or n == 0:
            ans = ""
            if p == 0:
                ans = "nl"
            elif n == 0:
                ans = "en"
            if side == "left":
                parent.left = None
                parent.lvalue = ans
            else:
                parent.right = None
                parent.rvalue = ans
        else:
            attributeNo, left, right, infoGain = pickAttribute(input, p, n, attributeList)
            if len(left) == 0 or len(right) == 0: #info gain is 0
                if side == "left":
                    parent.left = None
                    parent.lvalue = answer(right, input)
                else:
                    parent.right = None
                    parent.rvalue = answer(left, input)
            else:
                attributeList.remove(attributeNo)
                node.feature = attributeNo
                node.left = Node()
                node.right = Node()
                make(left, depth - 1, attributeList.copy(), node.left, node, "left")
                make(right, depth - 1, attributeList.copy(), node.right, node, "right")
    else:
        if side == "left":
            parent.left = None
            parent.lvalue = answer(input, input)
        else:
            parent.right = None
            parent.rvalue = answer(input, input)


def format_tree(parent, root):
    if root!=None:
        if root.feature == None:
            if parent.left.feature == None:
                parent.left = None
            if parent.right.feature == None:
                parent.right = None
        else:
            format_tree(root, root.left)
            format_tree(root, root.right)


def save_root(root, outputFileName):
    """
    Saving the tree/model in a file
    :param root: object to be saved
    :param outputFileName: name of the output file
    :return: None
    """
    file = open(outputFileName,"wb")
    pickle.dump(root,file)


def tree_train(input, outputFileName):
    """
    Training - creates the decision tree model
    :param input: list of examples given
    :param outputFileName: file to save the model in
    :return: None
    """
    attributeList = list(range(len(input[0]) - 1))
    root = Node()
    make(input, 6, attributeList, root, None, None)
    save_root(root, outputFileName)
    format_tree(None, root)
    print("Model created!")


def compute_result(root, line):
    """
    Computes through the model and predicts if the line will output en/nl
    :param root: root of the decision tree
    :param line: line to be evaluated
    :return: en/nl
    """
    if root.feature == 0:
        result = lab2.presence_common_dutch_words(line)
    elif root.feature == 1:
        result = lab2.presence_common_english_words(line)
    elif root.feature == 2:
        result = lab2.consecutive_repeated_vowels(line)
    elif root.feature == 3:
        result = lab2.consecutive_repeated_letters(line)
    elif root.feature == 4:
        result = lab2.countij(line)
    elif root.feature == 5:
        result = lab2. avg_word_length(line)
    if result == True:
        if root.left != None:
            return compute_result(root.left, line)
        else:
            return root.lvalue
    else:
        if root.right != None:
            return compute_result(root.right, line)
        else:
            return root.rvalue


def tree_predict(root, fileName):
    """
    Evaluates each line in the test file and writes the output in the file
    :param root: root of the decision tree
    :param fileName: file containing sentences to be tested
    :return: None
    """
    ans = ""
    with open(fileName, encoding='utf-8') as file:
        for line in file:
            ans = ans + compute_result(root, line) + "\n"
    f = open("dt_answer.out", "w")
    f.write(ans)
