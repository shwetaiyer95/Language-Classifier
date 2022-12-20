import sys
import decision_tree
import adaboost
import pickle


def read_examples(exampleFileName):
    """
    Reads the training examples file and creates a 2-D list
    :param exampleFileName: file containing the examples
    :return: 2D list - a row represents an example from the file.
    Each row has 7 columns, 1 for each of the 6 features and one for the output en/nl
    """
    input = list()
    with open(exampleFileName, encoding='utf-8') as file:
        for eachLine in file:
            input.append(features(eachLine.lower()))
    return input


def presence_common_dutch_words(line):
    """
    Checks if the line contains the commonly used words in dutch
    :param line: line to be evaluated
    :return: True, if no commonly used dutch words are present. False, otherwise
    """
    commonDutch = {"het", "de", "dat", "dit", "een", "zo", "waar", "hoe", "waarom", "je", "ik", "en"}
    for word in line.split():
        if word in commonDutch:
            return False
    return True


def presence_common_english_words(line):
    """
    Checks if the line contains the commonly used words in english
    :param line: line to be evaluated
    :return: True, if commonly used english words are present. False, otherwise.
    """
    commonEnglish = {"the", "that", "this", "a", "so", "where", "how", "why", "you", "i", "and"}
    for word in line.split():
        if word in commonEnglish:
            return True
    return False


def consecutive_repeated_vowels(line):
    """
    Checks if same vowels appear consecutively.
    If they appear > 1 times = dutch
    :param line: line to be evaluated
    :return: False, if same vowels appear consecutively greater than one times.
    """
    count = 0
    for word in line.split():
        if word.find("aa") != -1 or word.find("ee") != -1 or \
                word.find("ii") != -1 or word.find("oo") != -1 or word.find("uu") != -1:
            count += 1
    if count > 1:
        return False
    else:
        return True


def consecutive_repeated_letters(line):
    """
    Checks if the same letters appear consecutively.
    If this happens > 2 times = dutch
    :param line: line to be evaluated
    :return: False, if we get consecutive repeated letters > 2 times. True otherwise.
    """
    alphabet = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
                "v", "w", "x", "y", "z"}
    count = 0
    for word in line.split():
        for letter in alphabet:
            str = letter + letter
            if word.find(str) != -1:
                count += 1
    if count > 2:
        return False
    else:
        return True


def countij(line):
    """
    Checks for presence of ij in any word.
    If it appears > 1 times = dutch (In dutch, Y is sometimes written as ij)
    :param line: line to be evaluated
    :return: False, if count is greater than 1. True, oherwise
    """
    count = 0
    for word in line.split():
        if word.find("ij") != -1:
            count += 1
    if count > 1:
        return False
    else:
        return True


def avg_word_length(line):
    """
    Calculates the average word length of the line
    :param line: line to be evaluated
    :return: False if avg length is greater than 5. True, otherwise.
    """
    sum = 0
    count = 0
    for word in line.split():
        sum = sum + len(word)
        count += 1
    avg = sum/count
    if avg > 5:
        return False
    else:
        return True


def features(line):
    """
    Processes the line to get the feature value for it.
    :param line: line to be processed
    :return: list containing true/false values from ecah of the features and en/nl
    """
    y = line[:2]
    line = line[3:]
    to_return = list()
    # True for english and False for dutch
    # feature 0: presence of commonly used dutch words
    to_return.append(presence_common_dutch_words(line))
    # feature 1: presence of commonly used english words
    to_return.append(presence_common_english_words(line))
    # feature 2: same vowels appearing consecutively > 1 times = dutch
    to_return.append(consecutive_repeated_vowels(line))
    # feature 3: same letters appearing consecutively > 2 times = dutch
    to_return.append(consecutive_repeated_letters(line))
    # feature 4: presence of ij > 1 times = dutch (In dutch, Y is sometimes written as ij)
    to_return.append(countij(line))
    # feature 5: Average word length > 5 = dutch
    to_return.append(avg_word_length(line))
    to_return.append(y)
    return to_return


def prediction(modelFileName, testFileName):
    """
    Calls the prediction function of decision tree or adaboost depending on the object
    stored in the model.
    :param modelFileName: file where the model is stored
    :param testFileName: test examples to test the model against
    :return: None
    """
    file = open(modelFileName, 'rb')
    obj = pickle.load(file)
    if isinstance(obj, decision_tree.Node):
        decision_tree.tree_predict(obj, testFileName)
    else:
        adaboost.adaboost_predict(obj, testFileName)


def main():
    if len(sys.argv) != 1:
        type = sys.argv[1]
        if type == "train":
            if len(sys.argv) == 5:
                if sys.argv[4] == "dt":
                    input_list = read_examples(sys.argv[2])
                    decision_tree.tree_train(input_list, sys.argv[3])
                else:
                    input_list = read_examples(sys.argv[2])
                    adaboost.adaboost_train(input_list, sys.argv[3])
            else:
                print("Usage: train <examples> <hypothesisOut> <learning-type>")
        elif type == "predict":
            if len(sys.argv) == 4:
                prediction(sys.argv[2], sys.argv[3])
            else:
                print("Usage: predict <hypothesis> <file>")
        else:
            print("Usage: train <examples> <hypothesisOut> <learning-type>\npredict <hypothesis> <file>")
    else:
        print("Usage: train <examples> <hypothesisOut> <learning-type>\npredict <hypothesis> <file>")
        type = input("train or predict:")
        if type == "train":
            learning = input("What type of learning algorithm (dt or ada):")
            if learning == "dt":
                example = input("Enter Example file Name:")
                output = input("Enter file name to write the model to:")
                input_list = read_examples(example)
                decision_tree.tree_train(input_list, output)
            else:
                example = input("Enter Example file name:")
                output = input("Enter file name to write the model to:")
                input_list = read_examples(example)
                adaboost.adaboost_train(input_list, output)
        elif type == "predict":
            model = input("Enter file name that contains a trained model: ")
            testfile = input("Enter file name containing 15 word sentences: ")
            prediction(model, testfile)


if __name__ == '__main__':
    main()
