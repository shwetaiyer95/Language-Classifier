# Language-Classifier
Language classification using Decision Tree and Adaboost

### Information about the python code files:
The main python file is classifier.py

It calls the respective methods from adaboost.py and decision_tree.py

The classifier.py file takes input in 2 ways, either through command line arguments or through the console.

## Features:
There are a total of 6 features.
1. Existence of commonly used Dutch words:<br/>
I selected some common terms that would be used in all the languages to say something. Like the words the, a, where, etc in Dutch. So, if any of the sentences contains one of these words, there is a high possibility of the sentence being Dutch.
2. Existence of commonly used English words:<br/>
Most English sentences use the, a, who, where, etc. So, this feature checks if these words are present in the sentence. If they are, then there is a high possibility of the sentence being English.
3. Count of times the same vowels appear consecutively:<br/>
I noticed that Dutch words often use “aa”, “ee”, “ii”, “oo”, “uu” together. For example, een, stopplaats. This is not that commonly observed in English words. So, this has been added as a feature. To consider the fact that there are English words that use consecutive vowels, an additional condition has been added. The count of this appearance has to be greater than 1. If such an occurrence happens only once in a sentence, then there is a 50-50 chance of it being an English sentence. But if it happens more than once then there is a high possibility of the sentence being Dutch.
4. Count of times the same letters appear consecutively:<br/>
To increase the chances of correct prediction, I added the feature to count the number of times any letter appears twice consecutively. For instance, “appear” or “allerlei”. As you can see, this is common in English words as well. But the number of times a letter is repeated consecutively in Dutch is a lot. So, if this happens more than two times in a given sentence, it is highly possible that the sentence is Dutch.
5. Presence of “ij”:<br/>
In Dutch, Y is sometimes written as ij. So, the number of times ij will appear together in a word in Dutch is higher than in English. Therefore, I calculate the count of times ij appears in a word throughout the sentence. If more than 1 word in the sentence contains ij, then the chances of the sentence being Dutch are high.
6. Average word length:<br/>
I observed that Dutch words are generally long. Even “I” in Dutch is written as “ik”. So, I compute the average word length of the sentence and if the average is greater than 5, then it is highly possible that the sentence is Dutch.

## Decision Tree Learning:
The decision tree attributes at each level are selected based on information gain. Whichever attribute provides the maximum information gain for the current set of examples is chosen. Then the examples are divided on the basis of this attribute. Looking at the examples and attributes, the max depth was chosen as 6. There are 3 base cases to determine which language a sentence is written in. If the examples are all of en or nl then, stop. If we run out of attributes and the training set is not all in en or nl, then whichever is in majority is chosen. If there are no examples left, then majority of the parent is chosen.

For the testing, around 10 examples were taken and they are run through the decision tree. 100% accuracy was observed for these examples.

## Adaboost Learning:
In adaboost, the next hypothesis for the decision stump is selected by calculating the information gain and picking the one which provides the maximum information gain. After each iteration, the weight of the correctly classified examples decreases and the weight of the incorrectly identified examples increases. At the end of each iteration, an hi and its corresponding weight is obtained.

During the testing, each hypothesis is evaluated in the order that they were discovered during training. At the end of each evaluation, we get 1 (for en) or -1 (for nl). If at the end of a sentence evaluation, H(X) is negative then the sentence is Dutch, otherwise it is English.
