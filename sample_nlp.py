 
import math
from collections import Counter

def get_cosine(vec1, vec2):
    common = set(vec1.keys()) & set(vec2.keys())
    print(common)
    for x in common:
        print(x)
    numerator = sum([vec1[x] * vec2[x] for x in common])
    print('numerator:',numerator)

    sum1 = sum([vec1[x]**2 for x in vec1.keys()]) 
    sum2 = sum([vec2[x]**2 for x in vec2.keys()]) 
    print(sum1)
    print(sum2)
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    print('denominator:',denominator)
   
    if not denominator:
        return 0.0 
    else:
        return float(numerator) / denominator

def text_to_vector(text): 
    words = text.split() 
    return Counter(words)

text1 = 'How are you' 
text2 = 'where you are'

vector1 = text_to_vector(text1) 
print(vector1)
vector2 = text_to_vector(text2) 
cosine = get_cosine(vector1, vector2)

def text_to_vector(text): 
    words = text.split() 
    return Counter(words)

text1 = 'how are you' 
text2 = 'where you are'
vector1 = text_to_vector(text1)
vector2 = text_to_vector(text2)
common = set(vector1.keys()) & set(vector2.keys())
print(len(common))
dissimilar=set(vector1.keys()) ^ set(vector2.keys())
print(dissimilar)
def levenshtein(s1,s2): 
    if len(s1) > len(s2):
        s1,s2 = s2,s1 
    print(s1,s2)
    distances = range(len(s1) + 1) 
    print('s1:',distances)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        print(newDistances)
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1]) 
                print(newDistances)
            else:
                 newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1]))) 
        distances = newDistances
    return distances[-1]
    
levenshtein("analyze","analyse")
