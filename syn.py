#TODO - Should have Tenses.
import nltk
from nltk.corpus import wordnet
from collections import defaultdict
from nltk.tag import pos_tag
syno = {}
synonyms = {}
hab = []
kan = []
flat_list2 = []
question = ['kill','flower']
def extract_syn(question):
    for word in question:
        wordp = nltk.word_tokenize(word)
        tagged_senta = pos_tag(wordp)
        for wo, pos in tagged_senta:
            if pos != 'NNP':
                #*************SYNONYMS*******************
                for syn in wordnet.synsets(word):
                    #print(syn)
                    for l in syn.lemmas():
                        #print(l)
                        if word not in syno:
                            syno[word] = [l.name(),word]
                        else:
                            syno[word].append(l.name())
                            
                #***********HYPONYMS********************
                for i in range(0,len(wordnet.synsets(word))):
                    abc = wordnet.synset(wordnet.synsets(word)[i].name()).hyponyms()
                    for j in range(len(abc)):
                        hab.append(abc[j].lemma_names())
                flat_list2 = [item for sublist in hab for item in sublist]
                hab.clear()
                #print(flat_list2)
                for hypo in flat_list2: 
                    syno[word].append(hypo)
                flat_list2.clear()
                
                #***********HYPERNYMS********************  
                for i in range(0,len(wordnet.synsets(word))): 
                    xyz = wordnet.synset(wordnet.synsets(word)[i].name()).hypernyms() 
                    for h in range(len(xyz)): 
                        kan.append(xyz[h].lemma_names()) 
                flat_list1 = [item for sublist in kan for item in sublist] 
                kan.clear()
                for hyper in flat_list1: 
                    syno[word].append(hyper) 
                flat_list1.clear()
    for key, value in syno.items():
        syno[key] = set(value)
    return syno

synonyms = extract_syn(question)
print(synonyms)