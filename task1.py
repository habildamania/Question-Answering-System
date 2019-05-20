import spacy
import nltk
from nltk import tokenize
from nltk.tokenize import RegexpTokenizer
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
import glob
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import defaultdict
from nltk.tag import pos_tag
from nltk import Tree

def Process_Articles(file_string):
	corpora = []
	sentences = []
	sentences_document = []
	content = open(file_string,'r', encoding='utf-8-sig').read()
	corpora.append(tokenize.sent_tokenize(content))
	#print (corpora)   #Sentence Level Tokenization.
	for sublist in corpora:
		for item in sublist:
			sentences.append(item)
		
	#print (sentences)
	return sentences
	#print(sentences)

	
def Tokenize_Sentences(sentences):
	tokens = []
	tokenizer = RegexpTokenizer(r'\w+')
	#print(sentences)
	for sentence in sentences:
		tokens.append(tokenizer.tokenize(sentence))
	#print(tokens)		
	#return tokens	
	return tokens


def Lemmatize_Sentences(sentences, tokens):
	lemmas =[]
	lemmaspa = []
	lemmatizer = WordNetLemmatizer()
	nlp = spacy.load("en_core_web_sm")
	#print (sentences[0])
	for sentence in sentences:
		newline = nlp(sentence)
		for word in newline:
			lemmaspa.append(word.lemma_)
	#lemmas = []
	for token_list in tokens:
		temp = []
		for i in range(len(token_list)):
			temp.append(lemmatizer.lemmatize(token_list[i]))
		lemmas.append(temp)
	return lemmas, lemmaspa

def POS_Tag_Words(lemmatized_words):
	tags = []
	for lemma_list in lemmatized_words:
		tags.append(nltk.pos_tag(lemma_list))
	return tags

def Wordnet_Features(lemmatized_words):
	syno = {}
	synonyms = {}
	hab = []
	kan = []
	dam = []
	rawa = []
	question = ['create']
	for lem_list in lemmatized_words:
		for word in lem_list:
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
					#***********MERONYMS********************
					for i in range(0,len(wordnet.synsets(word))):
						lmn = wordnet.synset(wordnet.synsets(word)[i].name()).part_meronyms()
						for k in range(len(lmn)):
							dam.append(lmn[k].lemma_names())
					flat_list3 = [item for sublist in dam for item in sublist]
					dam.clear()
					#print(flat_list3)
					for mero in flat_list3: 
						syno[word].append(mero)
					flat_list3.clear()
					#***********HOLONYMS********************
					for i in range(0,len(wordnet.synsets(word))):
						pqr = wordnet.synset(wordnet.synsets(word)[i].name()).part_holonyms()
						for o in range(len(pqr)):
							rawa.append(pqr[o].lemma_names())
					flat_list4 = [item for sublist in rawa for item in sublist]
					rawa.clear()
					#print(flat_list4)
					for holo in flat_list4: 
						syno[word].append(holo)
					flat_list4.clear()

	for key, value in syno.items():
		syno[key] = set(value)
	#print(syno)
	return syno    
#**************HELPER FUNCTIONS FOR dependency_parsing() ***************************	

def tok_format(tok):
	return "_".join([tok.orth_, tok.tag_, tok.dep_])

def to_nltk_tree(node):
	if node.n_lefts + node.n_rights > 0:
		return Tree(tok_format(node), [to_nltk_tree(child) for child in node.children])
	else:
		return tok_format(node)

#***********************************************************************************

def dependency_parsing(sentences):
	tree_list = []
	en_nlp = spacy.load('en')
	command = "Who killed Abraham Lincoln"
	for sentence in sentences:
		en_doc = en_nlp(u'' + sentence) 
		tree = [to_nltk_tree(sent.root) for sent in en_doc.sents]
		tree_list.append(tree)
	return tree_list

def Store_in_file(output_file_string):
	f= open(output_file_string,"w+")
	f.write("****************Output for Tokenization*******************\n")
	for item in Extracted_tokens:
		f.write("%s\n" % item)
	f.write("\n\n")
	f.write("****************Output for Lemmatization*******************\n")
	for item in Extracted_lemmas_spacy:
		f.write("%s,\t" % item)
	f.write("\n\n")
	f.write("****************Output for POS Tagging*******************\n")
	for item in Extracted_tags:
		f.write("%s,\t" % item)
	f.write("\n\n")
	f.write("****Output for Wordnet Features (Synonyms, Hyponyms, Hypernyms, Meronyms, Holonyms)****\n")
	f.write("%s,\t" % Extracted_features)
	f.write("\n\n")
	f.write("****************Output for Dependency Parse*******************\n")
	for item in Extracted_dependency_parse:
		f.write("%s,\n\n" % item)
	f.close()
	return "Thank you"


if __name__ == "__main__":
	Extracted_sentences = Process_Articles("WikipediaArticles/ElonMusk.txt")
	Extracted_tokens = Tokenize_Sentences(Extracted_sentences)
	Extracted_lemmas, Extracted_lemmas_spacy = Lemmatize_Sentences(Extracted_sentences, Extracted_tokens)
	Extracted_tags = POS_Tag_Words(Extracted_lemmas)
	Extracted_features = Wordnet_Features(Extracted_lemmas)
	Extracted_dependency_parse = dependency_parsing(Extracted_sentences)
	success_message = Store_in_file("task1.txt")
	print(success_message)
	#print(Extracted_dependency_parse)
	
	

