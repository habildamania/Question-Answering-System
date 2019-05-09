from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import spacy
import glob
import nltk
from nltk import tokenize
import sys
import numpy as np
from nltk.corpus import wordnet
from collections import defaultdict
from nltk.tag import pos_tag
import string
import json
import pprint


#get all filenames
#print (filenames)

nlp = spacy.load('en_core_web_sm')
class QuestionAnswerModule:
	filenames = glob.glob("WikipediaArticles/*.txt")

	#read file contents and store it in format ["content1", "content2"]
	def readfiles(self):
		corpora = []
		content= ""
		for fname in self.filenames:
			#print(fname)
			#file = open(fname,'r',encoding='utf-8-sig')
			#for line in file:
			#	content += line.rstrip('\n')
			content = open(fname,'r',encoding='utf-8-sig').read()
			#print (content)
			corpora.append(content)
		return corpora

	#make question as your last document and calculate tf-idf vectors
	def tf_idf(self,corpora):
		vectorizer = TfidfVectorizer()
		tf_idfmatrix = vectorizer.fit_transform(corpora)
		np.set_printoptions(threshold=sys.maxsize)
		#print(tf_idfmatrix.shape)
		#vector = vector.toarray()
		return tf_idfmatrix

	#find cosine similarity of question with documents
	def cosine_sim(self, vector):
		cos_array = cosine_similarity(vector[-1:],vector)
		#print(cos_array)
		return cos_array

	#return top k documents for the question
	def get_top_k(self,cos_array, k):
		#return top 3 documents
		flat = cos_array.flatten()
		ind = np.argpartition(flat,-k)[-k:]
		ind = ind[np.argsort(-flat[ind])]
		#print (ind)
		ind = ind[1:]
		#print(self.filenames[ind[1]],self.filenames[ind[2]],self.filenames[ind[3]])
		return ind

	#do dependency parsing on the question and store words except stop words
	# in the form [(word, dependecy parse tag, pos tag)]
	def dep_parse_ques(self,question,ques_types):
		search_list = []
		doc = nlp(question)
		root = ""
		nsub = ""
		for token in doc:
			if token.text.lower() not in stopwords.words('english') and token.text.lower() not in ques_types:
				
				#print(token.text, token.pos_, token.dep_)
				if token.dep_ in ("ROOT","acl","advcl","amod","advmod","compound","csubj","nsubjpass",
				 	"nn","attr","dobj","npmod","nsubj","pobj","acomp"):
					#print(token.text, token.pos_, token.dep_)
					#if token.dep_ in ("ROOT") or token.pos_ in ("VERB"):
					search_list.append([token.text.lower(),token.dep_,token.pos_])
					if(token.dep_ == "ROOT"):
						root = token.text
					if(token.dep_ == "nsubj"):
						nsub = token.text
		if root == "":
			root = nsub
		return root,search_list

	#extract sentences from the top 3 documents based on root and its synonyms
	def extract_sentences_root(self, ques, search_dict,top_indices):
		sentences_set = []
		ques_v = []
		#print (ques)
		for word in ques.split(" "):
			if word:
				ques_v.append(word)
		#print(ques_v)
		for t in top_indices:
			file = self.filenames[t]
			with open(file,'r',encoding='utf-8-sig') as fp:
				content = fp.read()
				content = tokenize.sent_tokenize(content)
				for line in content:
					#print(line)
					#if any of the words in search list are in the line just read then
					newline = nlp(line)
					line_v = []
					for word in newline:
						line_v.append(word.lemma_)
					token = []
					for key, value in search_dict.items():
						key = nlp(key)
						for t in key:
							token.append(t)
							token.append(t.lemma_)
						if any(str(word) in line_v for word in token):
							#print ("found ", line)
							sentences_set.append(line)
					for key, value in search_dict.items():
						for val in value:
							if val in line_v and any(word in line_v for word in ques_v):
								#print("word:",val," ",line)
								sentences_set.append(line)
					#sentences_set.append(line)
		#sentences_set = set(sentences_set)
		return sentences_set

	def check_ques_type(self, question,sorted_overlapped):
		#print('Entered', question,sorted_overlapped[0][0])
		if any(word in question.lower() for word in ['who','whom']):
			filtered = self.extract_sent_named_entity('who', sorted_overlapped)
		elif 'when' in question.lower():
			filtered = self.extract_sent_named_entity('when', sorted_overlapped)
		else:
			filtered = self.extract_sent_named_entity('where', sorted_overlapped)
		return filtered

	def extract_sent_named_entity(self, question, sorted_overlapped):
		#print('Entered entity function', question, sorted_overlapped[0][0])
		flag = 0
		ent_type = []
		nlp = spacy.load("en_core_web_sm")
		if question == 'who':
			print("Entered who type\n")
			for sent in sorted_overlapped:
				#print(sent[0])
				doc = nlp(sent[0])
				for ent in doc.ents:
					if (ent.label_ == "PERSON") or (ent.label_== "ORG"):   #Change here based on desired entity
						ent_type.append((sent[0],ent.text))
						flag = 1
					#print(ent.text, ent.start_char, ent.end_char, ent.label_)
					if (flag ==1):
						break

		elif question == 'where':
			print("Entered where type\n")
			for sent in sorted_overlapped:
				#print(sent[0])
				doc = nlp(sent[0])
				for ent in doc.ents:
					if (ent.label_ == "LOC") or (ent.label_== "FAC") or (ent.label_ == "GPE"):      #Change here based on desired entity
						ent_type.append((sent[0],ent.text))
						flag = 1
					#print(ent.text, ent.start_char, ent.end_char, ent.label_)
					if (flag ==1):
						break
					
		else: #type == when
			print("Entered when type\n")
			for sent in sorted_overlapped:
				#print(sent[0])
				doc = nlp(sent[0])
				for ent in doc.ents:
					if (ent.label_ == "DATE") or (ent.label_== "TIME"):     #Change here based on desired entity
						ent_type.append((sent[0],ent.text))
						flag = 1
					#print(ent.text, ent.start_char, ent.end_char, ent.label_)
					if (flag ==1):
						break
		return ent_type

	# find synonyms of the words in the list
	def extract_syn(self, search_list):
		syno = {}
		hab = []
		kan = []
		flat_list2 = []
		for list_ in search_list:
			wordp = nltk.word_tokenize(list_[0])
			tagged_senta = pos_tag(wordp)
			word = list_[0]
			for wo, pos in tagged_senta:
				if list_[2] != 'PROPN':
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
					'''for i in range(0,len(wordnet.synsets(word))):
						abc = wordnet.synset(wordnet.synsets(word)[i].name()).hyponyms()
						for j in range(len(abc)):
							hab.append(abc[j].lemma_names())
					flat_list2 = [item for sublist in hab for item in sublist]
					hab.clear()
					#print(flat_list2)
					for hypo in flat_list2: 
						syno[word].append(hypo)
					flat_list2.clear()'''
		                
					#***********HYPERNYMS********************  
					'''for i in range(0,len(wordnet.synsets(word))): 
						xyz = wordnet.synset(wordnet.synsets(word)[i].name()).hypernyms() 
						for h in range(len(xyz)): 
							kan.append(xyz[h].lemma_names()) 
					flat_list1 = [item for sublist in kan for item in sublist] 
					kan.clear()
					for hyper in flat_list1: 
						syno[word].append(hyper) 
					flat_list1.clear()'''
		for key, value in syno.items():
			syno[key] = set(value)
		return syno

	def overlap(self, top_indices, ques):
		sentences_set = []
		ques_v = []
		ques = nlp(ques)
		for word in ques:
			if word:
				ques_v.append(word.lemma_)
		#print(ques_v)
		for t in top_indices:
			file = self.filenames[t]
			with open(file,'r',encoding='utf-8-sig') as fp:
				content = fp.read()
				content = tokenize.sent_tokenize(content)
				for line in content:
					newline = nlp(line)
					line_v = []
					for word in newline:
						line_v.append(word.lemma_)
					sentences_set.append((line,len(list(set(ques_v).intersection(set(line_v))))))
		return sentences_set

	def Sort_Tuple(self,tup):  
		tup.sort(key = lambda x: x[1],reverse = True)  
		return tup

	def generateJson(self,question,sentences,documents):
		f= open("jsondata.txt","a+")
		jsonData = dict({})
		jsonData["Question"] = question
		jsonData["answers"] = dict({})
		ansDict = dict()
		for i in range(0,len(sentences)):
			ansDict[str(i+1)] = sentences[i][1]
		jsonData["answers"] = ansDict

		jsonData["sentences"] = dict({})
		sentDict = dict()
		for i in range(0,len(sentences)):
			sentDict[str(i+1)] = sentences[i][0]
		jsonData["sentences"] = sentDict

		jsonData["documents"] = dict({})
		docDict = dict()
		for i in range(0,len(sentences)):
			docDict[str(i+1)] = documents.split("\\")[1]
		jsonData["documents"] = docDict

		jsonData = [jsonData]
		data = json.dumps(jsonData)


		print(jsonData)
		#print("data")
		#print(data)
		f.write(data)
		f.write("\n")
  
	def getfilename(self,index):
		return self.filenames[index[0]]

def main():
	#quesfile = input("Enter your filename : ")
	#with open(quesfile,'r',encoding='utf-8-sig') as fp:
	#question = fp.readline().rstrip()
#	while question:
	#print(question)
	question = input("Enter question: ")
	translator = str.maketrans('', '', string.punctuation)
	question = question.translate(translator)
	ques_types = ['who','whom','when','where']							#types of questions we are handling
	ob = QuestionAnswerModule() 
	doc = nlp(question)
	ques = ""
	for token in doc:
		if token.text.lower() not in stopwords.words('english') and token.text.lower() not in ques_types:
			#if token.pos_ in ("NOUN","PROPN"):
			ques+=token.text+" "
	#print(ques)
	corpora = ob.readfiles()
	corpora.append(ques)
	vector = ob.tf_idf(corpora)						
	cos_array = ob.cosine_sim(vector)
	top_indices = ob.get_top_k(cos_array, 2)							# get indices of top 4 documents
	
	#print(top_indices)
	root, ques_search_list = ob.dep_parse_ques(question,ques_types)		#parse question into word, dependency parse tag
	#print (ques_search_list)
	syn_list = ob.extract_syn(ques_search_list)									# get synonyms of the root verb
	#print(syn_list)
	
	#sentences_set = ob.extract_sentences_root(ques,syn_list,top_indices)		# get filtered sentences on the basis of root and its synonyms in question
	#print (sentences_set)
	#new_ques = sentences_set
	s = ""
	for word in question.split(" "):
		if word.lower() not in ques_types and word.lower() not in stopwords.words('english'):
			s += word + " "
	for key,value in syn_list.items():
		for val in value:
			s+=val + " "
	#print(s)
	overlap_sent = ob.overlap(top_indices, s)
	sorted_overlapped = ob.Sort_Tuple(overlap_sent)[0:20]
	filtered_res = ob.check_ques_type(question,sorted_overlapped)
	print(filtered_res[0:2])
	filename = ob.getfilename(top_indices)
	#ob.generateJson(question,filtered_res[0:2],filename)
	#question = fp.readline()
	'''root = nlp(root)
	filtered_res = []
	nounlist = []
	quesnoun = []
	for t in ques_search_list:
		if t[2] == 'PROPN':
			quesnoun.append(t[0])
	print (quesnoun)
	for sent in sorted_overlapped:
		s = nlp(sent[0])
		roots = []
		ans = []
		for ent in s.ents:
			if ent.label_ in ('DATE','TIME'):
				ans.append(ent.text)
				for token in s:
					if token.dep_ == 'ROOT':
						roots.append(token.lemma_.lower())
					if token.dep_ in ('nsubj','dobj','compound'):
						nounlist.append(token.text.lower())
				#if str(root[0]) in roots or str(root[0].lemma_) in roots:
				for value in syn_list[str(root[0])]:
					if value in roots or str(root[0].lemma_) in roots:
						for v in quesnoun:
							if v in nounlist:
								filtered_res.append((sent[0],ans))
								break
	#filtered_res = set(filtered_res[0])
	print (filtered_res[0:2])'''
	
	'''new_ques.append(s.lower())
	#print (new_ques)
	vector = ob.tf_idf(new_ques)
	cos_array = ob.cosine_sim(vector)
	top_indices = ob.get_top_k(cos_array, 8)
	f = cos_array.flatten()
	for index in top_indices:
	 	print ("index:",index," ","cosine:",f[index]," ",new_ques[index])
	#print (ques_search_list)
	#print(syn_list['killed'])
	#print(sentences_set)
	#for word in search_list:
	#	print (word)'''


if __name__ == "__main__":
    main()