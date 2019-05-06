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
question = "Who are you?"
sentences_set = ["My name is Habil",
				 "I had a chicken dinner", 
				 "You are a good man,Theon", 
				 "Welcome Mr. Habil", 
				 "I was born on January 2, 1997"]
def check_ques_type(question,sentences_set):
	filtered = []
	if any(word in question.lower() for word in ['who','whom']):
		filtered = extract_sent_named_entity('who',sentences_set)
	elif 'when' in question.lower():
		filtered = extract_sent_named_entity('when',sentences_set)
	else:
		filtered = extract_sent_named_entity('where',sentences_set)
	return filtered

def extract_sent_named_entity(self, sentences_set):
	flag = 0
	ent_type = []
	nlp = spacy.load("en_core_web_sm")
	if self == 'who':
		for sent in sentences_set:
			print(sent)
			doc = nlp(sent)
			for ent in doc.ents:
				if ent.label_ == "PERSON":
					ent_type.append(sent)
					flag = 1
				print(ent.text, ent.start_char, ent.end_char, ent.label_)
				if (flag ==1):
					break

	elif self == 'when':
		for sent in sentences_set:
			print(sent)
			doc = nlp(sent)
			for ent in doc.ents:
				if ent.label_ == "PERSON":
					ent_type.append(sent)
					flag = 1
				print(ent.text, ent.start_char, ent.end_char, ent.label_)
				if (flag ==1):
					break
				
	else: #type == where
		for sent in sentences_set:
			print(sent)
			doc = nlp(sent)
			for ent in doc.ents:
				if ent.label_ == "PERSON":
					ent_type.append(sent)
					flag = 1
				print(ent.text, ent.start_char, ent.end_char, ent.label_)
				if (flag ==1):
					break
	return ent_type
entity_type = []
entity_type = check_ques_type(question,sentences_set)
print("Entity Set", entity_type)
