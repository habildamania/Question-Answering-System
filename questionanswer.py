from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import glob
import nltk
from nltk import tokenize
import sys
import numpy as np

#get all filenames
#print (filenames)

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
			corpora.append(content)
		return corpora

	#make question as your last document and calculate tf-idf vectors
	def tf_idf(self,ques):
		corpora = self.readfiles()
		corpora.append(ques)
		vectorizer = TfidfVectorizer()
		tf_idfmatrix = vectorizer.fit_transform(corpora)
		np.set_printoptions(threshold=sys.maxsize)
		print(tf_idfmatrix.shape)
		#vector = vector.toarray()
		return tf_idfmatrix

	#find cosine similarity of question with documents
	def cosine_sim(self, vector):
		cos_array = cosine_similarity(vector[-1:],vector)
		print(cos_array)
		return cos_array

	#return top k documents for the question
	def get_top_k(self,cos_array, k):
		#return top 3 documents
		flat = cos_array.flatten()
		ind = np.argpartition(flat,-k)[-k:]
		ind = ind[np.argsort(-flat[ind])]
		ind = ind[1:]
		#print(self.filenames[ind[1]],self.filenames[ind[2]],self.filenames[ind[3]])
		return ind

def main():
	question = input("Enter your question : ") 
	#print(question)
	ob = QuestionAnswerModule() 
	vector = ob.tf_idf(question)
	cos_array = ob.cosine_sim(vector)
	top_indices = ob.get_top_k(cos_array, 4)				#indices of top 4 documents
	print(top_indices)

if __name__ == "__main__":
    main()