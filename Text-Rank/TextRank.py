
# coding: utf-8

# # Implementation of TextRank
# (Based on: https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)

# The input text is given below

# In[ ]:


text = '''
در نمایشگاه کتاب بندرعباس آثار دونالد ترامپ در حوزه کتاب‌های موفقیت به وفور عرضه می‌شود. این در حالی است که رئیس‌جمهور آمریکا مراکز فرهنگی ما را تهدید می‌کند اما ما کتاب‌های او را تبلیغ می‌کنیم!]

 '''


# ### Cleaning Text Data
# 
# The raw input text is cleaned off non-printable characters (if any) and turned into lower case.
# The processed input text is then tokenized using gensim library functions. 

# In[ ]:


from hazm import *
from textcleaner import clean_text_by_sentences
from textcleaner import clean_text_by_sentences

print(clean_text_by_sentences(text))


# ### POS Tagging For Lemmatization
# 
# Hazm is used for <b>POS tagging</b> the input text so that the words can be lemmatized based on their POS tags.
# 
# Description of POS tags: 
# 
# 
# http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

# In[ ]:


tagger = POSTagger(model='resources/POSTagger.model')
tagged = tagger.tag(word_tokenize(text))

print ("Tokenized Text with POS tags: \n")


# ### Lemmatization
# 
# The tokenized text (mainly the nouns and adjectives) is normalized by <b>lemmatization</b>.
# In lemmatization different grammatical counterparts of a word will be replaced by single
# basic lemma. For example, 'glasses' may be replaced by 'glass'. 
# 
# Details about lemmatization: 
#     
# https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html

# In[ ]:


lemmatizer = Lemmatizer()

adjective_tags = ['JJ','JJR','JJS']

lemmatized_text = []

for word in tagged:
    if word[1] in adjective_tags:
        lemmatized_text.append(str(lemmatizer.lemmatize(word[0],pos="a")))
    else:
        lemmatized_text.append(str(lemmatizer.lemmatize(word[0]))) #default POS = noun
        
print ("Text tokens after lemmatization of adjectives and nouns: \n")
print (lemmatized_text)


# ### POS tagging for Filtering
# 
# The <b>lemmatized text</b> is <b>POS tagged</b> here. The tags will be used for filtering later on.

# In[ ]:


POS_tag = tagger.tag(lemmatized_text)

print ("Lemmatized text with POS tags: \n")
print (POS_tag)


# ### Complete stopword generation
# 
# Even if we remove the aforementioned stopwords, still some extremely common nouns, adjectives or gerunds may
# remain which are very bad candidates for being keywords (or part of it). 
# 
# An external file constituting a long list of stopwords is loaded and all the words are added with the previous
# stopwords to create the final list 'stopwords-plus' which is then converted into a set. 
# 
# (Source of stopwords data: https://www.ranks.nl/stopwords)
# 
# Stopwords-plus constitute the sum total of all stopwords and potential phrase-delimiters. 
# 
# (The contents of this set will be later used to partition the lemmatized text into n-gram phrases. But, for now, I will simply remove the stopwords, and work with a 'bag-of-words' approach. I will be developing the graph using unigram texts as vertices)

# In[ ]:


stopwords = []
stopword_file = open("resources/STOPWORDS.txt", "r")
#Source = https://www.ranks.nl/stopwords

lots_of_stopwords = []

for line in stopword_file.readlines():
    lots_of_stopwords.append(str(line.strip()))

stopwords_plus = []
stopwords_plus = stopwords + lots_of_stopwords
stopwords_plus = set(stopwords_plus)
stopwords_plus
#Stopwords_plus contain total set of all stopwords


# ### Removing Stopwords 
# 
# Removing stopwords from lemmatized_text. 
# Processeced_text condtains the result.

# In[ ]:


processed_text = []
for word in lemmatized_text:
    print(word)
    if word not in stopwords_plus:
        processed_text.append(word)
print (processed_text)


# ## Vocabulary Creation
# 
# Vocabulary will only contain unique words from processed_text.

# In[ ]:


vocabulary = list(set(processed_text))
print (vocabulary)


# ### Building Graph
# 
# TextRank is a graph based model, and thus it requires us to build a graph. Each words in the vocabulary will serve as a vertex for graph. The words will be represented in the vertices by their index in vocabulary list.  
# 
# The weighted_edge matrix contains the information of edge connections among all vertices.
# I am building wieghted undirected edges.
# 
# weighted_edge[i][j] contains the weight of the connecting edge between the word vertex represented by vocabulary index i and the word vertex represented by vocabulary j.
# 
# If weighted_edge[i][j] is zero, it means no edge connection is present between the words represented by index i and j.
# 
# There is a connection between the words (and thus between i and j which represents them) if the words co-occur within a window of a specified 'window_size' in the processed_text.
# 
# The value of the weighted_edge[i][j] is increased by (1/(distance between positions of words currently represented by i and j)) for every connection discovered between the same words in different locations of the text. 
# 
# The covered_coocurrences list (which is contain the list of pairs of absolute positions in processed_text of the words whose coocurrence at that location is already checked) is managed so that the same two words located in the same positions in processed_text are not repetitively counted while sliding the window one text unit at a time.
# 
# The score of all vertices are intialized to one. 
# 
# Self-connections are not considered, so weighted_edge[i][i] will be zero.

# In[ ]:


import numpy as np
import math
vocab_len = len(vocabulary)

weighted_edge = np.zeros((vocab_len,vocab_len),dtype=np.float32)

score = np.zeros((vocab_len),dtype=np.float32)
window_size = 3
covered_coocurrences = []

for i in range(0,vocab_len):
    score[i]=1
    for j in range(0,vocab_len):
        if j==i:
            weighted_edge[i][j]=0
        else:
            for window_start in range(0,(len(processed_text)-window_size)):
                
                window_end = window_start+window_size
                
                window = processed_text[window_start:window_end]
                
                if (vocabulary[i] in window) and (vocabulary[j] in window):
                    
                    index_of_i = window_start + window.index(vocabulary[i])
                    index_of_j = window_start + window.index(vocabulary[j])
                    
                    # index_of_x is the absolute position of the xth term in the window 
                    # (counting from 0) 
                    # in the processed_text
                      
                    if [index_of_i,index_of_j] not in covered_coocurrences:
                        weighted_edge[i][j]+=1/math.fabs(index_of_i-index_of_j)
                        covered_coocurrences.append([index_of_i,index_of_j])


# ### Calculating weighted summation of connections of a vertex
# 
# inout[i] will contain the sum of all the undirected connections\edges associated withe the vertex represented by i.

# In[ ]:


inout = np.zeros((vocab_len),dtype=np.float32)

for i in range(0,vocab_len):
    for j in range(0,vocab_len):
        inout[i]+=weighted_edge[i][j]


# ### Scoring Vertices
# 
# The formula used for scoring a vertex represented by i is:
# 
# score[i] = (1-d) + d x [ Summation(j) ( (weighted_edge[i][j]/inout[j]) x score[j] ) ] where j belongs to the list of vertieces that has a connection with i. 
# 
# d is the damping factor.
# 
# The score is iteratively updated until convergence. 

# In[ ]:


MAX_ITERATIONS = 50
d=0.85
threshold = 0.0001 #convergence threshold

for iter in range(0,MAX_ITERATIONS):
    prev_score = np.copy(score)
    
    for i in range(0,vocab_len):
        
        summation = 0
        for j in range(0,vocab_len):
            if weighted_edge[i][j] != 0:
                summation += (weighted_edge[i][j]/inout[j])*score[j]
                
        score[i] = (1-d) + d*(summation)
    
    if np.sum(np.fabs(prev_score-score)) <= threshold: #convergence condition
        print("Converging at iteration "+str(iter)+"....")
        break


# In[ ]:


for i in range(0,vocab_len):
    print("Score of "+vocabulary[i]+": "+str(score[i]))


# ### Phrase Partiotioning
# 
# Paritioning lemmatized_text into phrases using the stopwords in it as delimeters.
# The phrases are also candidates for keyphrases to be extracted. 

# In[ ]:


phrases = []

phrase = " "
for word in lemmatized_text:
    
    if word in stopwords_plus:
        if phrase!= " ":
            phrases.append(str(phrase).strip().split())
        phrase = " "
    elif word not in stopwords_plus:
        phrase+=str(word)
        phrase+=" "

print("Partitioned Phrases (Candidate Keyphrases): \n")
print(phrases)


# ### Create a list of unique phrases.
# 
# Repeating phrases\keyphrase candidates has no purpose here, anymore. 

# In[ ]:


unique_phrases = []

for phrase in phrases:
    if phrase not in unique_phrases:
        unique_phrases.append(phrase)

print("Unique Phrases (Candidate Keyphrases): \n")
print(unique_phrases)


# ### Thinning the list of candidate-keyphrases.
# 
# Removing single word keyphrases-candidates that are present multi-word alternatives. 

# In[ ]:


for word in vocabulary:
    #print word
    for phrase in unique_phrases:
        if (word in phrase) and ([word] in unique_phrases) and (len(phrase)>1):
            #if len(phrase)>1 then the current phrase is multi-worded.
            #if the word in vocabulary is present in unique_phrases as a single-word-phrase
            # and at the same time present as a word within a multi-worded phrase,
            # then I will remove the single-word-phrase from the list.
            unique_phrases.remove([word])
            
print("Thinned Unique Phrases (Candidate Keyphrases): \n")
print(unique_phrases)    


# ### Scoring Keyphrases
# 
# Scoring the phrases (candidate keyphrases) and building up a list of keyphrases\keywords
# by listing untokenized versions of tokenized phrases\candidate-keyphrases.
# Phrases are scored by adding the score of their members (words\text-units that were ranked by the graph algorithm)
# 

# In[ ]:


phrase_scores = []
keywords = []
for phrase in unique_phrases:
    phrase_score=0
    keyword = ''
    for word in phrase:
        keyword += str(word)
        keyword += " "
        phrase_score+=score[vocabulary.index(word)]
    phrase_scores.append(phrase_score)
    keywords.append(keyword.strip())

i=0
words_with_scores = []
for keyword in keywords:
#     print(words_with_scores)
    tup = (keyword,phrase_scores[i])
    words_with_scores.append(tup)
#     print ("Keyword: '"+str(keyword)+"', Score: "+str(phrase_scores[i]))
    i+=1
 
# print(len(words_with_scores))
# print(word_with_scores)
for i,ws in enumerate(words_with_scores):
    print(f"{i}",ws)


# ### Ranking Keyphrases
# 
# Ranking keyphrases based on their calculated scores. Displaying top keywords_num no. of keyphrases.

# In[ ]:


sorted_index = np.flip(np.argsort(phrase_scores),0)

print(phrase_scores)

keywords_num = 10

print("Keywords:\n")

for i in range(0,keywords_num):
    try:
        print(str(i),str(keywords[sorted_index[i]])+", ", end=' ')
    except:
        continue


# # Input:
# 
# Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types of systems and systems of mixed types.
# 
# # Extracted Keywords:
# 
# * minimal supporting set,  
# * minimal generating set,  
# * minimal set,  
# * linear diophantine equation,  
# * nonstrict inequations,  
# * strict inequations,  
# * system,  
# * linear constraint,  
# * solution,  
# * upper bound, 
# 
