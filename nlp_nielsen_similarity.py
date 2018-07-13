def similarity_sentence(text1,validate):
    ## Imporing required modules
    import pandas as pd
    from nltk.corpus import stopwords
    from keras.preprocessing.sequence import pad_sequences
    from keras.preprocessing.text import Tokenizer
    from nltk import word_tokenize   
    import nltk
    import re
    from collections import Counter
    from stemming.porter2 import stem
    from nltk.util import ngrams


    ## cleaning the sentence by removing the stop words and other than Alphabets

    def Sentence_clean(text):
        sentence_clean= re.sub("[^a-zA-Z]"," ", text)
        sentence_clean = sentence_clean.lower()
        tokens = word_tokenize(sentence_clean)
        stop_words = set(stopwords.words("english"))
        sentence_clean_words = [w for w in tokens if not w in stop_words]
        return ' '.join(sentence_clean_words)


    # # Splitting the sentence by single word

    def unigram_n(text): 
        words = text.split() 
        return Counter(words)


    # # Stemming(root word) the each word

    def to_do_stem(text1):
        s=''
        for word in list(text1.split(" ")):
            s+=' '+(str(stem(word)))
        return s



    # # Splitting the sentence by two words

    def bigram_n(text):
        bigram = ngrams(text.split(), n=2)
        neh=list(bigram)
        s=[]
        for i in list(range(len(neh))):
            s.append((neh[i][0]+' '+neh[i][1]))
        return Counter(s)


    # # Splitting the sentence by three words

    def trigram_n(text):
        bigram = ngrams(text.split(), n=3)
        neh=list(bigram)
        t=[]
        for i in list(range(len(neh))):
            t.append((neh[i][0]+' '+neh[i][1]+' '+neh[i][2]))
        return Counter(t)


    # # Getting the probability for the Sentence with other
    def word_match_probability(len_vec1,num_questions,list_1):
        list_proba=[]
        for i in list(range(num_questions)):
            prob=list_1[i] / len_vec1
            list_proba.append(prob)
        return list_proba
    
    
    def n_gram_nielsen(text1):
        #unigram common count in Description
        list_summ1=[]
        list_summ2=[]
        list_summ3=[]
        for i in list(range(len(stem_question_summary))):
            clean_text1=Sentence_clean(text1)
            stem_text1=to_do_stem(clean_text1)
            text2 = stem_question_summary[i] 
            vector1_uni = unigram_n(stem_text1)
            vector2_uni = unigram_n(text2)
            vector1_bi = bigram_n(stem_text1)
            vector2_bi = bigram_n(text2)
            vector1_tri = trigram_n(stem_text1)
            vector2_tri = trigram_n(text2)
            common_uni = set(vector1_uni.keys()) & set(vector2_uni.keys())
            common_bi = set(vector1_bi.keys()) & set(vector2_bi.keys())
            common_tri = set(vector1_tri.keys()) & set(vector2_tri.keys())
            list_summ1.append(len(common_uni))
            list_summ2.append(len(common_bi))
            list_summ3.append(len(common_tri))
        return list_summ1,list_summ2,list_summ3,vector1_uni,vector1_bi,vector1_tri


    # # Imporing the Train Data

    train_data_sim = pd.read_csv('C:\\Users\\nnidamanuru\\Documents\\nielsen_neh1.csv')

    print(train_data_sim.shape)

    print(train_data_sim.info())


    # # Issue summary and description sentences are moved into the variables

    question_summary=train_data_sim['Issue Summary']

    question_Description=train_data_sim['Issue Description']

    nltk.download('punkt')
    nltk.download('stopwords')


    # # Cleaning the Sentences


    clean_question_summary=list(map(Sentence_clean,question_summary))

    clean_question_Description=list(map(Sentence_clean,question_Description))

     # # Stemming the words in Sentence

    stem_question_summary=list(map(to_do_stem,clean_question_summary))


    stem_question_Description=list(map(to_do_stem,clean_question_Description))   
    

# # Below are for getting the common words count by Single,Two and Three words respectively
    if validate=='description':
        print("Entered into Description")        

        
        list_des1,list_des2,list_des3,vector1_uni,vector1_bi,vector1_tri=n_gram_nielsen(text1)
                 

        ## Getting the probability of matching a sentence 
    
    
        output_uni=word_match_probability(len(vector1_uni),len(stem_question_Description),list_des1)


        output_bi=word_match_probability(len(vector1_bi),len(stem_question_Description),list_des2)


        output_tri=word_match_probability(len(vector1_tri),len(stem_question_Description),list_des3)



        print(len(output_uni))


        # # Making each list into DataFrame


        score_uni=pd.DataFrame(output_uni)
        list_des1=pd.DataFrame(list_des1)
        list_des2=pd.DataFrame(list_des2)
        list_des3=pd.DataFrame(list_des3)
        score_bi=pd.DataFrame(output_bi)
        score_tri=pd.DataFrame(output_tri)
        stem_question_Description=pd.DataFrame(stem_question_Description)

        result=pd.concat([stem_question_Description,score_uni,list_des1,score_bi,list_des2,score_tri,list_des3],axis=1)

        result.columns=['stem_question_Description','score_uni','list_des1','score_bi','list_des2','score_tri','list_des3']


        ## Sorting for the top 5 values

        fresult = result.sort_values('score_uni',ascending=False)


        #print(fresult.head(5))
        return fresult.head(5)


            
    
    ## Below are for getting the common words count in summary by Single,Two and Three words respectively

    elif validate=='summary':
        print("Entered into summary")
                

        
        list_summ1,list_summ2,list_summ3,vector1_uni,vector1_bi,vector1_tri=n_gram_nielsen(text1)
        
          

        ## Getting the probability of matching a sentence


        output_uni_summ=word_match_probability(len(vector1_uni),len(stem_question_summary),list_summ1)
        output_bi_summ=word_match_probability(len(vector1_bi),len(stem_question_summary),list_summ2)
        output_tri_summ=word_match_probability(len(vector1_tri),len(stem_question_summary),list_summ3)


        ## Making each list into DataFrame

        score_uni_summ=pd.DataFrame(output_uni_summ)
        list_summ1=pd.DataFrame(list_summ1)
        list_summ2=pd.DataFrame(list_summ2)
        list_summ3=pd.DataFrame(list_summ3)
        score_bi_summ=pd.DataFrame(output_bi_summ)
        score_tri_summ=pd.DataFrame(output_tri_summ)
        stem_question_summary=pd.DataFrame(stem_question_summary)

        result=pd.concat([stem_question_summary,score_uni_summ,list_summ1,score_bi_summ,list_summ2,score_tri_summ,list_summ3],axis=1)


        result.columns=['stem_question_summary','score_uni_summ','list_summ1','score_bi_summ','list_summ2','score_tri_summ','list_summ3']


        ## Sorting for the top 5 values


        fresult = result.sort_values('score_uni_summ',ascending=False)

        #print(fresult.head(5))


    ## Write CSV file from DataFrame


        return fresult.head(5)



    







