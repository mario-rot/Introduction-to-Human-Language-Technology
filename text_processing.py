from typing import List, Dict, Tuple
import string
import collections
import nltk
from nltk.metrics import jaccard_distance
from nltk.corpus.reader.wordnet import VERB, NOUN, ADJ, ADV
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
from collections import Counter
import spacy
nlp = spacy.load("en_core_web_sm")

# from nltk.text import Text
# from nltk.corpus import gutenberg
# nltk.download('gutenberg')
# nltk.download('stopwords')

class text_processing(): 
    def __init__(self,
             data: List
             ):
        self.data = data
        self.cleaned_data = []
        self.tokenized_data = []
        self.lemmatized_data = []
        self.most_common_lemmatized_data = []
        self.lesk_data = []
        self.lesk_lemmatized_data = []
        self.name_entities_nltk_data = []
        self.name_entities_spacy_data = []
        self.pos_map = {'N': NOUN,
                        'V':VERB,
                        'J':ADJ,
                        'R':ADV}
        self.match = {'j':"s", 'j':"a", 'r':"r", 'n':"n", 'v':"v"}
        self.correcting = {'n':'n', 'v':'v', 'j':'a', 'r':'r'}
        self.wnl = nltk.stem.WordNetLemmatizer()

    def __len__(self):
        return len(self.data)

    def clean_data(self, data = False, auto = True, lowercase = False, stopwords = False, minwords_len = False, signs = False):
        c_data = self.data if not data else data
        if auto:
            lowercase = True
            stopwords=set(nltk.corpus.stopwords.words('english'))
            signs = string.punctuation
            minwords_len = 2
            for element in c_data:
                self.cleaned_data.append(self.clean_sentence(element, lowercase, stopwords, minwords_len, signs))
        else: 
            for element in c_data:
                 self.cleaned_data.append(self.clean_sentence(element, lowercase, stopwords, minwords_len, signs))
        return self.cleaned_data

    def clean_sentence(self,sentence, lowercase = True, stopwords = False, minwords_len = False, signs = False):
        sentence = sentence.split(' ')
        if lowercase:
            sentence = [word.lower() for word in sentence]
        if signs:
            sentence = [word if not any(caracter in signs for caracter in word) else self.remove_signs(word, signs) for word in sentence]
        if stopwords:
            sentence = [word for word in sentence if word not in stopwords and word.isalpha()]
        if minwords_len:
            sentence = [word for word in sentence if len(word) > minwords_len]
        return sentence

    def tokenize_data(self, data = False):
        t_data = self.data if not data else data
        for element in t_data:
            self.tokenized_data.append(nltk.word_tokenize(element))
        return self.tokenized_data

    def frequency(self, Global = False):
        if self.cleaned_data == []:
            print('\n -- Data hasn\'t been cleaned, to calculate the frequency the data is going to be cleaned with the default parameters --\n')
            self.clean_data() 
        if not Global:
            frequency = []
            for element in self.cleaned_data:
                frequency.append(pd.Series({k:v for k,v in collections.Counter(element).most_common()}))
            return frequency
        else: 
            t_data = [element for sublist in self.cleaned_data for element in sublist]
            frequency = pd.Series({k:v for k,v in collections.Counter(t_data).most_common()})
            return frequency

    def lemmatize_data(self, type_data = 'cleaned'):
        if type_data == 'cleaned':
            if self.cleaned_data == []:
                print('\n -- Data hasn\'t been cleaned, to lemmatize the data is going to be cleaned with the default parameters --\n')
                self.clean_data()
            l_data = self.cleaned_data
        elif type_data == 'tokenized':
            if self.tokenized_data == []:
                print('\n -- Data hasn\'t been tokenized, to lemmatize the data is going to be tokenized --\n')
                self.tokenize_data()
            l_data = self.tokenized_data
        for element in l_data:
            self.lemmatized_data.append(self.lemmatize_sentence(element))
        return self.lemmatized_data

    def lemmatize_sentence(self,sentence, cleaned = True, r_pos_tag = False):
        if not cleaned:
            sentence = self.clean_data([sentence])[0]
        tagged = nltk.pos_tag(sentence)
        if r_pos_tag:
            return [[self.lemmatize(self, pair), pair[1]] for pair in tagged]
        return[self.lemmatize(self, pair) for pair in tagged]

    def mc_lemmatize_data(self, type_data = 'cleaned'):
        if type_data == 'cleaned':
            if self.cleaned_data == []:
                print('\n -- Data hasn\'t been cleaned, to apply lesk to data is going to be cleaned with the default parameters --\n')
                self.clean_data()
            l_data = self.cleaned_data
        elif type_data == 'tokenized':
            if self.tokenized_data == []:
                print('\n -- Data hasn\'t been tokenized, to apply lesk to data is going to be tokenized --\n')
                self.tokenize_data()
            l_data = self.tokenized_data
        for element in l_data:
            self.most_common_lemmatized_data.append(self.most_common_lemma_sentece(element))
        return self.most_common_lemmatized_data
    
    def most_common_lemma_sentece(self,sentence, cleaned = True):
        if not cleaned:
            sentence = self.clean_data([sentence])[0]
        return[self.get_most_common_lemma(self, word) for word in nltk.pos_tag(sentence)]

    def apply_lesk_data(self, type_data = 'cleaned', all = False):
        if type_data == 'cleaned':
            if self.cleaned_data == []:
                print('\n -- Data hasn\'t been cleaned, to lemmatize the data is going to be cleaned with the default parameters --\n')
                self.clean_data()
            ls_data = self.cleaned_data
        elif type_data == 'tokenized':
            if self.tokenized_data == []:
                print('\n -- Data hasn\'t been tokenized, to lemmatize the data is going to be tokenized --\n')
                self.tokenize_data()
            ls_data = self.tokenized_data
        for element in ls_data:
            self.lesk_data.append(self.apply_lesk_sentence(element, all))
        return self.lesk_data

    def apply_lesk_sentence(self, sentence, all = False):
        pairs = nltk.pos_tag(sentence)
        synsets = []
        for pair in pairs:
            word, pos = self.filter_pos(self, pair)
            if pos:
                synset = nltk.wsd.lesk(sentence, word, pos)
            else:
                synset = False
            if synset:
                synsets.append(synset.lemmas()[0].name())
            elif not synset and all:
                synsets.append(pair[0])
        return synsets

    def apply_lesk_lemmas_data(self, type_data = 'cleaned', all = False):
        if type_data == 'cleaned':
            if self.cleaned_data == []:
                print('\n -- Data hasn\'t been cleaned, to lesk lemmatize the data is going to be cleaned with the default parameters --\n')
                self.clean_data()
            ls_data = self.cleaned_data
        elif type_data == 'tokenized':
            if self.tokenized_data == []:
                print('\n -- Data hasn\'t been tokenized, to lesk lemmatize the data is going to be tokenized --\n')
                self.tokenize_data()
            ls_data = self.tokenized_data
        for element in ls_data:
            self.lesk_lemmatized_data.append(self.apply_lesk_lemmas_sentence(element, all))
        return self.lesk_lemmatized_data

    def apply_lesk_lemmas_sentence(self, sentence, all = False):
        lemmatized_sentence = self.lemmatize_sentence(sentence, r_pos_tag = True)
        synsets = []
        for pair in lemmatized_sentence:
            word, pos = self.filter_pos(self, pair)
            if pos:
                synset = nltk.wsd.lesk(sentence, word, pos)
            else:
                synset = False
            if synset:
                synsets.append(synset.lemmas()[0].name())
            elif not synset and all:
                synsets.append(pair[0])
        return synsets

    def name_entities_nltk(self, type_data = 'cleaned'):
        if type_data == 'cleaned':
            if self.cleaned_data == []:
                print('\n -- Data hasn\'t been cleaned, to get named entitites the data is going to be cleaned with the default parameters --\n')
                self.clean_data()
            ls_data = self.cleaned_data
        elif type_data == 'tokenized':
            if self.tokenized_data == []:
                print('\n -- Data hasn\'t been tokenized, to get named entitites the data is going to be tokenized --\n')
                self.tokenize_data()
            ls_data = self.tokenized_data
        for element in ls_data:
            self.name_entities_nltk_data.append(self.named_entities_nltk_sentence(element))
        return self.name_entities_nltk_data 

    def named_entities_nltk_sentence(self, sentence):
        x = nltk.pos_tag(sentence)
        res = nltk.ne_chunk(x)
        named_entities = []
        for item in res:
            try: 
                ne = item.label()
                named_entities.append(ne)
            except:
                named_entities.append(item[0])

        if named_entities != []:
            return named_entities

    def name_entities_spacy(self):
        ne_data = self.data
        for element in ne_data:
            self.name_entities_spacy_data.append(self.name_entities_spacy_sentence(element))
        return self.name_entities_spacy_data

    def name_entities_spacy_sentence(self, sentence):
        doc = nlp(sentence)
        with doc.retokenize() as retokenizer:
            tokens = [token for token in doc]
            for ent in doc.ents:
                retokenizer.merge(doc[ent.start:ent.end], 
                                attrs={"LEMMA": " ".join([tokens[i].text for i in range(ent.start, ent.end)])})
        res = []
        for ent in doc:
            if ent.ent_type_ != '':
                res.append(ent.ent_type_)
                # pass
            else:
                res.append(ent.text) 
            # res.append(ent.text)
        return res

    @staticmethod
    def remove_signs(wrd,signs):
        wrd = list(wrd)
        wrd = [word for word in wrd if not any(caracter in signs for caracter in word)]
        wrd = ''.join(wrd)
        return wrd

    @staticmethod
    def lemmatize(self, p):
        if p[1][0] in {'N','V','J','R'}:
            return self.wnl.lemmatize(p[0].lower(), pos=self.pos_map[p[1][0]])
        return p[0]

    @staticmethod
    def get_most_common_lemma(self,pair):
        try:
            synsets = wn.synsets(pair[0], self.match[pair[1][0].lower()])
            if synsets != []:
                return Counter([j for i in synsets for j in i.lemmas()]).most_common(1)[0][0].name()
            else:
                return Counter([j for i in wn.synsets(pair[0]) for j in i.lemmas()]).most_common(1)[0][0].name()
        except:
            return pair[0]

    @staticmethod
    def filter_pos(self, pair):
        if pair[1][0].lower() in list(self.correcting.keys()):
            return pair[0], self.correcting[pair[1][0].lower()]
        return None, None

class compute_metrics():
    def __init__(self,
             data: List = False,
             metrics: List = False
             ):
        self.data = np.array(data, dtype = object).T
        self.metrics = metrics
        self.methods = {'jaccard': self.jaccard}
        
    def do(self):
        results = []
        for met in self.metrics:
            results.append(self.methods[met]())
        return results
    
    def jaccard(self,data=False):
        j_data = self.data if not data else data
        result = []
        for row in j_data:
            result.append(jaccard_distance(set(row[0]),
                                           set(row[1]))*10)
        return result

class train_tags():
    def __init__(self,
             data,
             amounts_data,
             test_n,
             models
             ):
        self.data = data
        self.amount_data = amounts_data
        self.test_n = test_n
        self.models = models
        self.times = {key:[] for key in self.models}
        self.total_results = {key:[] for key in self.models}
        
    def do(self):
        pbar = tqdm(total=100)
        test_data = self.data[self.test_n:]

        for i in tqdm(self.amount_data):
            train_data = self.data[:i]

            # Hidden Markov Model
            if 'HMM' in self.models:
                time_before = time.time()
                trainer = nltk.tag.hmm.HiddenMarkovModelTrainer()
                HMM = trainer.train_supervised(train_data)
                self.total_results['HMM'].append(round(HMM.accuracy(test_data), 3))
                self.times['HMM'].append(time.time() - time_before)
            
            # Trigrams'n'Tags
            if 'TnT' in self.models:
                time_before = time.time()
                TnT = nltk.tag.tnt.TnT()
                TnT.train(train_data)
                self.total_results['TnT'].append(round(TnT.accuracy(test_data), 3))
                self.times['TnT'].append(time.time() - time_before)

            #  Perceptron tagger
            if 'PER' in self.models:
                time_before = time.time()
                PER = nltk.tag.perceptron.PerceptronTagger(load=False)
                PER.train(train_data)
                self.total_results['PER'].append(round(PER.accuracy(test_data), 3))
                self.times['PER'].append(time.time() - time_before)

            # Conditional Random Fields
            if 'CRF' in self.models:
                time_before = time.time()
                CRF = nltk.tag.CRFTagger()
                CRF.train(train_data,'crf_tagger_model')
                self.total_results['CRF'].append(round(CRF.accuracy(test_data), 3))
                self.times['CRF'].append(time.time() - time_before)

            print(i)

        return self.times, self.total_results
    
    def results(self):
        df = pd.DataFrame.from_dict(self.total_results)

        if 'HMM' in self.models:
            plt.plot(self.amount_data, 'HMM', data=df, marker='.')
        if 'TnT' in self.models:
            plt.plot(self.amount_data, 'TnT', data=df, marker='.')
        if 'PER' in self.models:
            plt.plot(self.amount_data, 'PER', data=df, marker='.', markersize = 10)
        if 'CRF' in self.models:
            plt.plot(self.amount_data, 'CRF', data=df, marker='.')

        plt.legend()
        plt.show()

        df_times = pd.DataFrame.from_dict(self.times).round(3)
        df_times['Sentences'] = self.amount_data
        print(df_times)
    

if __name__ == '__main__':
    dt = pd.read_csv('Complementary Material/test-gold/STS.input.SMTeuroparl.txt',sep='\t',header=None)
    dt['gs'] = pd.read_csv('Complementary Material/test-gold/STS.gs.SMTeuroparl.txt',sep='\t',header=None)
    tp1 = text_processing(dt[0])
    tp1.tokenize_data()
    tp2 = text_processing(dt[1])
    tp2.tokenize_data()
    # print('SS', tp1.mc_lemmatize_data())
    mets = compute_metrics([tp1.name_entities_spacy(),tp2.name_entities_spacy()], ['jaccard']).do()
    print(mets[0])
    # tt = train_tags(nltk.corpus.treebank.tagged_sents(), [500,1000,1500,2000,2500,3000], 3000, ['HMM','TnT','PER','CRF'])
    # tt.do()
    # tt.results
