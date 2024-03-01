import logging
from threading import Thread
import os
import psutil
import bz2
import pickle
import numpy

import wx

#ML libraries
import gensim
#If not getting updated, use this code https://github.com/maximtrp/bitermplus
import bitermplus as btm
from sklearn.decomposition import NMF as nmf
from sklearn.feature_extraction.text import TfidfVectorizer
from top2vec import Top2Vec
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from gensim.models.coherencemodel import CoherenceModel

import Common.CustomEvents as CustomEvents
import Common.Objects.Utilities.Samples as SamplesUtilities

class CaptureThread(Thread):
    def __init__(self, notify_window, main_frame, model_paramaters, model_type):
        Thread.__init__(self)
        self._notify_window = notify_window
        self.main_frame = main_frame
        self.model_parameters = model_paramaters
        self.model_type = model_type
        self.start()
    
    def run(self):
        dataset_key = self.model_parameters['dataset_key']
        self.model_parameters['tokensets'], field_list = SamplesUtilities.CaptureTokens(dataset_key, self.main_frame)
        wx.PostEvent(self._notify_window, CustomEvents.CaptureResultEvent(self.model_type, self.model_parameters, field_list))


class LDATrainingThread(Thread):
    """LDATrainingThread Class."""
    def __init__(self, notify_window, current_workspace_path, key, tokensets, num_topics, num_passes, alpha, eta):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self.daemon = True
        self._notify_window = notify_window
        self.current_workspace_path = current_workspace_path
        self.key = key
        self.tokensets = tokensets
        self.num_topics = num_topics
        self.num_passes = num_passes
        self.alpha = alpha
        self.eta = eta
        self.start()

    def run(self):
        '''Generates an LDA model'''
        logger = logging.getLogger(__name__+"LDATrainingThread["+str(self.key)+"].run")
        logger.info("Starting")
        if not os.path.exists(self.current_workspace_path+"/Samples/"+self.key):
            os.makedirs(self.current_workspace_path+"/Samples/"+self.key)

        logger.info("Starting generation of model")
        tokensets_keys = list(self.tokensets.keys())
        tokensets_values = list(self.tokensets.values())
        dictionary = gensim.corpora.Dictionary(tokensets_values)
        dictionary.compactify()
        dictionary.save(self.current_workspace_path+"/Samples/"+self.key+'/ldadictionary.dict')
        logger.info("Dictionary created")
        raw_corpus = [dictionary.doc2bow(tokenset) for tokenset in tokensets_values]
        gensim.corpora.MmCorpus.serialize(self.current_workspace_path+"/Samples/"+self.key+'/ldacorpus.mm', raw_corpus)
        corpus = gensim.corpora.MmCorpus(self.current_workspace_path+"/Samples/"+self.key+'/ldacorpus.mm')
        logger.info("Corpus created")

        if self.alpha is not None:
            alpha = self.alpha
        else:
            alpha = 'symmetric'
        if self.eta is not None:
            eta = self.eta
        else:
            eta = 'auto'
        
        cpus = psutil.cpu_count(logical=False)
        if cpus is None or cpus < 2:
            workers = 1
        else:
            workers = cpus-1

        model = gensim.models.ldamulticore.LdaMulticore(workers=workers,
                                                        corpus=corpus,
                                                        id2word=dictionary,
                                                        num_topics=self.num_topics,
                                                        passes=self.num_passes,
                                                        alpha=alpha,
                                                        eta=eta)
        model.save(self.current_workspace_path+"/Samples/"+self.key+'/ldamodel.lda', 'wb')
        logger.info("Completed generation of model")
        # Init output
        # capture all document topic probabilities both by document and by topic
        document_topic_prob = {}
        model_document_topics = model.get_document_topics(corpus, minimum_probability=0.0, minimum_phi_value=0)
        for doc_num in range(len(corpus)):
            doc_row = model_document_topics[doc_num]
            doc_topic_prob_row = {}
            for i, prob in doc_row:
                doc_topic_prob_row[i+1] = prob
            document_topic_prob[tokensets_keys[doc_num]] = doc_topic_prob_row

        logger.info("Finished")
        result={'key': self.key, 'document_topic_prob':document_topic_prob}
        wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))

class BitermTrainingThread(Thread):
    """BitermTrainingThread Class."""
    def __init__(self, notify_window, current_workspace_path, key, tokensets, num_topics, num_passes):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self.daemon = True
        self._notify_window = notify_window
        self.current_workspace_path = current_workspace_path
        self.key = key
        self.tokensets = tokensets
        self.num_topics = num_topics
        self.num_passes = num_passes
        self.start()

    def run(self):
        '''Generates an Biterm model'''
        logger = logging.getLogger(__name__+"BitermTrainingThread["+str(self.key)+"].run")
        logger.info("Starting")
        
        if not os.path.exists(self.current_workspace_path+"/Samples/"+self.key):
            os.makedirs(self.current_workspace_path+"/Samples/"+self.key)

        text_keys = []
        texts = []
        for key in self.tokensets:
            text_keys.append(key)
            text = ' '.join(self.tokensets[key])
            texts.append(text)

        logger.info("Starting generation of biterm model")

        X, vocab, vocab_dict = btm.get_words_freqs(texts)
        with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/vocab.pk', 'wb') as outfile:
            pickle.dump(vocab, outfile)
        logger.info("Vocab created")

        # Vectorizing documents
        docs_vec = btm.get_vectorized_docs(texts, vocab)
        with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/transformed_texts.pk', 'wb') as outfile:
            pickle.dump(docs_vec, outfile)
        logger.info("Texts transformed")

        logger.info("Starting Generation of BTM")
        biterms = btm.get_biterms(docs_vec)

        model = btm.BTM(X, vocab, T=self.num_topics, M=20, alpha=50/8, beta=0.01)
        p_zd = model.fit_transform(docs_vec, biterms, iterations=self.num_passes, verbose=False)
        with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/btm.pk', 'wb') as outfile:
            pickle.dump(model, outfile)
        logger.info("Completed Generation of BTM")

        document_topic_prob = {}
        for doc_num in range(len(p_zd)):
            doc_row = p_zd[doc_num]
            doc_topic_prob_row = {}
            for i in range(len(doc_row)):
                doc_topic_prob_row[i+1] = doc_row[i]
            document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row

        logger.info("Finished")
        result={'key': self.key, 'document_topic_prob':document_topic_prob}
        wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))

class NMFTrainingThread(Thread):
    """NMFTrainingThread Class."""
    def __init__(self, notify_window, current_workspace_path, key, tokensets, num_topics):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self.daemon = True
        self._notify_window = notify_window
        self.current_workspace_path = current_workspace_path
        self.key = key
        self.tokensets = tokensets
        self.num_topics = num_topics
        self.start()

    def run(self):
        '''Generates an NMF model'''
        logger = logging.getLogger(__name__+"NMFTrainingThread["+str(self.key)+"].run")
        logger.info("Starting")
        
        if not os.path.exists(self.current_workspace_path+"/Samples/"+self.key):
            os.makedirs(self.current_workspace_path+"/Samples/"+self.key)

        text_keys = []
        texts = []
        for key in self.tokensets:
            text_keys.append(key)
            text = ' '.join(self.tokensets[key])
            texts.append(text)

        logger.info("Starting generation of NMF model")

        tfidf_vectorizer = TfidfVectorizer(max_features=len(self.tokensets.values()), preprocessor=SamplesUtilities.dummy, tokenizer=SamplesUtilities.dummy, token_pattern=None)
        
        tfidf = tfidf_vectorizer.fit_transform(self.tokensets.values())
        with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/tfidf.pk', 'wb') as outfile:
           pickle.dump(tfidf, outfile)
        
        logger.info("Texts transformed")

        logger.info("Starting Generation of NMF")

        model = nmf(self.num_topics, random_state=1).fit(tfidf)

        # must fit tfidf as above before saving tfidf_vectorizer
        with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/tfidf_vectorizer.pk', 'wb') as outfile:
           pickle.dump(tfidf_vectorizer, outfile)

        topics = tfidf_vectorizer.get_feature_names_out()
        topic_pr = model.transform(tfidf)
        topic_pr_sum = numpy.sum(topic_pr, axis=1, keepdims=True)
        probs = numpy.divide(topic_pr, topic_pr_sum, out=numpy.zeros_like(topic_pr), where=topic_pr_sum!=0)

        with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/nmf_model.pk', 'wb') as outfile:
            pickle.dump(model, outfile)
        logger.info("Completed Generation of NMF")

        document_topic_prob = {}
        for doc_num in range(len(probs)):
            doc_row = probs[doc_num]
            doc_topic_prob_row = {}
            for i in range(len(doc_row)):
                doc_topic_prob_row[i+1] = doc_row[i]
            document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row

        logger.info("Finished")
        result={'key': self.key, 'document_topic_prob':document_topic_prob}
        wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))





class Top2VecTrainingThread(Thread):
    """Top2VecTrainingThread Class."""
    def __init__(self, notify_window, current_workspace_path, key, tokensets, num_topics):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self.daemon = True
        self._notify_window = notify_window
        self.current_workspace_path = current_workspace_path
        self.key = key
        self.tokensets = tokensets
        self.num_topics = num_topics
        self.start()

    def run(self):
        '''Generates a Top2Vec model'''
        logger = logging.getLogger(__name__+"Top2VecTrainingThread["+str(self.key)+"].run")
        logger.info("Starting")
        
        if not os.path.exists(self.current_workspace_path+"/Samples/"+self.key):
            os.makedirs(self.current_workspace_path+"/Samples/"+self.key)

        texts = [' '.join(tokenset) for tokenset in self.tokensets.values()]

        logger.info("Starting generation of Top2Vec model")

        model = Top2Vec(texts, embedding_model="distiluse-base-multilingual-cased", min_count=10, workers=-1)

        logger.info("Top2Vec model generated")

        # Save the Top2Vec model
        model.save(self.current_workspace_path+"/Samples/"+self.key+'/top2vec_model')

        logger.info("Top2Vec model saved")

        # Get document-topic probabilities
        document_topic_prob = model.get_documents_topics(doc_ids=self.tokensets.keys())

        logger.info("Finished")
        result = {'key': self.key, 'document_topic_prob': document_topic_prob}
        wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))


class BertopicTrainingThread(Thread):
    """BERTopicTrainingThread Class."""
    def __init__(self, notify_window, current_workspace_path, key, tokensets, num_topics):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self.daemon = True
        self._notify_window = notify_window
        self.current_workspace_path = current_workspace_path
        self.key = key
        self.tokensets = tokensets
        self.num_topics = num_topics
        self.start()

    def run(self):
        '''Generates a BERTopic model'''
        logger = logging.getLogger(__name__+"BertopicTrainingThread["+str(self.key)+"].run")
        logger.info("Starting")
        
        if not os.path.exists(self.current_workspace_path+"/Samples/"+self.key):
            os.makedirs(self.current_workspace_path+"/Samples/"+self.key)

        text_keys = []
        texts = []
        for key in self.tokensets:
            text_keys.append(key)
            text = ' '.join(self.tokensets[key])
            texts.append(text)

        logger.info("Starting generation of BERTopic model")

        # Step 2.1 - Extract embeddings
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Step 2.2 - Reduce dimensionality
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
        
        # Step 2.3 - Cluster reduced embeddings
        hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        
        # Step 2.4 - Tokenize topics
        vectorizer_model = CountVectorizer(stop_words="english")
        
        # Step 2.5 - Create topic representation
        ctfidf_model = ClassTfidfTransformer()
        
        topic_model = BERTopic(
            embedding_model=embedding_model,    # Step 1 - Extract embeddings
            umap_model=umap_model,              # Step 2 - Reduce dimensionality
            hdbscan_model=hdbscan_model,        # Step 3 - Cluster reduced embeddings
            vectorizer_model=vectorizer_model,  # Step 4 - Tokenize topics
            ctfidf_model=ctfidf_model,          # Step 5 - Extract topic words
            nr_topics=self.num_topics           # Step 6 - Diversify topic words
        )

        # Fit BERTopic model
        topics, probabilities = topic_model.fit_transform(texts)

        logger.info("BERTopic model fitted")

        # Save the model and document-topic probabilities
        with bz2.BZ2File(self.current_workspace_path+"/Samples/"+self.key+'/bertopic_model.pk', 'wb') as outfile:
            pickle.dump(topic_model, outfile)

        document_topic_prob = {}
        for doc_num in range(len(probabilities)):
            doc_topic_prob_row = {}
            for topic_id, prob in enumerate(probabilities[doc_num]):
                doc_topic_prob_row[topic_id] = prob
            document_topic_prob[text_keys[doc_num]] = doc_topic_prob_row

        result = {'key': self.key, 'document_topic_prob': document_topic_prob}
        wx.PostEvent(self._notify_window, CustomEvents.ModelCreatedResultEvent(result))



