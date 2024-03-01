import logging
import random
import bz2
import pickle
from datetime import datetime
from collections import OrderedDict

#ML Libraries
import pandas as pd
import numpy as np
import gensim
import bitermplus as btm
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

from Common.Objects.Generic import GenericObject
import Common.Objects.Threads.Samples as SamplesThreads
import Common.Objects.Datasets as Datasets

class Sample(GenericObject):
    '''Instances of Sample objects'''
    def __init__(self, name, dataset_key, sample_type):
        GenericObject.__init__(self, name=name)

        #properties that automatically update last_changed_dt
        self._dataset_key = dataset_key
        self._sample_type = sample_type
        self._model = None
        self._generated_flag = False
        self._start_dt = None
        self._end_dt = None
        self._selected = False

        self._fields_list = None
        self._applied_filter_rules = None
        self._tokenization_package_versions = None
        self._tokenization_choice = None

        #objects that have their own last_changed_dt and thus need to be checked dynamically
        self.parts_dict = OrderedDict()
        self.selected_documents = []

    def __repr__(self):
        return 'Sample[%s][%s]' % (self.name, self.key,)

    @property
    def dataset_key(self):
        return self._dataset_key
    @dataset_key.setter
    def dataset_key(self, value):
        self._dataset_key = value
        self.last_changed_dt = datetime.now()

    @property
    def sample_type(self):
        return self._sample_type
    @sample_type.setter
    def sample_type(self, value):
        self._sample_type = value
        self.last_changed_dt = datetime.now()
    
    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, value):
        self._model = value
        self.last_changed_dt = datetime.now()
    
    @property
    def generated_flag(self):
        return self._generated_flag
    @generated_flag.setter
    def generated_flag(self, value):
        self._generated_flag = value
        self.last_changed_dt = datetime.now()
    
    @property
    def start_dt(self):
        return self._start_dt
    @start_dt.setter
    def start_dt(self, value):
        self._start_dt = value
        self.last_changed_dt = datetime.now()
    
    @property
    def end_dt(self):
        return self._end_dt
    @end_dt.setter
    def end_dt(self, value):
        self._end_dt = value
        self.last_changed_dt = datetime.now()
    
    @property
    def selected(self):
        if not hasattr(self, '_selected'):
            self._selected = False
        return self._selected
    @selected.setter
    def selected(self, value):
        self._selected = value
        self.last_changed_dt = datetime.now()
    
    @property
    def fields_list(self):
        return self._fields_list
    @fields_list.setter
    def fields_list(self, value):
        self._fields_list = value
        self.last_changed_dt = datetime.now()
    
    @property
    def applied_filter_rules(self):
        return self._applied_filter_rules
    @applied_filter_rules.setter
    def applied_filter_rules(self, value):
        self._applied_filter_rules = value
        self.last_changed_dt = datetime.now()
    
    @property
    def tokenization_choice(self):
        return self._tokenization_choice
    @tokenization_choice.setter
    def tokenization_choice(self, value):
        self._tokenization_choice = value
        self.last_changed_dt = datetime.now()
    
    @property
    def tokenization_package_versions(self):
        return self._tokenization_package_versions
    @tokenization_package_versions.setter
    def tokenization_package_versions(self, value):
        self._tokenization_package_versions = value
        self.last_changed_dt = datetime.now()

    @property
    def last_changed_dt(self):
        for part_name in self.parts_dict:
            tmp_last_changed_dt = self.parts_dict[part_name].last_changed_dt
            if tmp_last_changed_dt > self._last_changed_dt:
                self._last_changed_dt = tmp_last_changed_dt
        return self._last_changed_dt
    @last_changed_dt.setter
    def last_changed_dt(self, value):
        self._last_changed_dt = value

    def Generate(self):
        logger = logging.getLogger(__name__+"."+repr(self)+".Generate")
        logger.info("Starting")
        logger.info("Finished")

    def DestroyObject(self):
        logger = logging.getLogger(__name__+"."+repr(self)+".DestroyObject")
        logger.info("Starting")
        #any children models or reviews
        for part_key in list(self.parts_dict.keys()):
            self.parts_dict[part_key].DestroyObject()
        logger.info("Finished")

    def Reload(self):
        logger = logging.getLogger(__name__+""+repr(self)+".Reload")
        logger.info("Starting")
        logger.info("Finished")

    def Load(self, current_workspace):
        logger = logging.getLogger(__name__+"."+repr(self)+".Load")
        logger.info("Starting")
        logger.info("Finished")

    def Save(self, current_workspace):
        logger = logging.getLogger(__name__+"."+repr(self)+".Save")
        logger.info("Starting")
        logger.info("Finished")

class RandomSample(Sample):
    def __init__(self, name, dataset_key, model_parameters):
        Sample.__init__(self, name, dataset_key, "Random")

        #list that is not managed thus need lastchanged_dt to be updated when changed
        self.doc_ids = model_parameters['doc_ids']

    def __repr__(self):
        return 'RandomSample[%s][%s]' % (self.name, self.key,)

    def Generate(self, datasets):
        logger = logging.getLogger(__name__+"."+repr(self)+".Generate")
        logger.info("Starting")
        self.start_dt = datetime.now()
        if not self.generated_flag:
            random.shuffle(self.doc_ids)
            self.last_changed_dt = datetime.now()
            self.generated_flag = True
            model_part = ModelPart(self, "Randomly Ordered Documents", self.doc_ids, datasets)
            self.parts_dict[model_part.key] = model_part
        self.end_dt = datetime.now()
        logger.info("Finished")

class TopicSample(Sample):
    def __init__(self, name, dataset_key, sample_type, model_parameters):
        Sample.__init__(self, name, dataset_key, sample_type)

        #properties that automatically update last_changed_dt
        self._word_num = 0
        self._document_cutoff = 0.25
        self._document_topic_prob = None
        #fixed properties that may be externally accessed but do not change after being initialized
        self._tokensets = model_parameters['tokensets']
        self._num_topics = model_parameters['num_topics']

        #dictionary that is managed with setters

        #objects that have their own last_changed_dt and thus need to be checked dynamically
        
        #variable that should only be used internally and are never accessed from outside

    def __repr__(self):
        return 'TopicSample[%s][%s]' % (self.name, self.key,)
    
    @property
    def key(self):
        return self._key
    @key.setter
    def key(self, value):
        self._key = value
        self._new_filedir = "/"+self.sample_type+"/"+self._key
        self._last_changed_dt = datetime.now()

    @property
    def word_num(self):
        return self._word_num
    @word_num.setter
    def word_num(self, value):
        for part_key in self.parts_dict:
            self.parts_dict[part_key].word_num = value
        self._word_num = value
        self.last_changed_dt = datetime.now()

    @property
    def document_cutoff(self):
        return self._document_cutoff
    @document_cutoff.setter
    def document_cutoff(self, value):
        self._document_cutoff = value
        self.last_changed_dt = datetime.now()

    @property
    def document_topic_prob(self):
        return self._document_topic_prob
    @document_topic_prob.setter
    def document_topic_prob(self, value):
        self._document_topic_prob = value
        self.last_changed_dt = datetime.now()

    @property
    def num_topics(self):
        return self._num_topics
    
    @property
    def tokensets(self):
        return self._tokensets

    def ApplyDocumentCutoff(self):
        logger = logging.getLogger(__name__+"."+repr(self)+".ApplyDocumentCutoff")
        logger.info("Starting")
        document_set = set()
        document_topic_prob_df = pd.DataFrame(data=self.document_topic_prob).transpose()

        def UpdateLDATopicPart(topic):
            document_list = []
            document_s = document_topic_prob_df[topic].sort_values(ascending=False)
            document_list = document_s.index[document_s >= self.document_cutoff].tolist()
            document_set.update(document_list)
            self.parts_dict[topic].part_data = document_list

        for topic in self.parts_dict:
            if isinstance(self.parts_dict[topic], Part) and topic != 'unknown':
                UpdateLDATopicPart(topic)
            elif isinstance(self.parts_dict[topic], MergedPart):
                for subtopic in self.parts_dict[topic].parts_dict:
                    if isinstance(self.parts_dict[topic].parts_dict[subtopic], Part) and topic != 'unknown':
                        UpdateLDATopicPart(topic)
        
        # unknown_list = set(self._tokensets) - document_set
        # unknown_df = document_topic_prob_df[document_topic_prob_df.index.isin(unknown_list)]
        # unknown_series = unknown_df.max(axis=1).sort_values()
        # new_unknown_list = list(unknown_series.index.values)
        
        # document_topic_prob_df["unknown"] = 0.0
        # document_topic_prob_df.loc[unknown_list, "unknown"] = 1.0
        # self.document_topic_prob = document_topic_prob_df.to_dict(orient='index')
                        
        # Convert unknown_list set to a list
        unknown_list = list(set(self._tokensets) - document_set)
        unknown_df = document_topic_prob_df[document_topic_prob_df.index.isin(unknown_list)]
        unknown_series = unknown_df.max(axis=1).sort_values()
        new_unknown_list = list(unknown_series.index.values)

        document_topic_prob_df["unknown"] = 0.0
        document_topic_prob_df.loc[unknown_list, "unknown"] = 1.0
        self.document_topic_prob = document_topic_prob_df.to_dict(orient='index')

        self.parts_dict['unknown'].part_data = list(new_unknown_list)
        logger.info("Finished")

class LDASample(TopicSample):
    def __init__(self, name, dataset_key, model_parameters):
        TopicSample.__init__(self, name, dataset_key, "LDA", model_parameters)

        #fixed properties that may be externally accessed but do not change after being initialized
        self._num_passes = model_parameters['num_passes']
        self._alpha = model_parameters['alpha']
        self._eta = model_parameters['eta']

        #these need to be removed before pickling during saving due to threading and use of multiple processes
        #see __getstate__ for removal and Load and Reload for readdition
        self.training_thread = None
        self.dictionary = None
        self.corpus = None
        self.model = None

    def __repr__(self):
        return 'LDASample[%s][%s]' % (self.name, self.key,)

    def __getstate__(self):
        state = dict(self.__dict__)
        #state['res'] = None
        state['training_thread'] = None
        state['dictionary'] = None
        state['corpus'] = None
        state['model'] = None
        return state
    
    @property
    def num_passes(self):
        return self._num_passes

    @property
    def alpha(self):
        return self._alpha

    @property
    def eta(self):
        return self._eta

    def GenerateStart(self, notify_window, current_workspace_path, start_dt):
        logger = logging.getLogger(__name__+"."+repr(self)+".GenerateStart")
        logger.info("Starting")
        self.start_dt = start_dt
        self.training_thread = SamplesThreads.LDATrainingThread(notify_window,
                                                                current_workspace_path,
                                                                self.key,
                                                                self.tokensets,
                                                                self.num_topics,
                                                                self._num_passes,
                                                                self.alpha,
                                                                self.eta)
        logger.info("Finished")
    
    def GenerateFinish(self, result, dataset, current_workspace):
        logger = logging.getLogger(__name__+"."+repr(self)+".GenerateFinish")
        logger.info("Starting")
        self.generated_flag = True
        self.training_thread.join()
        self.training_thread = None
        self._tokensets = list(self.tokensets.keys())
        self.dictionary = gensim.corpora.Dictionary.load(current_workspace+"/Samples/"+self.key+'/ldadictionary.dict')
        self.corpus = gensim.corpora.MmCorpus(current_workspace+"/Samples/"+self.key+'/ldacorpus.mm')
        self.model = gensim.models.ldamodel.LdaModel.load(current_workspace+"/Samples/"+self.key+'/ldamodel.lda')

        self.document_topic_prob = result['document_topic_prob']

        for i in range(self.num_topics):
            topic_num = i+1
            self.parts_dict[topic_num] = LDATopicPart(self, topic_num, dataset)
        self.parts_dict['unknown'] = TopicUnknownPart(self, 'unknown', [], dataset)

        self.word_num = 10
        self.ApplyDocumentCutoff()
        
        self.end_dt = datetime.now()
        logger.info("Finished")

    def Load(self, current_workspace):
        logger = logging.getLogger(__name__+"."+repr(self)+".Load")
        logger.info("Starting")
        if self.generated_flag:
            self.dictionary = gensim.corpora.Dictionary.load(current_workspace+"/Samples/"+self.key+'/ldadictionary.dict')
            self.corpus = gensim.corpora.MmCorpus(current_workspace+"/Samples/"+self.key+'/ldacorpus.mm')
            self._model = gensim.models.ldamodel.LdaModel.load(current_workspace+"/Samples/"+self.key+'/ldamodel.lda')
        logger.info("Finished")

    def Save(self, current_workspace):
        logger = logging.getLogger(__name__+"."+repr(self)+".Save")
        logger.info("Starting")
        if self.model is not None:
            self.model.save(current_workspace+"/Samples/"+self.key+'/ldamodel.lda', 'wb')
        if self.dictionary is not None:
            self.dictionary.save(current_workspace+"/Samples/"+self.key+'/ldadictionary.dict')
        if self.corpus is not None:
            gensim.corpora.MmCorpus.serialize(current_workspace+"/Samples/"+self.key+'/ldacorpus.mm', self.corpus)
        logger.info("Finished")

class BitermSample(TopicSample):
    def __init__(self, name, dataset_key, model_parameters):
        TopicSample.__init__(self, name, dataset_key, "Biterm", model_parameters)

        #fixed properties that may be externally accessed but do not change after being initialized
        self._num_passes = model_parameters['num_passes']

        #these need to be removed before pickling during saving due to threading and use of multiple processes
        #see __getstate__ for removal and Load and Reload for readdition
        self.training_thread = None
        self.transformed_texts = None
        self.vocab = None
        self.model = None

    def __repr__(self):
        return 'Biterm Sample[%s][%s]' % (self.name, self.key,)

    def __getstate__(self):
        state = dict(self.__dict__)
        state['training_thread'] = None
        state['transformed_texts'] = None
        state['vocab'] = None
        state['model'] = None
        return state
    def __repr__(self):
        return 'BitermSample: %s' % (self.key,)

    @property
    def num_passes(self):
        return self._num_passes
    
    def GenerateStart(self, notify_window, current_workspace_path, start_dt):
        logger = logging.getLogger(__name__+"."+repr(self)+".GenerateStart")
        logger.info("Starting")
        self.start_dt = start_dt
        self.training_thread = SamplesThreads.BitermTrainingThread(notify_window,
                                                                   current_workspace_path,
                                                                   self.key,
                                                                   self.tokensets,
                                                                   self.num_topics,
                                                                   self._num_passes)
        logger.info("Finished")
    
    def GenerateFinish(self, result, dataset, current_workspace):
        logger = logging.getLogger(__name__+"."+repr(self)+".GenerateFinish")
        logger.info("Starting")
        self.generated_flag = True
        self.training_thread.join()
        self.training_thread = None
        self._tokensets = list(self.tokensets.keys())
        with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/transformed_texts.pk', 'rb') as infile:
            self.transformed_texts = pickle.load(infile)
        with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/vocab.pk', 'rb') as infile:
            self.vocab = pickle.load(infile)
        with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/btm.pk', 'rb') as infile:
            self.model = pickle.load(infile)

        self.document_topic_prob = result['document_topic_prob']

        for i in range(self.num_topics):
            topic_num = i+1
            self.parts_dict[topic_num] = BitermTopicPart(self, topic_num, dataset)
        self.parts_dict['unknown'] = TopicUnknownPart(self, 'unknown', [], dataset)

        self.word_num = 10
        self.ApplyDocumentCutoff()
        
        self.end_dt = datetime.now()
        logger.info("Finished")

    def Load(self, current_workspace):
        logger = logging.getLogger(__name__+"."+repr(self)+".Load")
        logger.info("Starting")
        if self.generated_flag:
            with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/transformed_texts.pk', 'rb') as infile:
                self.transformed_texts = pickle.load(infile)
            with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/vocab.pk', 'rb') as infile:
                self.vocab = pickle.load(infile)
            with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/btm.pk', 'rb') as infile:
                self._model = pickle.load(infile)
        logger.info("Finished")

    def Save(self, current_workspace):
        logger = logging.getLogger(__name__+"."+repr(self)+".Save")
        logger.info("Starting")
        if self.transformed_texts is not None:
            with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/transformed_texts.pk', 'wb') as outfile:
                pickle.dump(self.transformed_texts, outfile)
        if self.vocab is not None:
            with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/vocab.pk', 'wb') as outfile:
                pickle.dump(self.vocab, outfile)
        if self.model is not None:
            with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/btm.pk', 'wb') as outfile:
                pickle.dump(self.model, outfile)
        logger.info("Finished")

class NMFSample(TopicSample):
    def __init__(self, name, dataset_key, model_parameters):
        TopicSample.__init__(self, name, dataset_key, "NMF", model_parameters)

        #fixed properties that may be externally accessed but do not change after being initialized

        #these need to be removed before pickling during saving due to threading and use of multiple processes
        #see __getstate__ for removal and Load and Reload for readdition
        self.training_thread = None
        self.vectorizer = None
        self.transformed_texts = None
        self.model = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state['training_thread'] = None
        state['vectorizer'] = None
        state['transformed_texts'] = None
        state['model'] = None
        return state

    def __repr__(self):
        return 'NMFSample[%s][%s]' % (self.name, self.key,)
    
    def GenerateStart(self, notify_window, current_workspace_path, start_dt):
        logger = logging.getLogger(__name__+"."+repr(self)+".GenerateStart")
        logger.info("Starting")
        self.start_dt = start_dt
        self.training_thread = SamplesThreads.NMFTrainingThread(notify_window,
                                                                   current_workspace_path,
                                                                   self.key,
                                                                   self.tokensets,
                                                                   self.num_topics)
        logger.info("Finished")
    
    def GenerateFinish(self, result, dataset, current_workspace):
        logger = logging.getLogger(__name__+"."+repr(self)+".GenerateFinish")
        logger.info("Starting")
        self.generated_flag = True
        self.training_thread.join()
        self.training_thread = None
        self._tokensets = list(self.tokensets.keys())
        with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/tfidf_vectorizer.pk', 'rb') as infile:
           self.vectorizer = pickle.load(infile)
        with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/tfidf.pk', 'rb') as infile:
            self.transformed_texts = pickle.load(infile)
        with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/nmf_model.pk', 'rb') as infile:
            self.model = pickle.load(infile)

        self.document_topic_prob = result['document_topic_prob']

        for i in range(self.num_topics):
            topic_num = i+1
            self.parts_dict[topic_num] = NMFTopicPart(self, topic_num, dataset)
        self.parts_dict['unknown'] = TopicUnknownPart(self, 'unknown', [], dataset)

        self.word_num = 10
        self.ApplyDocumentCutoff()
        
        self.end_dt = datetime.now()
        logger.info("Finished")

    def OldLoad(self, workspace_path):
        logger = logging.getLogger(__name__+"."+repr(self)+".Load")
        logger.info("Starting")
        self._workspace_path = workspace_path
        if self.generated_flag:
            with bz2.BZ2File(self._workspace_path+self.filedir+'/tfidf_vectorizer.pk', 'rb') as infile:
                self.vectorizer = pickle.load(infile)
            with bz2.BZ2File(self._workspace_path+self.filedir+'/tfidf.pk', 'rb') as infile:
                self.transformed_texts = pickle.load(infile)
            with bz2.BZ2File(self._workspace_path+self.filedir+'/nmf_model.pk', 'rb') as infile:
                self._model = pickle.load(infile)
        logger.info("Finished")

    def Load(self, current_workspace):
        logger = logging.getLogger(__name__+"."+repr(self)+".Load")
        logger.info("Starting")
        if self.generated_flag:
            with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/tfidf_vectorizer.pk', 'rb') as infile:
                self.vectorizer = pickle.load(infile)
            with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/tfidf.pk', 'rb') as infile:
                self.transformed_texts = pickle.load(infile)
            with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/nmf_model.pk', 'rb') as infile:
                self._model = pickle.load(infile)
        logger.info("Finished")

    def Save(self, current_workspace):
        logger = logging.getLogger(__name__+"."+repr(self)+".Save")
        logger.info("Starting")
        if self.vectorizer is not None:
            with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/tfidf_vectorizer.pk', 'wb') as outfile:
                pickle.dump(self.vectorizer, outfile)
        if self.transformed_texts is not None:
            with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/tfidf.pk', 'wb') as outfile:
                pickle.dump(self.transformed_texts, outfile)
        if self.model is not None:
            with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/nmf_model.pk', 'wb') as outfile:
                pickle.dump(self.model, outfile)
        logger.info("Finished")



class Top2VecSample(TopicSample):
    def __init__(self, name, dataset_key, model_parameters):
        TopicSample.__init__(self, name, dataset_key, "Top2Vec", model_parameters)

        #fixed properties that may be externally accessed but do not change after being initialized

        #these need to be removed before pickling during saving due to threading and use of multiple processes
        #see __getstate__ for removal and Load and Reload for readdition
        self.training_thread = None
        self.vectorizer = None
        self.transformed_texts = None
        self.model = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state['training_thread'] = None
        state['vectorizer'] = None
        state['transformed_texts'] = None
        state['model'] = None
        return state

    def __repr__(self):
        return 'Top2VecSample[%s][%s]' % (self.name, self.key,)
    
    def GenerateStart(self, notify_window, current_workspace_path, start_dt):
        logger = logging.getLogger(__name__+"."+repr(self)+".GenerateStart")
        logger.info("Starting")
        self.start_dt = start_dt
        self.training_thread = SamplesThreads.Top2VecTrainingThread(notify_window,
                                                                   current_workspace_path,
                                                                   self.key,
                                                                   self.tokensets,
                                                                   self.num_topics)
        logger.info("Finished")
    
    def GenerateFinish(self, result, dataset, current_workspace):
        logger = logging.getLogger(__name__+"."+repr(self)+".GenerateFinish")
        logger.info("Starting")
        self.generated_flag = True
        self.training_thread.join()
        self.training_thread = None
        self._tokensets = list(self.tokensets.keys())
        with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/tfidf_vectorizer.pk', 'rb') as infile:
           self.vectorizer = pickle.load(infile)
        with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/tfidf.pk', 'rb') as infile:
            self.transformed_texts = pickle.load(infile)
        with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/top2vec_model.pk', 'rb') as infile:
            self.model = pickle.load(infile)

        self.document_topic_prob = result['document_topic_prob']

        for i in range(self.num_topics):
            topic_num = i+1
            self.parts_dict[topic_num] = Top2VecTopicPart(self, topic_num, dataset)
        self.parts_dict['unknown'] = TopicUnknownPart(self, 'unknown', [], dataset)

        self.word_num = 10
        self.ApplyDocumentCutoff()
        
        self.end_dt = datetime.now()
        logger.info("Finished")

    def OldLoad(self, workspace_path):
        logger = logging.getLogger(__name__+"."+repr(self)+".Load")
        logger.info("Starting")
        self._workspace_path = workspace_path
        if self.generated_flag:
            with bz2.BZ2File(self._workspace_path+self.filedir+'/tfidf_vectorizer.pk', 'rb') as infile:
                self.vectorizer = pickle.load(infile)
            with bz2.BZ2File(self._workspace_path+self.filedir+'/tfidf.pk', 'rb') as infile:
                self.transformed_texts = pickle.load(infile)
            with bz2.BZ2File(self._workspace_path+self.filedir+'/top2vec_model.pk', 'rb') as infile:
                self._model = pickle.load(infile)
        logger.info("Finished")

    def Load(self, current_workspace):
        logger = logging.getLogger(__name__+"."+repr(self)+".Load")
        logger.info("Starting")
        if self.generated_flag:
            with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/tfidf_vectorizer.pk', 'rb') as infile:
                self.vectorizer = pickle.load(infile)
            with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/tfidf.pk', 'rb') as infile:
                self.transformed_texts = pickle.load(infile)
            with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/top2vec_model.pk', 'rb') as infile:
                self._model = pickle.load(infile)
        logger.info("Finished")

    def Save(self, current_workspace):
        logger = logging.getLogger(__name__+"."+repr(self)+".Save")
        logger.info("Starting")
        if self.vectorizer is not None:
            with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/tfidf_vectorizer.pk', 'wb') as outfile:
                pickle.dump(self.vectorizer, outfile)
        if self.transformed_texts is not None:
            with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/tfidf.pk', 'wb') as outfile:
                pickle.dump(self.transformed_texts, outfile)
        if self.model is not None:
            with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/top2vec_model.pk', 'wb') as outfile:
                pickle.dump(self.model, outfile)
        logger.info("Finished")


class BertopicSample(TopicSample):
    def __init__(self, name, dataset_key, model_parameters):
        TopicSample.__init__(self, name, dataset_key, "Bertopic", model_parameters)

        #fixed properties that may be externally accessed but do not change after being initialized

        #these need to be removed before pickling during saving due to threading and use of multiple processes
        #see __getstate__ for removal and Load and Reload for readdition
        self.training_thread = None
        self.vectorizer = None
        self.transformed_texts = None
        self.model = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state['training_thread'] = None
        state['vectorizer'] = None
        state['transformed_texts'] = None
        state['model'] = None
        return state

    def __repr__(self):
        return 'BertopicSample[%s][%s]' % (self.name, self.key,)
    
    def GenerateStart(self, notify_window, current_workspace_path, start_dt):
        logger = logging.getLogger(__name__+"."+repr(self)+".GenerateStart")
        logger.info("Starting")
        self.start_dt = start_dt
        self.training_thread = SamplesThreads.BertopicTrainingThread(notify_window,
                                                                   current_workspace_path,
                                                                   self.key,
                                                                   self.tokensets,
                                                                   self.num_topics)
        logger.info("Finished")
    
    def GenerateFinish(self, result, dataset, current_workspace):
        logger = logging.getLogger(__name__+"."+repr(self)+".GenerateFinish")
        logger.info("Starting")
        self.generated_flag = True
        self.training_thread.join()
        self.training_thread = None
        self._tokensets = list(self.tokensets.keys())
        with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/tfidf_vectorizer.pk', 'rb') as infile:
           self.vectorizer = pickle.load(infile)
        with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/tfidf.pk', 'rb') as infile:
            self.transformed_texts = pickle.load(infile)
        with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/Bertopic_model.pk', 'rb') as infile:
            self.model = pickle.load(infile)

        self.document_topic_prob = result['document_topic_prob']

        for i in range(self.num_topics):
            topic_num = i+1
            self.parts_dict[topic_num] = BertopicTopicPart(self, topic_num, dataset)
        self.parts_dict['unknown'] = TopicUnknownPart(self, 'unknown', [], dataset)

        self.word_num = 10
        self.ApplyDocumentCutoff()
        
        self.end_dt = datetime.now()
        logger.info("Finished")

    def OldLoad(self, workspace_path):
        logger = logging.getLogger(__name__+"."+repr(self)+".Load")
        logger.info("Starting")
        self._workspace_path = workspace_path
        if self.generated_flag:
            with bz2.BZ2File(self._workspace_path+self.filedir+'/tfidf_vectorizer.pk', 'rb') as infile:
                self.vectorizer = pickle.load(infile)
            with bz2.BZ2File(self._workspace_path+self.filedir+'/tfidf.pk', 'rb') as infile:
                self.transformed_texts = pickle.load(infile)
            with bz2.BZ2File(self._workspace_path+self.filedir+'/Bertopic_model.pk', 'rb') as infile:
                self._model = pickle.load(infile)
        logger.info("Finished")

    def Load(self, current_workspace):
        logger = logging.getLogger(__name__+"."+repr(self)+".Load")
        logger.info("Starting")
        if self.generated_flag:
            with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/tfidf_vectorizer.pk', 'rb') as infile:
                self.vectorizer = pickle.load(infile)
            with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/tfidf.pk', 'rb') as infile:
                self.transformed_texts = pickle.load(infile)
            with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/Bertopic_model.pk', 'rb') as infile:
                self._model = pickle.load(infile)
        logger.info("Finished")

    def Save(self, current_workspace):
        logger = logging.getLogger(__name__+"."+repr(self)+".Save")
        logger.info("Starting")
        if self.vectorizer is not None:
            with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/tfidf_vectorizer.pk', 'wb') as outfile:
                pickle.dump(self.vectorizer, outfile)
        if self.transformed_texts is not None:
            with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/tfidf.pk', 'wb') as outfile:
                pickle.dump(self.transformed_texts, outfile)
        if self.model is not None:
            with bz2.BZ2File(current_workspace+"/Samples/"+self.key+'/Bertopic_model.pk', 'wb') as outfile:
                pickle.dump(self.model, outfile)
        logger.info("Finished")
        

class MergedPart(GenericObject):
    def __init__(self, parent, key, name=None):
        if name is None:
            name="Merged Part "+str(key)
        GenericObject.__init__(self, key, parent=parent, name=name)

        #properties that automatically update last_changed_dt
        self._selected  = False

        #objects that have their own last_changed_dt and thus need to be checked dynamically
        self.parts_dict = OrderedDict()

    def __repr__(self):
        return 'Merged Part %s' % (self.key,)

    @property
    def selected(self):
        if not hasattr(self, '_selected'):
            self._selected = False
        return self._selected
    @selected.setter
    def selected(self, value):
        self._selected = value
        self.last_changed_dt = datetime.now()

    @property
    def last_changed_dt(self):
        for part_key in self.parts_dict:
            tmp_last_changed_dt = self.parts_dict[part_key].last_changed_dt
            if tmp_last_changed_dt > self._last_changed_dt:
                self._last_changed_dt = tmp_last_changed_dt
        return self._last_changed_dt
    @last_changed_dt.setter
    def last_changed_dt(self, value):
        self._last_changed_dt = value

    def DestroyObject(self):
        #any children Samples or GroupedSamples
        for part_key in list(self.parts_dict.keys()):
            self.parts_dict[part_key].DestroyObject()
        del self.parent.parts_dict[self.key]
        self.parent.last_changed_dt = datetime.now()

class ModelMergedPart(MergedPart):
    def __repr__(self):
        return 'Model Merged Part %s' % (self.key,)

    def UpdateDocumentNum(self, document_num, dataset):
        logger = logging.getLogger(__name__+".ModelMergedPart["+str(self.key)+"].UpdateDocumentNum")
        logger.info("Starting")
        for part_key in self.parts_dict:
            self.parts_dict[part_key].UpdateDocumentNum(document_num, dataset)
        self.last_changed_dt = datetime.now()
        logger.info("Finished")

class TopicMergedPart(ModelMergedPart):
    '''Instances of Merged LDA Topic objects'''
    def __init__(self, parent, key, name=None):
        if name is None:
            name = "Merged Topic: "+str(key)
        ModelMergedPart.__init__(self, parent, key, name=name)

        #properties that automatically update last_changed_dt
        self._word_num = 0

    def __repr__(self):
        return 'Merged Topic %s' % (self.key,) if self.label == "" else 'Merged Topic %s: %s' % (self.key, self.label,)
    
    @property
    def word_num(self):
        return self._word_num
    @word_num.setter
    def word_num(self, value):
        self._word_num = value
        for part_key in self.parts_dict:
            self.parts_dict[part_key].word_num = value
        self.last_changed_dt = datetime.now()

    def GetTopicKeywordsList(self):
        keywords_dict = {}
        for part_key in self.parts_dict:
            part_keywords = self.parts_dict[part_key].GetTopicKeywordsList()
            for keyword, prob in part_keywords:
                if keyword in keywords_dict:
                    keywords_dict[keyword] = keywords_dict[keyword] + prob
                else:
                    keywords_dict[keyword] = prob
        keywords_sorted = sorted(keywords_dict.items(), key=lambda x:x[1], reverse=True)
        return keywords_sorted

class Part(GenericObject):
    def __init__(self, parent, key, name=None):
        if name is None:
            name = "Part "+str(key)
        GenericObject.__init__(self, key, parent=parent, name=name)

        #properties that automatically update last_changed_dt
        self._document_num = 0
        self._selected = False
        
        #dictionary that is managed with setters
        self.documents = []

    def __repr__(self):
        return 'Part %s' % (self.key,)

    @property
    def document_num(self):
        return self._document_num
    @document_num.setter
    def document_num(self, value):
        self._document_num = value
        self.last_changed_dt = datetime.now()
    
    @property
    def selected(self):
        if not hasattr(self, '_selected'):
            self._selected = False
        return self._selected
    @selected.setter
    def selected(self, value):
        self._selected = value
        self.last_changed_dt = datetime.now()

    def DestroyObject(self):
        del self.parent.parts_dict[self.key]
        self.parent.last_changed_dt = datetime.now()

class ModelPart(Part):
    '''Instances of a part'''
    def __init__(self, parent, key, part_data, dataset, name=None):
        if name is None:
            name = "Model Part "+str(key)
        Part.__init__(self, parent, key, name=name)

        #properties that automatically update last_changed_dt
        self._part_data = part_data

        self.UpdateDocumentNum(10, dataset)

    def __repr__(self):
        return 'Model Part %s' % (self.key,)

    @property
    def part_data(self):
        return self._part_data
    @part_data.setter
    def part_data(self, value):
        self._part_data = value
        self.last_changed_dt = datetime.now()

    def UpdateDocumentNum(self, document_num, dataset):
        logger = logging.getLogger(__name__+".ModelPart["+str(self.key)+"].UpdateDocumentNum")
        logger.info("Starting")

        #cannot have more documents than what is available
        if document_num > len(self.part_data):
            document_num = len(self.part_data)
        #shrink if appropriate
        if document_num < self.document_num:
            self.documents = self.documents[:document_num]
            self.last_changed_dt = datetime.now()
            self.document_num = document_num
        #grow if approrpriate
        elif document_num > self.document_num:
            for i in range(self.document_num, document_num):
                doc_id = self.part_data[i]
                if isinstance(dataset, Datasets.Dataset):
                    document = dataset.GetDocument(doc_id)
                if document is not None:
                    self.documents.append(document.key)
                    document.AddSampleConnections(self)
                    self.last_changed_dt = datetime.now()
            self.document_num = document_num
        logger.info("Finished")

class TopicPart(ModelPart):
    '''Instances of Topic objects'''
    def __init__(self, parent, key, dataset, name=None):
        if name is None:
            name = "Topic "+str(key)
        ModelPart.__init__(self, parent, key, [], dataset, name)
        
        #properties that automatically update last_changed_dt
        self._word_num = 0
        self._word_list = []

    def __repr__(self):
        return 'Topic %s' % (self.key,) if self.label == "" else 'Topic %s: %s' % (self.key, self.label,)

    @property
    def word_num(self):
        return self._word_num
    @word_num.setter
    def word_num(self, value):
        self._word_num = value
        self.last_changed_dt = datetime.now()
    
    @property
    def word_list(self):
        return self._word_list
    @word_list.setter
    def word_list(self, value):
        self._word_list = value
        self.last_changed_dt = datetime.now()

    def GetTopicKeywordsList(self):
        return self.word_list[0:self.word_num]

class LDATopicPart(TopicPart):
    @property
    def word_num(self):
        return self._word_num
    @word_num.setter
    def word_num(self, value):
        logger = logging.getLogger(__name__+".LDATopicPart["+str(self.key)+"].word_num")
        logger.info("Starting")
        self._word_num = value
        if len(self.word_list) < value:
            self.word_list.clear()
            if isinstance(self.parent, ModelMergedPart):
                self.word_list.extend(self.parent.parent.model.show_topic(self.key-1, topn=value))
            else:
                self.word_list.extend(self.parent.model.show_topic(self.key-1, topn=value))
        self.last_changed_dt = datetime.now()
        logger.info("Finished")

class BitermTopicPart(TopicPart):
    @property
    def word_num(self):
        return self._word_num
    @word_num.setter
    def word_num(self, value):
        logger = logging.getLogger(__name__+".BitermTopicPart["+str(self.key)+"].word_num")
        logger.info("Starting")
        if len(self.word_list) < value:
            self.word_list.clear()
            if isinstance(self.parent, ModelMergedPart):
                word_df = btm.get_top_topic_words(self.parent.parent.model, words_num=value, topics_idx=[self.key-1])
                word_list = word_df.values.tolist()
                prob_list = []
                for word in word_list:
                    word_idx = np.where(self.parent.model.vocabulary_ == word)
                    prob_list.append(self.parent.parent.model.matrix_topics_words_[self.key-1][word_idx][0])
                self.word_list = list(zip(word_list, prob_list))
            else:
                word_df = btm.get_top_topic_words(self.parent.model, words_num=value, topics_idx=[self.key-1])
                word_list = []
                prob_list = []
                for word in word_df.values.tolist():
                    word_idx = np.where(self.parent.model.vocabulary_ == word[0])
                    word_list.append(word[0])
                    prob_list.append(self.parent.model.matrix_topics_words_[self.key-1][word_idx[0]][0])
                self.word_list = list(zip(word_list, prob_list))
        self._word_num = value
        logger.info("Finished")

class NMFTopicPart(TopicPart):
    @property
    def word_num(self):
        return self._word_num
    @word_num.setter
    def word_num(self, value):
        logger = logging.getLogger(__name__+".NMFTopicPart["+str(self.key)+"].word_num")
        logger.info("Starting")
        if len(self.word_list) < value:
            self.word_list.clear()
            if isinstance(self.parent, ModelMergedPart):
                #TODO: test that this works when self.parent is a ModelMergedPart
                components_df = pd.DataFrame(self.parent.parent.model.components_, columns=self.parent.parent.vectorizer.get_feature_names_out())
                topic = components_df.iloc[self.key-1]
                word_prob_list = topic.nlargest(value)
                word_list = word_prob_list.index.tolist()
                prob_list = word_prob_list.tolist()
                self.word_list = list(zip(word_list, prob_list))
            else:
                components_df = pd.DataFrame(self.parent.model.components_, columns=self.parent.vectorizer.get_feature_names_out())
                topic = components_df.iloc[self.key-1]
                word_prob_list = topic.nlargest(value)
                word_list = word_prob_list.index.tolist()
                prob_list = word_prob_list.tolist()
                self.word_list = list(zip(word_list, prob_list))
        self._word_num = value
        logger.info("Finished")


class Top2VecTopicPart(TopicPart):
    @property
    def word_num(self):
        return self._word_num
    
    @word_num.setter
    def word_num(self, value):
        logger = logging.getLogger(__name__ + ".Top2VecTopicPart[" + str(self.key) + "].word_num")
        logger.info("Starting")
        if len(self.word_list) < value:
            self.word_list.clear()
            try:
                # Retrieve the Top2Vec model from the parent object
                top2vec_model = self.parent.parent.model  # Assuming this is how you access the Top2Vec model
                
                # Get topics and keywords from the Top2Vec model
                topics, probs, _ = top2vec_model.get_topics(num_topics=value)
                
                # Iterate over topics and add top words to word_list
                for topic_idx, topic in enumerate(topics):
                    word_prob_list = list(zip(topic, probs[topic_idx]))
                    self.word_list.extend(word_prob_list)
            except Exception as e:
                print("Error:", e)
                logger.error("Error occurred: %s", e)
        self._word_num = value
        logger.info("Finished")


class BertopicTopicPart(TopicPart):
    @property
    def word_num(self):
        return self._word_num
    
    @word_num.setter
    def word_num(self, value):
        logger = logging.getLogger(__name__ + ".BertopicTopicPart[" + str(self.key) + "].word_num")
        logger.info("Starting")
        if len(self.word_list) < value:
            self.word_list.clear()
            try:
                # Retrieve the Bertopic model from the parent object
                bertopic_model = self.parent.parent.model  # Assuming this is how you access the Bertopic model
                
                # Get topics and keywords from the Bertopic model
                topics, probs = bertopic_model.get_topics(num_topics=value)
                
                # Iterate over topics and add top words to word_list
                for topic_idx, topic in enumerate(topics):
                    word_prob_list = list(zip(topic, probs[topic_idx]))
                    self.word_list.extend(word_prob_list)
            except Exception as e:
                print("Error:", e)
                logger.error("Error occurred: %s", e)
        self._word_num = value
        logger.info("Finished")



class TopicUnknownPart(ModelPart):
    '''Instances of Topic Unknown Part objects'''
    def __init__(self, parent, key, word_list, dataset, name="Unknown"):
        ModelPart.__init__(self, parent, key, [], dataset, name=name)
        
        #properties that automatically update last_changed_dt
        self._word_num = 0
        self._word_list = []

    @property
    def word_num(self):
        return self._word_num
    @word_num.setter
    def word_num(self, value):
        self._word_num = 0

    @property
    def word_list(self):
        return self._word_list
    @word_list.setter
    def word_list(self, value):
        _word_list = []

    def GetTopicKeywordsList(self):
        return []

