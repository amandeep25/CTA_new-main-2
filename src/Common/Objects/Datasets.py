import logging
from datetime import datetime

import Common.Constants as Constants
from Common.Objects.Generic import GenericObject
import Common.Objects.Samples as Samples

#Object classes to facilitate controlling datasets
class Dataset(GenericObject):
    '''instances of Datasets.'''
    def __init__(self, name, dataset_source, dataset_type, language, retrieval_details):
        GenericObject.__init__(self, name=name)
        
        #properties that automatically update last_changed_dt
        self._dataset_source = dataset_source
        self._dataset_type = dataset_type
        self._retrieval_details = retrieval_details
        self._data = {}
        self._language = language
        self._tokenization_choice = 2
        self._tokenization_package_versions = None
        self._total_docs = 0
        self._total_tokens = 0
        self._total_uniquetokens = 0
        self._total_docs_remaining = 0
        self._total_tokens_remaining = 0
        self._total_uniquetokens_remaining = 0

        #properties that when modfied need to have changed date manual updated
        self.filter_rules = []
        self.filter_rules.append((Constants.FILTER_RULE_ANY, Constants.FILTER_RULE_ANY, 'X', Constants.FILTER_RULE_REMOVE))
        self.filter_rules.append((Constants.FILTER_RULE_ANY, Constants.FILTER_RULE_ANY, 'SPACE', Constants.FILTER_RULE_REMOVE))
        self.filter_rules.append((Constants.FILTER_RULE_ANY, Constants.FILTER_RULE_ANY, 'PUNCT', Constants.FILTER_RULE_REMOVE))
        self.filter_rules.append((Constants.FILTER_RULE_ANY, Constants.FILTER_RULE_ANY, 'NUM', Constants.FILTER_RULE_REMOVE))
        self.filter_rules.append((Constants.FILTER_RULE_ANY, Constants.FILTER_RULE_ANY, 'SYM', Constants.FILTER_RULE_REMOVE))
        self.filter_rules.append((Constants.FILTER_RULE_ANY, Constants.FILTER_RULE_ANY, Constants.FILTER_RULE_ANY, Constants.FILTER_RULE_REMOVE_SPACY_AUTO_STOPWORDS))
        self.filter_rules.append((Constants.FILTER_RULE_ANY, Constants.FILTER_RULE_ANY, Constants.FILTER_RULE_ANY, (Constants.FILTER_TFIDF_REMOVE, Constants.FILTER_TFIDF_LOWER, 75.0)))

        #objects that have their own last_changed_dt and thus need to be checked dynamically
        self.available_fields = {}
        self.label_fields = {}
        self.computational_fields = {}
        self.selected_documents = []
        self.documents = {}

    def __repr__(self):
        return 'Dataset[%s][%s]' % (self.name, self.key,)

    @property
    def dataset_source(self):
        return self._dataset_source
    @dataset_source.setter
    def dataset_source(self, value):
        self._dataset_source = value
        self.last_changed_dt = datetime.now()

    @property
    def dataset_type(self):
        return self._dataset_type
    @dataset_type.setter
    def dataset_type(self, value):
        self._dataset_type = value
        self.last_changed_dt = datetime.now()

    @property
    def retrieval_details(self):
        return self._retrieval_details
    @retrieval_details.setter
    def retrieval_details(self, value):
        self._retrieval_details = value
        self.last_changed_dt = datetime.now()
    
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, value):
        self._data = value
        self.last_changed_dt = datetime.now()

    @property
    def language(self):
        return self._language
    @language.setter
    def language(self, value):
        self._language = value
        self.last_changed_dt = datetime.now()

    @property
    def tokenization_choice(self):
        if not hasattr(self, '_tokenization_choice'):
            self._tokenization_choice = 0
        return self._tokenization_choice
    @tokenization_choice.setter
    def tokenization_choice(self, value):
        self._tokenization_choice = value
        for field in self.computational_fields:
            self.computational_fields[field].tokenization_choice = value
        self.last_changed_dt = datetime.now()
    
    @property
    def tokenization_package_versions(self):
        return self._tokenization_package_versions
    @tokenization_package_versions.setter
    def tokenization_package_versions(self, value):
        self._tokenization_package_versions = value
        self.last_changed_dt = datetime.now()
    
    @property
    def total_docs(self):
        return self._total_docs
    @total_docs.setter
    def total_docs(self, value):
        self._total_docs = value
        self.last_changed_dt = datetime.now()
    
    @property
    def total_tokens(self):
        return self._total_tokens
    @total_tokens.setter
    def total_tokens(self, value):
        self._total_tokens = value
        self.last_changed_dt = datetime.now()
    
    @property
    def total_uniquetokens(self):
        return self._total_uniquetokens
    @total_uniquetokens.setter
    def total_uniquetokens(self, value):
        self._total_uniquetokens = value
        self.last_changed_dt = datetime.now()
    
    @property
    def total_docs_remaining(self):
        return self._total_docs_remaining
    @total_docs_remaining.setter
    def total_docs_remaining(self, value):
        self._total_docs_remaining = value
        self.last_changed_dt = datetime.now()
    
    @property
    def total_tokens_remaining(self):
        return self._total_tokens_remaining
    @total_tokens_remaining.setter
    def total_tokens_remaining(self, value):
        self._total_tokens_remaining = value
        self.last_changed_dt = datetime.now()
    
    @property
    def total_uniquetokens_remaining(self):
        return self._total_uniquetokens_remaining
    @total_uniquetokens_remaining.setter
    def total_uniquetokens_remaining(self, value):
        self._total_uniquetokens_remaining = value
        self.last_changed_dt = datetime.now()

    @property
    def last_changed_dt(self):
        for key in self.available_fields:
            tmp_last_changed_dt = self.available_fields[key].last_changed_dt
            if tmp_last_changed_dt > self._last_changed_dt:
                self._last_changed_dt = tmp_last_changed_dt
        for key in self.computational_fields:
            tmp_last_changed_dt = self.computational_fields[key].last_changed_dt
            if tmp_last_changed_dt > self._last_changed_dt:
                self._last_changed_dt = tmp_last_changed_dt
        for key in self.documents:
            tmp_last_changed_dt = self.documents[key].last_changed_dt
            if tmp_last_changed_dt > self._last_changed_dt:
                self._last_changed_dt = tmp_last_changed_dt
        return self._last_changed_dt
    @last_changed_dt.setter
    def last_changed_dt(self, value):
        self._last_changed_dt = value
    
    def DestroyObject(self):
        #any children Fields
        for key in list(self.available_fields.keys()):
            self.available_fields[key].DestroyObject()
        #make sure that symbolic connections are purged
        self.computational_fields.clear()
        self.label_fields.clear()
        #any children Documents:
        for document_key in list(self.documents.keys()):
            self.documents[document_key].DestroyObject()
        #remove self from parent if any
        if self.parent is not None:
            if self.key in self.parent.datasets:
                if self.parent.datasets[self.key] == self:
                    del self.parent.datasets[self.key]
            self.parent.last_changed_dt = datetime.now()
            self.parent = None

    def AddFilterRule(self, new_rule):
        self.filter_rules.append(new_rule)
        self._last_changed_dt = datetime.now()

    def GetDocument(self, doc_id):
        if doc_id in self._data:
            new_doc = None
            for doc in self.documents.values():
                if doc.doc_id == doc_id:
                    new_doc = doc
                    break
            if new_doc == None:
                new_doc = Document(self, doc_id)
                self.documents[new_doc.key] = new_doc
            return self.documents[new_doc.key]
        else:
            return None

class Field(GenericObject):
    '''instances of Fields.'''
    def __init__(self, parent, name, dataset, desc, fieldtype):
        GenericObject.__init__(self, parent=parent, name=name)
        
        #properties that automatically update last_changed_dt
        self._dataset = dataset
        self._desc = desc
        self._fieldtype = fieldtype
        self._tokenization_choice = 0
        self._tokenset = None
        self._included_tokenset_df = None

    def __repr__(self):
        return 'Field[%s][%s]' % (self.name, self.key,)
    
    @property
    def dataset(self):
        return self._dataset
    @dataset.setter
    def dataset(self, value):
        self._dataset = value
        self.last_changed_dt = datetime.now()
    
    @property
    def desc(self):
        return self._desc
    @desc.setter
    def desc(self, value):
        self._desc = value
        self.last_changed_dt = datetime.now()

    @property
    def fieldtype(self):
        return self._fieldtype
    @fieldtype.setter
    def fieldtype(self, value):
        self._fieldtype = value
        self.last_changed_dt = datetime.now()

    @property
    def tokenization_choice(self):
        return self._tokenization_choice
    @tokenization_choice.setter
    def tokenization_choice(self, value):
        self._tokenization_choice = value
        self.last_changed_dt = datetime.now()

    @property
    def tokenset(self):
        return self._tokenset
    @tokenset.setter
    def tokenset(self, value):
        self._tokenset = value
        self.last_changed_dt = datetime.now()
    
    @property
    def included_tokenset_df(self):
        return self._included_tokenset_df
    @included_tokenset_df.setter
    def included_tokenset_df(self, value):
        self._included_tokenset_df = value
        self.last_changed_dt = datetime.now()
    
    def DestroyObject(self):
        #remove self from Dataset
        if self.dataset != None:
            if self in self.dataset.computational_fields.values():
                if self.dataset.computational_fields[self.key] == self:
                    del self.dataset.computational_fields[self.key]
            if self in self.dataset.available_fields.values():
                if self.dataset.available_fields[self.key] == self:
                    del self.dataset.available_fields[self.key]
            if self in self.dataset.label_fields.values():
                if self.dataset.label_fields[self.key] == self:
                    del self.dataset.label_fields[self.key]
            self.dataset.last_changed_dt = datetime.now()
            self.dataset = None
        if self.parent != None:
            self.parent.last_changed_dt = datetime.now()
            self.parent = None

class Document(GenericObject):
    '''instances of Document.'''
    def __init__(self, parent, doc_id):
        GenericObject.__init__(self, parent=parent)

        #properties that automatically update last_changed_dt
        self._doc_id = doc_id
        if 'url' in parent.data[doc_id]:
            self._url = parent.data[doc_id]['url']
        else:
            self._url = ''
        
        #dictionary that is managed with setters
        #self.data_dict = {}
        self.sample_connections = []

    def __repr__(self):
        return 'Document[%s][%s]' % (str(self.doc_id), self.key, )
    
    @property
    def doc_id(self):
        return self._doc_id
    @doc_id.setter
    def doc_id(self, value):
        self._doc_id = value
        self.last_changed_dt = datetime.now() 

    @property
    def url(self):
        return self._url
    @url.setter
    def url(self, value):
        self._url = value
        self.last_changed_dt = datetime.now() 

    def DestroyObject(self):
        if self.parent is not None:
            if self.key in self.parent.documents:
                if self.parent.documents[self.key] == self:
                    del self.parent.documents[self.key]
            self.parent.last_changed_dt = datetime.now()
            self.parent = None

    #def SetDocumentFields(self):
    #    for field_key in self.parent.computational_fields:
    #        if field_key in self.parent.data[self.key]:
    #            self.data_dict[field_key] = self.parent.data[self.key][field_key]
    #    self.last_changed_dt = datetime.now()

    def AddSampleConnections(self, obj):
        obj_module = getattr(obj, '__module__', None)
        key_path = []
        key_path.append((type(obj), obj.key))
        while obj.parent != None:
            obj = obj.parent
            key_path.insert(0, (type(obj), obj.key))
        self.sample_connections.append((obj_module, key_path))
        self.last_changed_dt = datetime.now()

    def RemoveSampleConnections(self, obj):
        obj_module = getattr(obj, '__module__', None)
        key_path = []
        key_path.append((type(obj), obj.key))
        while obj.parent != None:
            obj = obj.parent
            key_path.insert(0, (type(obj), obj.key))
        self.sample_connections.remove((obj_module, key_path))
        self.last_changed_dt = datetime.now()

    def GetSampleConnections(self, samples, selected=False):
        connection_objects = []
        for key_path in self.sample_connections:
            if key_path[0] == Samples.__name__:
                objects = []
                current_selected = False
                current_parent = samples
                for key in key_path[1]:
                    if isinstance(current_parent, dict):
                        if key[1] in current_parent:
                            current_parent = current_parent[key[1]]
                            if current_parent.selected:
                                current_selected = True
                            elif hasattr(current_parent, "selected_documents") and self.key in current_parent.selected_documents:
                                current_selected = True
                            objects.append(current_parent)
                        else:
                            self.sample_connections.remove(key_path)
                            current_parent = None
                            break
                    elif isinstance(current_parent, Samples.Sample):
                        if key[1] in current_parent.parts_dict:
                            current_parent = current_parent.parts_dict[key[1]]
                            if current_parent.selected:
                                current_selected = True
                            objects.append(current_parent)
                        else:
                            self.sample_connections.remove(key_path)
                            current_parent = None
                            break
                    elif isinstance(current_parent, Samples.MergedPart):
                        if key[1] in current_parent.parts_dict:
                            current_parent = current_parent.parts_dict[key[1]]
                            if current_parent.selected:
                                current_selected = True
                            objects.append(current_parent)
                        else:
                            self.sample_connections.remove(key_path)
                            current_parent = None
                            break
                    else:
                        self.sample_connections.remove(key_path)
                        current_parent = None
                        break
                if current_parent is not None:
                    if not selected or (current_selected and selected):
                        connection_objects.append(objects)

        return connection_objects
        
