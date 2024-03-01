import logging 
from datetime import datetime

from Common.Objects.Generic import GenericObject
import Common.Objects.Datasets as Datasets
import Common.Objects.Samples as Samples

class Code(GenericObject):
    def __init__(self, name, parent=None, key=None):
        GenericObject.__init__(self, name=name, parent=parent, key=key)

        self._colour_rgb = (255,255,255,)
        
        self.subcodes = {}
        self.connections = []
        self.doc_positions = {}

        self.quotations = []

    def __repr__(self):
        return 'Code[%s][%s]' % (self.name, self.key,)

    @property
    def colour_rgb(self):
        return self._colour_rgb
    @colour_rgb.setter
    def colour_rgb(self, value):
        self._colour_rgb = value
        self._last_changed_dt = datetime.now()

    @property
    def last_changed_dt(self):
        for subcode_key in self.subcodes:
            tmp_last_changed_dt = self.subcodes[subcode_key].last_changed_dt
            if tmp_last_changed_dt > self._last_changed_dt:
                self._last_changed_dt = tmp_last_changed_dt
        for quotation in self.quotations:
            tmp_last_changed_dt = quotation.last_changed_dt
            if tmp_last_changed_dt > self._last_changed_dt:
                self._last_changed_dt = tmp_last_changed_dt
        return self._last_changed_dt
    @last_changed_dt.setter
    def last_changed_dt(self, value):
        self._last_changed_dt = value
    
    def GetAncestors(self):
        ancestors = []
        if self.parent != None:
            ancestors.append(self.parent)
            ancestors.extend(self.parent.GetAncestors())
        return ancestors
    
    def GetDescendants(self):
        descendants = []
        for subcode in self.subcodes.values():
            descendants.append(subcode)
            descendants.extend(subcode.GetDescendants())
        return descendants

    def AddConnection(self, obj):
        obj_module = getattr(obj, '__module__', None)
        key_path = []
        key_path.append((type(obj), obj.key))
        while obj.parent != None:
            obj = obj.parent
            key_path.append((type(obj), obj.key))
        key_path.reverse()
        self.connections.append((obj_module, key_path))
        self.last_changed_dt = datetime.now()
    
    def RemoveConnection(self, obj):
        obj_module = getattr(obj, '__module__', None)
        key_path = []
        key_path.append((type(obj), obj.key))
        while obj.parent != None:
            obj = obj.parent
            key_path.append((type(obj), obj.key))
        key_path.reverse()
        if (obj_module, key_path) in self.connections:
            self.connections.remove((obj_module, key_path))
            self.last_changed_dt = datetime.now()

    def GetConnections(self, datasets, samples):
        connection_objects = []
        for key_path in reversed(self.connections):
            current_parent = None
            if key_path[0] == Datasets.__name__:
                current_parent = datasets
                for key in key_path[1]:
                    if isinstance(current_parent, dict):
                        if key[1] in current_parent:
                            current_parent = current_parent[key[1]]
                        else:
                            current_parent = None
                            break
                    elif isinstance(current_parent, Datasets.Dataset):
                        if key[0] ==  Datasets.Field:
                            if key[1] in current_parent.available_fields:
                                current_parent = current_parent.available_fields[key[1]]
                            else:
                                current_parent = None
                                break
                        elif key[0] ==  Datasets.Field:
                            if key[1] in current_parent.computational_fields:
                                current_parent = current_parent.computational_fields[key[1]]
                            else:
                                current_parent = None
                                break
                        elif key[0] ==  Datasets.Document:
                            if key[1] in current_parent.documents:
                                current_parent = current_parent.documents[key[1]]
                            else:
                                current_parent = None
                                break
                        else:
                            current_parent = None
                            break
                    else:
                        current_parent = None
                        break
            elif key_path[0] == Samples.__name__:
                current_parent = samples
                for key in key_path[1]:
                    if isinstance(current_parent, dict):
                        if key in current_parent:
                            current_parent = current_parent[key[1]]
                        else:
                            current_parent = None
                            break
                    elif isinstance(current_parent, Samples.Sample):
                        if key[1] in current_parent.parts_dict:
                            current_parent = current_parent.parts_dict[key[1]]
                        else:
                            current_parent = None
                            break
                    elif isinstance(current_parent, Samples.MergedPart):
                        if key[1] in current_parent.parts_dict:
                            current_parent = current_parent.parts_dict[key[1]]
                        else:
                            current_parent = None
                            break
                    else:
                        current_parent = None
                        break
            if current_parent is not None:
                connection_objects.append(current_parent)
            else:
                #remove keypaths that dont exist to cleanup from name changes
                self.connections.remove(key_path)
        return list(reversed(connection_objects))
    
    def DestroyObject(self):
        #any childrens
        for code_key in list(self.subcodes.keys()):
            self.subcodes[code_key].DestroyObject()
        for quotation in reversed(self.quotations):
            quotation.DestroyObject()
        #remove self from parent if any
        if self.parent is not None:
            if self.key in self.parent.subcodes:
                if self.parent.subcodes[self.key] == self:
                    del self.parent.subcodes[self.key]
            self.parent.last_changed_dt = datetime.now()
            self.parent = None

class Quotation(GenericObject):
    def __init__(self, parent, dataset_key, document_key, original_data=None, paraphrased_data=None):
        GenericObject.__init__(self, parent=parent)
        self._dataset_key = dataset_key
        self._document_key = document_key
        self._original_data = original_data
        self._paraphrased_data = paraphrased_data

    def __repr__(self):
        return 'Quotation[%s]' % (str(self.key))

    @property
    def dataset_key(self):
        return self._dataset_key

    @property
    def document_key(self):
        return self._document_key

    @property
    def original_data(self):
        return self._original_data
    @original_data.setter
    def original_data(self, value):
        self._original_data = value
        self.last_changed_dt = datetime.now()
    
    @property
    def paraphrased_data(self):
        return self._paraphrased_data
    @paraphrased_data.setter
    def paraphrased_data(self, value):
        self._paraphrased_data = value
        self.last_changed_dt = datetime.now()
    
    def DestroyObject(self):
        #remove self from parent if any
        if self.parent is not None:
            if self in self.parent.quotations:
                self.parent.quotations.remove(self)
            self.parent.last_changed_dt = datetime.now()
            self.parent = None

class Theme(GenericObject):
    def __init__(self, name, parent=None, key=None):
        GenericObject.__init__(self, name=name, parent=parent, key=key)

        self._colour_rgb = (255,255,255,)

        self.subthemes = {}
        
        self.code_keys = []

        self.quotations = []

    def __repr__(self):
        return 'Theme[%s][%s]' % (self.name, self.key,)

    @property
    def colour_rgb(self):
        return self._colour_rgb
    @colour_rgb.setter
    def colour_rgb(self, value):
        self._colour_rgb = value
        self._last_changed_dt = datetime.now()

    @property
    def last_changed_dt(self):
        for subcode_key in self.subthemes:
            tmp_last_changed_dt = self.subthemes[subcode_key].last_changed_dt
            if tmp_last_changed_dt > self._last_changed_dt:
                self._last_changed_dt = tmp_last_changed_dt
        for quotation in self.quotations:
            tmp_last_changed_dt = quotation.last_changed_dt
            if tmp_last_changed_dt > self._last_changed_dt:
                self._last_changed_dt = tmp_last_changed_dt
        return self._last_changed_dt
    @last_changed_dt.setter
    def last_changed_dt(self, value):
        self._last_changed_dt = value
    
    def GetAncestors(self):
        ancestors = []
        if self.parent != None:
            ancestors.append(self.parent)
            ancestors.extend(self.parent.GetAncestors())
        return ancestors

    def GetDescendants(self):
        descendants = []
        for subtheme in self.subthemes.values():
            descendants.append(subtheme)
            descendants.extend(subtheme.GetDescendants())
        return descendants

    def GetCodes(self, codes):
        included_codes = []
        for key in codes:
            if key in self.code_keys:
                included_codes.append(codes[key])
            included_codes.extend(self.GetCodes(codes[key].subcodes))
        return included_codes
    
    def DestroyObject(self):
        #any childrens
        for theme_key in list(self.subthemes.keys()):
            self.subthemes[theme_key].DestroyObject()
        #remove self from parent if any
        if self.parent is not None:
            if self.key in self.parent.subthemes:
                if self.parent.subthemes[self.key] == self:
                    del self.parent.subthemes[self.key]
            self.parent.last_changed_dt = datetime.now()
            self.parent = None

