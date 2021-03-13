from process_json import *
import numpy as np
BS = 32

def create_tags(text, a):
    a = a.lower()
    tags = ['O'] * len(text.split())
    inside = False
    isAnswerFound = False
    w = text.lower().split()
    i = 0
    start_index = 0
    current_entity = []
    
    while i < len(tags):
        w_stripped = w[i].strip()

        if inside:
            if w_stripped.endswith('__end'):
                inside = False
                end_word = w_stripped.split('__')[0]
                current_entity.append(end_word)
                if a in ' '.join(current_entity) or ' '.join(current_entity) in a:
                    isAnswerFound = True
                    tags[start_index] = 'B-ans'
                    tags[start_index+1 : i+1] = ['I-ans'] * ((i+1) - (start_index+1))
                start_index = 0
                current_entity = []   
            else:
                current_entity.append(w_stripped)
            
            i = i + 1  
            continue
        
        if w_stripped== 'beg____end':
            i = i + 1
            continue
            
        if w_stripped.startswith("beg__") and w_stripped.endswith("__end"):
            entity = w_stripped.split("__")[1]
            if entity.lower() == a:
                isAnswerFound = True
                tags[i] = 'B-ans'
                if inside:  # something went wrong, leave as is
                    inside = False
                    print("Inconsistent markup.")
            i = i + 1
            continue
                
        elif w_stripped.startswith("beg__"):
            #assert not inside
            inside = True
            start_index = i
            word = w_stripped.split("__")[-1]
            current_entity.append(word)
            i = i + 1
            
        else:
            i = i + 1

    return tags, isAnswerFound


class JsonData(JsonDataset):
    def __init__(self, dataset_file):
        super().__init__(dataset_file)
        self.dataset_counter = 0

    def json_to_plain(self, remove_notfound=False, stp="no-ent", include_q_cands=False):
        """
                :param stp: no-ent | ent; whether to mark entities in passage; if ent, a multiword entity is treated as 1 token
                :return: {"id": "",
                          "p": "",
                          "q", "",
                          "a", "",
                          "c", [""]}
                """
        count = 0
        for datum in self.dataset[DATA_KEY]:
            for qa in datum[DOC_KEY][QAS_KEY]:
                try:
                    fields = {}
                    qa_txt_option = (" " + qa[QUERY_KEY]) if include_q_cands else ""
                    # cand = [w for w in to_entities(datum[DOC_KEY][TITLE_KEY] + " " +
                    #                               datum[DOC_KEY][CONTEXT_KEY] + qa_txt_option).lower().split() if w.startswith('@entity')]
                    cand = [w for w in to_entities(datum[DOC_KEY][TITLE_KEY] + " " +
                                                datum[DOC_KEY][CONTEXT_KEY]).lower().split() if w.startswith('@entity')]
                    cand_q = [w for w in to_entities(qa_txt_option).lower().split() if w.startswith('@entity')]
                    if stp == "no-ent":
                        c = {ent_to_plain(e) for e in set(cand)}
                        a = ""
                        for ans in qa[ANS_KEY]:
                            if ans[ORIG_KEY] == "dataset":
                                a = ans[TXT_KEY].lower()
                        if remove_notfound:
                            if a not in c:
                                found_umls = False
                                for ans in qa[ANS_KEY]:
                                    if ans[ORIG_KEY] == "UMLS":
                                        umls_answer = ans[TXT_KEY].lower()
                                        if umls_answer in c:
                                            found_umls = True
                                            a = umls_answer
                                # if not found_umls:
                                #     continue
                        fields["c"] = list(c)
                        assert a
                        fields["a"] = a
                        document = remove_entity_marks(
                            datum[DOC_KEY][TITLE_KEY] + " " + datum[DOC_KEY][CONTEXT_KEY]).replace(
                            "\n"," ").lower()
                        doc_tags, isAnsweFound = create_tags(datum[DOC_KEY][TITLE_KEY] + " " + datum[DOC_KEY][CONTEXT_KEY],a)

                        if not isAnsweFound:
                            continue

                        fields["p"] = document
                        assert len(doc_tags)==len(fields["p"].split())
                        fields["p_tags"] = doc_tags
                        fields["q"] = remove_entity_marks(qa[QUERY_KEY]).replace("\n", " ").lower()
                        q_tags, _ = create_tags(qa[QUERY_KEY],a)
                        assert len(q_tags)==len(fields["q"].split())
                        fields["q_tags"] = q_tags
                        self.dataset_counter += 1



                    ####INGORE THIS OPTION, WE ARE ONLY WORKING WITH NO ENT OPTION
                    elif stp == "ent":
                        c = set(cand)
                        c_q = set(cand_q)
                        a = ""
                        for ans in qa[ANS_KEY]:
                            if ans[ORIG_KEY] == "dataset":
                                a = plain_to_ent(ans[TXT_KEY].lower())
                        if remove_notfound:
                            if a not in c:
                                found_umls = False
                                for ans in qa[ANS_KEY]:
                                    if ans[ORIG_KEY] == "UMLS":
                                        umls_answer = plain_to_ent(ans[TXT_KEY].lower())
                                        if umls_answer in c:
                                            found_umls = True
                                            a = umls_answer
                                # if not found_umls:
                                #     continue
                        fields["c"] = list(c) + list(c_q)
                        assert a
                        fields["a"] = a
                        document = to_entities(datum[DOC_KEY][TITLE_KEY] + " " + datum[DOC_KEY][CONTEXT_KEY]).replace(
                            "\n", " ").lower()
                        fields["p"] = document
                        fields["q"] = to_entities(qa[QUERY_KEY]).replace("\n", " ").lower()

                    else:
                        raise NotImplementedError

                    fields["id"] = qa[ID_KEY]

                    yield fields
                except AssertionError as e:
                    # Assertion error for few paragraphs
                    continue

    def get_dataset_counter(self):
        return self.dataset_counter


class MyDataReader():
    def __init__(self,data_path = '/Users/ahmedkoptanmacbook/Imp/ASU/Course Content/Spring 2020/CSE576NLP/Project/clicr_dataset/' + 'dev1.0.json',bs=None):
        self.sample_counter=0
        self.d = JsonData(data_path)
        self.bs = bs

    def send_batches(self,remove_notfound = True):
        data = []
        for i, inst in enumerate(self.d.json_to_plain(remove_notfound=remove_notfound, stp='no-ent')):
            if i>=self.sample_counter:
                data.append(inst)
                self.sample_counter+= 1
                if self.bs!=None:
                    if self.sample_counter % self.bs == 0:
                        return data
        return data
    
    def get_data_size(self):
        return self.d.get_dataset_counter()
    
    

