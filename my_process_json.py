from process_json import *
import numpy as np
BS = 32


"""def create_tags(text,a):
    #Text includes entities marked as BEG__w1 w2 w3__END. Transform to a tags list.
    mya = a.lower()
    a = mya.split()
    tags = []
    inside = False
    for w in text.split():
        w_stripped = w.strip()
        if w_stripped== 'BEG____END':
            continue
        if w_stripped.startswith("BEG__") and w_stripped.endswith("__END"):
            concept = w_stripped.split("_")[2]
            if concept.lower() in a:
                tags.append('B-ans')
                if inside:  # something went wrong, leave as is
                    print("Inconsistent markup.")
            else:
                tags.append('O')
        elif w_stripped.startswith("BEG__"):
            assert not inside
            inside = True
            concept = [w_stripped.split("_", 2)[-1]]

        elif w_stripped.endswith("__END"):
            if not inside:
                if w_stripped[:-5].lower() in a:
                    tags.append('I') #might be B
                else:
                    tags.append('O')
            else:
                concept.append(w_stripped.rsplit("_", 2)[0])
                if any(c.lower() in a for c in concept):
                    tags.append('B-ans')
                    for w in concept:
                        tags.append('I-ans')
                    tags.pop(-1)
                else:
                    for w in concept:
                        tags.append('O')
                inside = False
        else:
            if inside:
                concept.append(w_stripped)
            else:
                tags.append('O')

    return ' '.join(tags)"""
    
def create_tags(text,a):
    """
    Text includes entities marked as BEG__w1 w2 w3__END. Transform to a tags list.
    """
    a = a.lower()
    tags = []
    inside = False
    for w in text.split():
        w_stripped = w.strip()
        if w_stripped == 'BEG____END':
            continue
        if w_stripped.startswith("BEG__") and w_stripped.endswith("__END"):
            concept = w_stripped.split("_")[2]
            if a in concept.lower():
                tags.append('B-ans')
                if inside:  # something went wrong, leave as is
                    print("Inconsistent markup.")
            else:
                tags.append('O')
        elif w_stripped.startswith("BEG__"):
            assert not inside
            inside = True
            concept = [w_stripped.split("_", 2)[-1]]

        elif w_stripped.endswith("__END"):
            if not inside:
                if a in w_stripped[:-5].lower():
                    tags.append('I') #might be B
                else:
                    tags.append('O')
            else:
                concept.append(w_stripped.rsplit("_", 2)[0])
                if a in ' '.join(concept).lower():
                    tags.append('B-ans')
                    for w in concept:
                        tags.append('I-ans')
                    tags.pop(-1)
                else:
                    for w in concept:
                        tags.append('O')
                inside = False
        else:
            if inside:
                concept.append(w_stripped)
            else:
                tags.append('O')

    return ' '.join(tags)

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
                            if not found_umls:
                                continue
                    fields["c"] = list(c)
                    assert a
                    fields["a"] = a
                    document = remove_entity_marks(
                        datum[DOC_KEY][TITLE_KEY] + " " + datum[DOC_KEY][CONTEXT_KEY]).replace(
                        "\n"," ").lower()
                    doc_tags = create_tags(datum[DOC_KEY][TITLE_KEY] + " " + datum[DOC_KEY][CONTEXT_KEY],a).split()
                    fields["p"] = document
                    assert len(doc_tags)==len(fields["p"].split())
                    fields["p_beg_end"] = (datum[DOC_KEY][TITLE_KEY] + " " + datum[DOC_KEY][CONTEXT_KEY]).replace(
                        "\n"," ")
                    fields["p_tags"] = doc_tags
                    fields["q"] = remove_entity_marks(qa[QUERY_KEY]).replace("\n", " ").lower()
                    q_tags = create_tags(qa[QUERY_KEY],a).split()
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
                            if not found_umls:
                                continue
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
    