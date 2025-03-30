from llmutils.load_cora import get_raw_text_cora
from llmutils.load_citeseer import get_raw_text_citeseer
from llmutils.load_pubmed import get_raw_text_pubmed
from src.data_new import get_data

def load_data_txt(dataset, use_text=True, seed=0):
    if dataset == "cora":
        data, text = get_raw_text_cora(use_text, seed)
    elif dataset == "pubmed":
        data, text = get_raw_text_pubmed(use_text, seed)
    elif dataset == "citeseer":
        data, text = get_raw_text_citeseer(seed)
    return data, text

def get_text(dataset):
    data, text = load_data_txt(dataset, use_text=True, seed=0)
    if dataset == "cora":
        data = get_data(data)
        abstracts = text['abs']
        titles = text['title']
        label_names = text['label']
        paper_infos = []
        for i in range(len(abstracts)):
            abstract = abstracts[i]
            title = titles[i]
            paper_info = "Here is a technical paper on computer science. Its title is {{" + title + "}}. Its abstract is {{" + abstract + "}}."
            paper_infos.append(paper_info)
    elif dataset == "pubmed":
        data = get_data(data)
        abstracts = text['abs']
        titles = text['title']
        label_names = text['label']
        paper_infos = []
        for i in range(len(abstracts)):
            abstract = abstracts[i]
            title = titles[i]
            paper_info = "Here is a technical paper on computer science. Its title is {{" + title + "}}. Its abstract is {{" + abstract + "}}."
            paper_infos.append(paper_info)
    elif dataset == "citeseer":
        data = get_data(data)
        txts = text['text']
        label_names = text['label']
        paper_infos = []
        for i in range(len(txts)):
            txt = txts[i]
            paper_info = "Here is a technical paper on computer science. Its content is {{" + txt + "}}."
            paper_infos.append(paper_info)
    return data, paper_infos
































