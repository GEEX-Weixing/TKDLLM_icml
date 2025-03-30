from text_extract import load_data_txt
from src.data_new import get_data

# dataset_name = 'cora'
def get_text(dataset):
    data, text = load_data_txt(dataset, use_text=True, seed=0)
    if dataset == "cora":
        datas = get_data(data)
        abstracts = text['abs']
        titles = text['title']
        label_names = text['label']
        paper_infos = []
        for i in range(len(abstracts)):
            abstract = abstracts[i]
            title = titles[i]
            paper_info = "Title: {{" + title + "}}. Abstract: {{" + abstract + "}}."
            paper_infos.append(paper_info)
    elif dataset == "pubmed":
        datas = get_data(data)
        abstracts = text['abs']
        titles = text['title']
        label_names = text['label']
        paper_infos = []
        for i in range(len(abstracts)):
            abstract = abstracts[i]
            title = titles[i]
            paper_info = "Title: {{" + title + "}}. Abstract: {{" + abstract + "}}."
            paper_infos.append(paper_info)
    elif dataset == "citeseer":
        datas = get_data(data)
        txts = text['text']
        label_names = text['label']
        paper_infos = []
        for i in range(len(txts)):
            txt = txts[i]
            paper_info = "Here is a technical paper on computer science. Its content is {{" + txt + "}}."
            paper_infos.append(paper_info)
    return datas, paper_infos













