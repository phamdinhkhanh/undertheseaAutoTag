import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
import json
import urllib.request
import re
from datetime import datetime
import warnings
import logging
import pickle
import warnings
import pickle
import argparse
import os
from collections import defaultdict
from underthesea import pos_tag, chunk, sentiment
from core_nlp.tokenization import base_tokenizer, dict_models
from hyperparameter import Hyper
# https://www.dataquest.io/blog/python-json-tutorial

# LINK = 'http://10.220.92.51:8983/solr/master_adayroi_PriceRow_default/select?fl=productNameForSearch_text_vi,code_string,brandName_text_vi_mv&fq=(catalogId:%22adayroiProductCatalog%22%20AND%20catalogVersion:%22Online%22)&fq=isValidPrice_boolean:true&fq=isonline_boolean:true&indent=on&q=*:*&wt=json'
# LINK = 'http://solrslave01.adayroi.com/solr/master_adayroi_PriceRow_default/select?fl=code_string,categoryNameForSearch_text_vi_mv,categoryName_text_vi_mv,description_text_vi,productNameForSearch_text_vi,brandName_text_vi_mv&fq=(catalogId:%22adayroiProductCatalog%22%20AND%20catalogVersion:%22Online%22)&fq=isValidPrice_boolean:true&fq=isonline_boolean:true&indent=on&q=*:*&wt=json'

# warnings.simplefilter('ignore')
logging.basicConfig(format='%(asctime)s: %(levelname)s : %(message)s', level = logging.INFO)
# LINK = 'http://10.220.92.51:8983/solr/master_adayroi_PriceRow_default/select?fl=productNameForSearch_text_vi,code_string,brandName_text_vi_mv&fq=(catalogId:%22adayroiProductCatalog%22%20AND%20catalogVersion:%22Online%22)&fq=isValidPrice_boolean:true&fq=isonline_boolean:true&indent=on&q=*:*&wt=json'

with open(Hyper.CONFIGFILE, 'rb') as fp:
    config = json.loads(fp.read())
print('config sentiments: ', config['link_sentiment'])

parser = argparse.ArgumentParser(description='Tagging Adayroi onsite production')
parser.add_argument('--link', default=config['link'], help='link to clone data from solr')
parser.add_argument('--num_product', default=config['num_product'], type=int, help='number of product')
parser.add_argument('--link_old', default=config['link_old'], help='link tagging product previous day')
parser.add_argument('--link_new', default=config['link_new'], help='link tagging product today')
parser.add_argument('--link_output', default=config['link_output'], help='link tagging product with two information productCode and tag_list')
parser.add_argument('--link_sentiment', default=config['link_sentiment'], help='link json file include good sentiment code')
parser.add_argument('--link_cate', default=config['link_cate'], help='link json file include good sentiment code')
args = parser.parse_args()


class IOObject(object):
    def __init__(self, filename):
        self.filename = filename
        
    def _save_pickle(self, obj):
        with open(self.filename, "wb") as fp:   #Pickling
            pickle.dump(obj, fp)

    def _open_pickle(self):
        with open(self.filename, "rb") as fp:   #Unpickling
            obj = pickle.load(fp)
        return obj

    def _save_json(self, obj):
        with open(self.filename, "wb") as fp:
            json.dump(obj, fp)

    def _open_json(self):
        with open(self.filename, "rb") as fp:
            obj = json.loads(fp.read())
        return obj


class TagProductAll(object):
    def __init__(self, link):
        self.link = link
        self.dictionary = self._read_solr()
        self.sentiments = IOObject(args.link_sentiment)._open_json()

    # Đọc dữ liệu từ solr
    def _read_solr(self):
        response1 = urllib.request.urlopen(self.link)
        dictionary1 = json.loads(response1.read())
        no_product = dictionary1['response']['numFound']
        # no_product = args.num_product
        logging.info('number of product get: {}'.format(no_product))
        self.link = self.link + '&rows=' + str(no_product)
        response2 = urllib.request.urlopen(self.link)
        dictionary2 = json.loads(response2.read())
        self.cate_map_level = self._read_cate(args.link_cate)
        logging.info('completed read link!')
        return dictionary2
    
    # Đọc các mô tả tốt
    def _read_sentiments(self):
        sentiments = json.loads(args.link_sentiments)
        return sentiments

    # Đọc dữ liệu cateName
    def _read_cate(self, cate_path=args.link_cate):
        datacate = pd.read_excel(cate_path, sheet_name='displaycate', header=0)
        cate_map_level = dict()
        for i, row in datacate[['LVL', 'CATENAME']].iterrows():
            cate_map_level[row[1]] = row[0]
        cate_map_level['Danh mục hiển thị'] = 0
        return cate_map_level

    # Lọc ra những cate cấp từ 3-4
    def _filter_cate(self, texts):
        return [text for text in texts if self.cate_map_level[text] in (3, 4)]

    # Lấy cate cuối cùng trong danh sách list các cate
    def _last_child_cate(self, texts):
        return sorted([(self.cate_map_level[text], text) for text in texts], key=lambda x: x[0], reverse=True)[0][1]

    # Tạo bảng dữ liệu gồm productCode, productName, tags
    def _create_dataset(self):
        productName = [p['productNameForSearch_text_vi'] for p in self.dictionary['response']['docs']]
        productCode = [p['code_string'] for p in self.dictionary['response']['docs']]
        cateName = []
        for p in self.dictionary['response']['docs']:
            try:
                cateName.append(self._last_child_cate(p['categoryName_text_vi_mv']))
            except:
                cateName.append('')

        branchName = []
        for p in self.dictionary['response']['docs']:
            try:
                branchName.append(p['brandName_text_vi_mv'][0])
            except:
                branchName.append('')

        description = []
        for p in self.dictionary['response']['docs']:
            try:
                description.append(p['description_text_vi'])
            except:
                description.append('')

        # drop duplicate productCode
        if os.path.exists(args.link_old):
            #Đọc file cũ
            dataset_old = pd.read_csv(args.link_old)
            #Đọc file mới
            dataset_update = pd.DataFrame({'productCode': productCode, 'productName': productName, 
                                'cateName': cateName, 'branchName': branchName,
                                'description': description})
            dataset_update.drop_duplicates(subset=['productCode'], inplace=True)
            #Danh sách các sản phẩm chung
            productCodeCommon = pd.merge(dataset_old, dataset_update, on = 'productCode')['productCode'].tolist()
            #Lấy những sản phẩm mới
            dataset = dataset_update[~dataset_update['productCode'].isin(productCodeCommon)]
            dataset.set_index(np.arange(dataset.shape[0]))
            logging.info('Dataset shape: {}'.format(dataset.shape))
        else:
            dataset = pd.DataFrame({'productCode': productCode, 'productName': productName, 
                                'cateName': cateName, 'branchName': branchName,
                                'description': description})
            dataset.drop_duplicates(subset=['productCode'], inplace=True)
            dataset.set_index(np.arange(dataset.shape[0]))
            logging.info('Dataset shape: {}'.format(dataset.shape))

        dataset = self._update_sentiments(dataset)
        # Cập nhật tag 1
        dataset['tag1'] = [self._gen_tag1(item) for k, item in dataset[['cateName', 'sentiments']].iterrows()]
        # Cập nhật tag 2
        dataset = self._gen_tag2(dataset)
        # cập nhật tag 3
        dataset['tag3'] = self._gen_tag3(dataset)
        # cập nhật tag 4
        dataset['tag4'] = self._gen_tag4(dataset)
        # combine các tags
        dataset = self._combine_tags(dataset)
        logging.info('Completed tag_list!')
        return dataset

    # Tạo từ điển mô tả các vị trí của từ trong câu
    # Ví dụ: 'Trời ơi thật là một ngày đẹp trời. Kết quả trả về bao gồm {'trời': [0, 7], 'ơi':[1], 'thật':[2], 'là':[3], 'một':[4], 'ngày':[5], 'đẹp':[6]}
    def _gen_dict_desc(self, text):
        desc_dict = defaultdict()
        for k, item in enumerate(text.split()):
            if item in desc_dict:
                desc_dict[item].append(k)
            else:
                desc_dict[item] = [k]
        return desc_dict

    # Tính toán khoảng cách nhỏ nhất của 2 cụm vị trí
    # Ví dụ: 'trời xanh mây xanh' sau khi tạo từ điển vị trí sẽ là {'trời':[0], 'xanh':[1, 3], 'mây': [2]}
    # Đo khoảng cách giữa từ 'xanh' và 'trời' sẽ là 1. Mục đích là để tìm các từ có thể ghép cặp với nhau trong 1 khoảng cách nhất định nếu việc tìm từ đó liền kề nhau là không thể.
    # VD: 'trong và sạch' có thể được ghép cặp coi như từ 'trong sạch' bằng cách so khớp khoảng cách.
    def _cross_distance(self, first_list, second_list):
        return np.min([(y - x) for x in first_list for y in second_list if (y - x) >= 0])

    def _check_collocation(self, first, last, desc_dict, min_distance=2):
        try:
            distance = self._cross_distance(desc_dict[first], desc_dict[last])
        except:
            return False
        if (distance <= min_distance) & (distance >= 0):
            return True
        else:
            return False

    # print(_check_collocation('cho', 'tươi', desc_dict = desc_dict))

    # Loại bỏ các kí tự đặc biệt bằng khoảng trắng
    def _clean_text(self, text, regex=r'[-\(\)\"#/@;:<>{}\[\]`+=~%$&|.*!?,|\u200b]'):
        text = re.sub(regex, r' ', text.lower())
        text = re.sub(r'\s+', r' ', text)
        return text

    # Kiểm tra một cụm từ có trong mô tả hay không với khoảng cách là 2
    def _check_desc(self, search, description, min_distance=2):
        words = search.split()
        description = self._clean_text(description)
        desc_dict = self._gen_dict_desc(description)
        if len(words) == 1:
            if not words[0] in desc_dict:
                return False
        else:
            for i in range(len(words) - 1):
                if not self._check_collocation(words[i], words[i + 1], desc_dict, min_distance):
                    return False
        return True

    # Lấy ra các comment tốt
    def _extract_good_sents(self, text):
        return list(set([item for item in self.sentiments if self._check_desc(item.lower(), text.lower())]))


    # _check_desc('internet', dataset['description'][0])

    # Cập nhật trường comment vào dataset
    def _update_sentiments(self, dataset):
        sentiments = []
        start = datetime.now()
        for i, row in dataset[['productName', 'description']].iterrows():
            if (i % 1000 == 0) & (i != 0):
                end = datetime.now()
                print('epochs: {}, time executing: {}'.format(int(i / 1000), end - start))
                start = end
            sentiments_prod = []
            sentiments_des = []
            try:
                sentiments_prod.extend(self._extract_good_sents(row['productName']))
            except:
                sentiments_prod.append('')

            try:
                sentiments_des.extend(self._extract_good_sents(row['description']))
            except:
                sentiments_des.append('')
            sentiments_common = list(set(sentiments_prod).union(set(sentiments_des)))
            sentiments.append(sentiments_common)
        dataset['sentiments'] = sentiments
        return dataset


    # I. tag 1: Kết hợp giữa cateName và description =======================================================================================================
    # Ví dụ: cateName = 'điện thoại', description = 'màu xanh' ==> 'điện thoại màu xanh'
    def _gen_tag1(self, cate_sent):
        cate_sent_tags = []
        cateName = cate_sent['cateName'].lower()
        for sent in cate_sent['sentiments']:
            if sent not in cateName:
                cate_sent_tags.append(' '.join([cateName, sent]))
        return cate_sent_tags


    # II. tag 2: Tên sản phẩm chính + description ==========================================================================================================
    # Ví dụ: tên sản phẩm chính = 'điện thoại smartphone', description = 'màu xanh' ==> 'điện thoại smartphone màu xanh'
    def _gen_tag2(self, dataset):
        dataset['productNameClean'] = [self._clean_text(item) for item in dataset['productName']]
        dataset['productNameClean'] = [self._clean_text(item) for item in dataset['productName']]
        lm_tokenizer = dict_models.LongMatchingTokenizer()
        tok_core_nlp = [lm_tokenizer.tokenize(item) for item in dataset['productNameClean']]
        dataset['tok_core_nlp'] = tok_core_nlp
        # pos tag tên sản phẩm
        # mô tả các nhãn post: https://github.com/undertheseanlp/underthesea/wiki/M%C3%B4-t%E1%BA%A3-d%E1%BB%AF-li%E1%BB%87u-b%C3%A0i-to%C3%A1n-POS-Tag
        dataset['pos_tag'] = [pos_tag(item) for item in dataset['productNameClean']]
        # Tìm danh từ đầu trong tag
        dataset['min_pos_tag_N'] = [self._find_position_N(item) for item in dataset['pos_tag']]
        # Xác định tên sản phẩm
        dataset['main_product_N'] = [self._main_product_N(item['tok_core_nlp'], item['min_pos_tag_N']) for k, item in
                                     dataset[['tok_core_nlp', 'min_pos_tag_N']].iterrows()]
        # Xác định vị trí tên sản phẩm chính
        dataset['len_main_product_N'] = [len(item.split()) for item in dataset['main_product_N']]
        # Danh sách các danh từ đặc biệt là những từ có độ dài 1.
        self.special_N = set(dataset[dataset['len_main_product_N'] == 1]['main_product_N'])
        dataset['min_pos_tag_remove_short_N'] = [self._find_position_remove_short_N(item)
                                                 for item in dataset['pos_tag']]
        dataset['main_product_remove_short_N'] = [self._main_product_N(item['tok_core_nlp'], item['min_pos_tag_remove_short_N'])
                                                  for k, item in dataset[['tok_core_nlp', 'min_pos_tag_remove_short_N']].iterrows()]
        dataset['tag2'] = [self._gen_tag_main_product_desc(item)
                           for k, item in dataset[['main_product_remove_short_N', 'sentiments']].iterrows()]
        return dataset

    # Tìm vị trí đầu tiên mà từ bắt đầu là danh từ trong thuật toán pos_tag
    def _find_position_N(self, pos_tag):
        pos = 0
        for item in enumerate(pos_tag):
            # nếu là danh từ đầu tiên thì trả về pos
            if ('N' in item[1]):
                return pos
            else:
                pos += len(str(item[0]).split())
        return pos


    # Xuất phát từ danh từ trọng tâm (tên sản phẩm) thường đứng đầu nên ta sẽ ghép các từ từ tok_core_nlp tại vị trí từ min_pos_tag_N đổ về trước
    def _main_product_N(self, tokenization, min_pos_tag_N=0):
        pos = 0
        for i, item in enumerate(tokenization):
            if pos > min_pos_tag_N:
                return re.sub('_', ' ', ' '.join(tokenization[:i]))
            else:
                pos += len(item.split('_'))
        return re.sub('_', ' ', ' '.join(tokenization[:1]))

    # Loại bỏ những main product chỉ có độ dài là 1.
    def _find_position_remove_short_N(self, pos_tag):
        pos = 0
        for i, item in enumerate(pos_tag):
            # Nếu từ đầu tiên là danh từ và nằm trong danh sách special_N chọn từ tiếp theo (tiếp tục vòng lặp)
            if (i == 0) & ('N' in item[1]) & (item[0] in self.special_N):
                next
            # Nếu từ tiếp theo là danh từ trả về kết quả
            elif ('N' in item[1]):
                pos += len(item[0].split())
                return pos
            # Nếu không thì cộng vào độ dài của các từ liền trước
            else:
                pos += len(item[0].split())
        return pos

    # Tạo tag 2: Kết hợp giữa main_product_remove_short_N và description
    def _gen_tag_main_product_desc(self, main_product_sent):
        main_product_sent_tags = []
        for sent in main_product_sent['sentiments']:
            if sent not in main_product_sent['main_product_remove_short_N']:
                main_product_sent_tags.append(' '.join([main_product_sent['main_product_remove_short_N'], sent]))
        return main_product_sent_tags

    # III. tag 3: kết hợp giữa cateName và BranchName ==========================================================================================================
    # Ví dụ: cateName = 'điện thoại', branchName = 'samsung'
    # cateName + branchName = 'điện thoại samsung'
    def _gen_tag3(self, dataset):
        tag3 = []
        for k, (cateName, branch_name) in dataset[['cateName', 'branchName']].iterrows():
            branch_name = str(branch_name).lower()
            cateName = cateName.lower()
            if branch_name in cateName:
                tag3.append(cateName)
            elif cateName == '':
                tag3.append('')
            else:
                # Loại bỏ branch name nếu như đã xuất hiện trong cateName.
                tag = ' '.join([cateName, branch_name]).split()
                tag = ' '.join(sorted(set(tag), key=tag.index))
                tag3.append(tag)
        return tag3

    # IV. tag 4: tên sản phẩm chính + branchName =================================================================================================
    # Ví dụ: tên sản phẩm chính = 'điện thoại smartphone' + branchName = 'samsung' ==> 'điện thoại smartphone samsung'
    def _gen_tag4(self, dataset):
        tag4 = []
        for k, (main_product, branch_name) in dataset[['main_product_remove_short_N', 'branchName']].iterrows():
            branch_name = str(branch_name).lower()
            if branch_name in main_product:
                tag4.append(main_product)
            else:
                # remove duplicate word
                tag = ' '.join([main_product, branch_name]).split()
                tag = ' '.join(sorted(set(tag), key=tag.index))
                tag4.append(tag)
        return tag4

    # Kết hợp toàn bộ các tag1 - tag4
    def _combine_tags(self, dataset):
        keyword_suggestions = []
        for k, row in dataset[['tag2', 'tag3', 'tag4']].iterrows():
            row_kw = row['tag2']
            row_kw += [row['tag3']]
            row_kw += [row['tag4']]
            keyword_suggestions.append(row_kw)
        keyword_suggestions = [set(item) for item in keyword_suggestions]
        dataset['tags'] = [[item for item in tags] for tags in keyword_suggestions]
        return dataset


tagProduct = TagProductAll(args.link)
result = tagProduct._create_dataset()
result.to_csv(args.link_new)
result = result[['productCode', 'productName','tags']]
if os.path.exists(args.link_old):
    result_old = pd.read_csv(args.link_old, header = 0, index_col = 0)
    result = pd.concat([result_old, result], axis = 0)
    result.index = np.arange(result.shape[0])
    result.to_csv(args.link_output)
    result.to_csv(args.link_old)
else:
    result.to_csv(args.link_old)
