import os
import sys

import re

from typing import Optional

import pandas as pd
import numpy as np


from scipy.spatial.distance import pdist, squareform


from sentence_transformers import SentenceTransformer


class ProseptDescriptionSearcher():

    """
    Searching for a product in the database using existing description queries.
    
    :param mode: Read sql database or CSV file. If None - use sql connection. Options: 'csv' or None.
    :param data_connect_or_path: Connection to database or load data from filepath on disc.
    :param model_name_or_path: If it is a filepath on disc, it loads the model from that path. If it is not a path, it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model from Huggingface models repository with that name.
    :param number_of_matching: Parameter responsible for the number of searches for matching options
    :param cache_embeddings_update: Save product embeddings. Overwrites the current data.
    """

    #region init
    def __init__(self, data_connect_or_path,
                 mode: Optional[str] = None,
                 model_name_or_path: Optional[str] = None,
                 number_of_matching: Optional[int] = 5,
                 cache_embeddings_update: Optional[bool] = False,
                ):
        #Инициализируем модель
        if model_name_or_path is None:
            self.model = SentenceTransformer(os.path.join(os.getcwd(), 'model'))
        else:
            self.model = SentenceTransformer('LaBSE')
        
        self.number_of_matching = number_of_matching
        self.mode = mode
        self.cache_embeddings_update = cache_embeddings_update
        #Загружаем данные исходя из формата 
        try:
            if mode is None:
                self.dealer = pd.read_sql('SELECT * FROM marketing_product_dealer', data_connect_or_path)
                self.price = pd.read_sql('SELECT product_key, product_name, dealer_id from marketing_dealer_price', data_connect_or_path)
                self.product = pd.read_sql('SELECT name from marketing_product', data_connect_or_path)
                self.product_dealer_key = pd.read_sql('SELECT * from marketing_product_dealer_key', data_connect_or_path)
            elif mode == 'csv':
                self.dealer = pd.read_csv(os.path.join(data_connect_or_path,'assets','marketing_dealer.csv'),
                                          sep=';', index_col='id')
                self.price = pd.read_csv(os.path.join(data_connect_or_path,'assets','marketing_dealerprice.csv'),
                                         sep=';', index_col='id')
                self.product = pd.read_csv(os.path.join(data_connect_or_path,'assets','marketing_product.csv'),
                                           sep=';', index_col='id')
                self.product_dealer_key = pd.read_csv(os.path.join(data_connect_or_path,'assets','marketing_productdealerkey.csv'),
                                                      sep=';', index_col='id')
                
            self.clean_dealer_price()
            self.clean_product()
            self.clean_product_dealer_key()
            self.merge_data()
            self.clean_merged_df()
            self.generate_embeddings()
        except:
            print('Не удалось загрузить данные')
            sys.exit(1)
    
    #region clean_methods
    def clean_dealer_price(self):
        self.price.drop_duplicates(subset=['product_name'],
                                   keep='last',
                                   inplace=True)
        if self.mode == 'csv':
            self.price = self.price[['product_key','product_name','dealer_id']]
        
    def clean_product(self):
        self.product.drop_duplicates(subset=['name'], inplace=True)
        self.product.dropna(subset=['name'],inplace=True)
        if self.mode == 'csv':
            self.product = self.product['name']
            
    def clean_product_dealer_key(self):
        self.product_dealer_key.dropna(subset=['product_id'],
                                        inplace=True)
        
    def clean_merged_df(self):
        self.df.drop(columns=['product_key','dealer_id_x','name_x',
                                     'key','dealer_id_y',],
                     inplace=True)
        self.df.rename(columns={'name_y':'original_name'},
                       inplace=True)
        for col in ['product_name','original_name']:
            self.df[f'{col}_normalized'] = self.df[col].apply(ProseptDescriptionSearcher.clean_description)
        
        self.df.drop_duplicates(inplace=True)
    #endregion clean_methods
    
    #region merge_data
    def merge_data(self):
        self.df = self.price.merge(self.dealer,
                                   left_on='dealer_id',
                                   right_index=True)
        self.df = self.df.merge(self.product_dealer_key,
                                left_on='product_key',
                                right_on='key')
        self.df = self.df.merge(self.product,
                                left_on='product_id',
                                right_on='id')
    #endregion merge_data

    #region generate_embeddings
    def generate_embeddings(self):
        try:
            emb = pd.read_csv(f'{os.path.join(os.getcwd(), "cache_embeddings", "unique_original_embeddings.csv")}', index_col='original_name')
            emb['original_name_embeddings'] = emb['original_name_embeddings'].tolist()
            self.df['original_name_embeddings'] = self.df.merge(emb.set_index('original_name',drop=True)['original_name_embeddings'], left_on='original_name',right_index=True, how='left')
            self.df['original_name_embeddings'] = np.where(self.df['original_name_embeddings'].isna(),
                                                           self.model.encode(self.df['original_name_normalized'].values).tolist(),
                                                           self.df['original_name_embeddings'])
        except:
            self.df['original_name_embeddings'] = self.model.encode(self.df['original_name_normalized'].values).tolist()
            self.unique_original_embeddings = self.df.drop_duplicates(subset=['product_id'])[['product_id', 'original_name_embeddings']]

        if self.cache_embeddings_update:
            os.makedirs(os.path.join(os.getcwd(), 'cache_embeddings'), exist_ok=True)
            (
            self.unique_original_embeddings
                .to_csv(f'{os.path.join(os.getcwd(), "cache_embeddings", "unique_original_embeddings.csv")}',
                        index=False)
            ) 

        self.unique_embeddings_matrix = np.stack(self.unique_original_embeddings['original_name_embeddings'].values)
    #endregion generate_embeddings

    #region calculate_recall_at_n
    def recall_at_k(self, predicted, correct,):
        return int(correct in predicted[:self.number_of_matching])
    
    def calculate_recall_at_n(self, number_of_matching: Optional[int] = None):
        if number_of_matching is not None:
            self.number_of_matching = number_of_matching
        self.df['product_name_embeddings'] = self.model.encode(self.df['product_name_normalized'].values).tolist()
        # Ранжирование товаров и подсчет Recall at n
        total_recall = []
        for _, row in self.df.iterrows():
            # Расчет расстояний
            distances = pdist(np.vstack([row['product_name_embeddings'], self.unique_embeddings_matrix]), 'cosine')
            distance_matrix = squareform(distances)[0, 1:]
            top_5_indices = np.argsort(distance_matrix)[:self.number_of_matching]  # Топ-n индексов с наименьшими расстояниями

            # Получаем ID топ-n товаров
            top_5_products_ids = self.unique_original_embeddings.iloc[top_5_indices]['product_id'].values

            # Добавляем recall для текущего товара
            total_recall.append(self.recall_at_k(top_5_products_ids, row['product_id']))


        self.average_recall = np.mean(total_recall)
    #endregion calculate_recall_at_n

    #region predict
    def make_predict(self, descriptions_to_match:list, number_of_matching: Optional[int]):
            """
            Method for finding n product values by description.
            
            :param descriptions_to_match: List with description to find
            :param number_of_matching : Parameter responsible for the number of searches for matching options

            Return
            ------------
            Dictionary with:
            keys : inputs
            values : {'id' : id,
                      'match' : [value,value_2...value_n]}
            """

            self.result_dict = {}
            self.number_of_matching = number_of_matching
            for desc in descriptions_to_match:
                self.result_dict[desc] = {}
                desc_normal = ProseptDescriptionSearcher.clean_description(desc)
                vector = self.model.encode(desc_normal)
                distances = pdist(np.vstack([vector, self.unique_embeddings_matrix]), 'cosine')
                distance_matrix = squareform(distances)[0, 1:]
                top_5_indices = np.argsort(distance_matrix)[:self.number_of_matching]

                top_5_recommendations = self.unique_original_embeddings.iloc[top_5_indices]
                # print(f"\nВыбранный товар: {desc}")
                # print(f'-------------------------------')
                # print(f"Топ-{self.number_of_matching} рекомендованных товаров:")      
                for i, row in top_5_recommendations.iterrows():
                    self.result_dict[desc][f'id_{row["product_id"]}'] = self.df[self.df['product_id'] == row['product_id']]['original_name'].iloc[0]
                    #print(f"ID: {row['product_id']}, Название: {self.df[self.df['product_id'] == row['product_id']]['original_name'].iloc[0]}")
            
            return self.result_dict
            


  
    @staticmethod
    def clean_description(text):
 
        extra_words = (
            r'(?:готовый\sсостав|для|концентрат|просепт|prosept|средство|невымываемый|гелеобразный|канистра|'
            r'чистящее|спрей|универсальный|универсальная|универсальное|пэт|жидкое|моющее|гель|чистки|'
            r'концентрированное|professional|готовое|superrubber)'
        )
        # Приведение текста к нижнему регистру
        text = text.lower()
    
        # Удаление чисел в скобках
        text = re.sub(r'\(\d+\)', ' ', text)    
    
        # Вставка пробелов между кириллицей и латиницей
        text = re.sub(r'(?<=[а-яА-Я])(?=[a-zA-Z])|(?<=[a-zA-Z])(?=[а-яА-Я])', ' ', text)
    
        # Добавление пробелов между цифрами и буквами
        text = re.sub(r'(?<=\d)(?=[а-яА-Яa-zA-Z])', ' ', text)
        text = re.sub(r'(?<=[а-яА-Яa-zA-Z])(?=\d)', ' ', text)
    
        # Удаление диапазонов чисел, разделенных дефисом или двоеточием
        text = re.sub(r'\b\d+(?::\d+)?[-:]\d+(?::\d+)?\b', ' ', text)
    
        # Удаление серийных номеров или артикулов
        text = re.sub(r'\b\d+-\d+[а-яА-Яa-zA-Z]*\b', ' ', text)
    
        # Преобразование объемов из литров в миллилитры и веса из кг в граммы
        text = re.sub(r'(\d+[,.]\d+)\s*л', lambda x: f"{int(float(x.group(1).replace(',', '.')) * 1000)} мл", text)
        text = re.sub(r'(\d+[,.]\d+)\s*кг', lambda x: f"{int(float(x.group(1).replace(',', '.')) * 1000)} г", text)
    
        # Замена "/" и "." меджду слов на пробелы
        text = re.sub(r'[/\.]', ' ', text)
    
        text = text.replace('-', ' ')
    
        # Удаление избыточных слов из списка extra_words
        text = re.sub(extra_words, ' ', text)
    
        # Удаление пунктуационных знаков и специальных символов
        text = re.sub(r'[,"“”/\.()–;]', ' ', text)
    
        # Удаление слова "и"
        text = re.sub(r'\bи\b', ' ', text)
    
        # Удаление дефисов, окруженных пробелами
        text = re.sub(r'\s-\s', ' ', text)
    
        # Удаление всех двойных пробелов и пробелов в начале/конце строки
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
        # Замена фраз
        text = re.sub(r'средство чистящее универсальное prosept\s*(\d+\s*мл)', r'universal spray \1', text)
        text = re.sub(r'средство\s*для\s*уборки\s*после\s*строительства\s*prosept\s*duty\s*extra\s*(\d+\s*л)', r'уборки после строительства duty extra \1', text)
        text = re.sub(r'бань|баниисауны', 'бани сауны', text)
        text = re.sub(r'литров|литра|литр', 'л', text)
        text = re.sub(r'osb\s+proplast', 'osbproplast', text)
        text = re.sub(r'\bкор\b|\bорех\b', 'коричневый', text)
        text = re.sub(r'гриля\s+и\s+духовых\s+шкафов|удаления\s+пригоревших\s+жиров', 'жироудалитель cooky grill gel', text)
        text = re.sub(r'грунт\s+пропиточный', 'грунтовка', text)
        text = re.sub(r'удалитель\s+высолов', 'очиститель фасадов', text)
        text = re.sub(r'с\s+itrus', 'citrus', text)
        text = re.sub(r'\bс\sароматом\b', '', text)
        text = re.sub(r'(\d+)\sгруппа', r'\1-ая группа', text)
        text = re.sub(r'\bi\sгруппа\b', '1-ая группа', text)
        text = re.sub(r'\bii\sгруппа\b', '2-ая группа', text)
        text = re.sub(r'\biii\sгруппа\b', '3-я группа', text)
        text = re.sub(r'\biv\sгруппа\b', '4-ая группа', text)
        text = re.sub(r'\bprof l\b', 'prof', text)
        text = re.sub(r'\b(profi|prof i)\b', 'prof 1-ая группа', text)
        text = re.sub(r'антисептикневымываемый ecoultra зеленыйлес 1 л|антисептик невымываемый eco ultra зеленый лес 1 л', 'антисептик невымываемый ultra 1 л', text)
        text = re.sub(r'столешниц раковин из искусственного камня гранита|гриля духовых шкафов', '', text)
        text = re.sub(r'мыло эконом 5 л|мыло эконом класса без красителей ароматизаторов diona e 5 л|крем мыло эконом класса diona e без цвета запаха|мыло с перламутром без красителей ароматизаторов diona 5 л|мыло diona e без красителей ароматизаторов 5 л', 'мыло diona 5 л', text)
        text = re.sub(r'eco ultra невымываемый антисептик 20 л коричневый|антисептик невымываемый коричневый 5 л', 'антисептик eco ultra коричневый 5 л', text)
        text = re.sub(r'замазка заделки глубоких выбоин трещин', '', text)
        text = re.sub(r'шпаклевка выравнивающая акриловая plastix белая 1 кг', 'шпатлевка plastix 1400 г', text)
        text = re.sub(r'антисептик eco ultra коричневый 5 л|строительный антисептик невымываемый|антисептик eco ultra невымываемый коричневый 5 л|антисептик ecoultra коричневый 5 л|антисептик невымываемый eco ultra коричневый 5 л', 'антисептик невымываемый ultra коричневый 5 л', text)
        text = re.sub(r'антисептик невымываемый ecoultra коричневый 5 л|антисептик невымываемый eco ultra 5 л коричневый|антисептик невымываемый eco ultra 5 л|антисептик невымываемый коричневый eco ultra 5 л', 'антисептик невымываемый ultra коричневый 5 л', text)
        text = re.sub(r'антисептик eco ultra 5 л|антисептик невымываемый зеленый 5 л', 'антисептик невымываемый ultra 5 л', text)
        text = re.sub(r'спрей удаленияжира cookygrill 500 мл|жироудалитель 500 мл спрей', 'cooky grill 550 мл', text)
        text = re.sub(r'белья экзотических цветов|сантехники|гелеобразное усиленного действия удаления ржавчины минеральных отложений', '', text)
        text = re.sub(r'eco ultra невымываемый антисептик 5 л|антисептик невымываемый 5 л|строительный антисептик eco ultra невымываемый 5300 г', 'антисептик невымываемый ultra коричневый 5 л', text)
        text = re.sub(r'огнебиозащита древесины огнебио prof 1-ая группа 2 л', 'огнебиозащита огнебио prof 1-ая группа красный 5 л', text)
        text = re.sub(r'антижук 5 л', 'антисептик против жуков других биопоражении антижук 5 л', text)
        text = re.sub(r'удаления плесени 5 л|строительный антисептик глубокого проникновения 5600 г|удаленияплесени 5 л', 'антисептик против грибка плесени антиплесень 5 л', text)
        text = re.sub(r'profii|prof ii', '2-ая группа', text)
        text = re.sub(r'огнебиозащита огнебио prof 1-ая группа 10 л', 'огнебиозащита 1-ая группа бесцветный 10 кг', text)
        text = re.sub(r'клей герметик теплый шов о 15 мл', 'герметик акриловый цвет коричневый 15 кг', text)
        text = re.sub(r'bio lasur антисептик лессирующий защитно декоративный 9 л', 'антисептик лессирующий bio lasur тик 2700 мл', text)
        text = re.sub(r'уборки после отделочных работ 5 л', 'удаления гипсовой пыли duty white 5 л', text)
        text = re.sub(r'\bк\b', '', text)
        text = re.sub(r'клей герметик теплый шов с 15 мл|герметик акриловый паропроницаемый 15 кг белый', 'герметик акриловый цвет сосна 15 кг', text)
        text = re.sub(r'антисептик против жуков других биопоражении антисептик против жуков других биопоражении антижук 5 л', 'антисептик против жуков 5 л', text)
        text = re.sub(r'огнебиозащита древесины огнебио prof 1 высшая 1 ая группа 16 кг б мешок', 'сухой огнебиозащита мешок 16 кг', text)
        text = re.sub(r'строительный антисептик глубокого проникновения 9900 г', 'антисептик грунт osb base 10 л', text)
        text = re.sub(r'удаленияржавчины bathacid 750 мл|усиленного действия удаления ржавчины минеральных отложений bath acid + цитруса 750 мл', 'bath acid + 750 мл', text)
        text = re.sub(r'строительный антисептик 5600 г', 'антисептик внутренних работ sauna 5 л', text)
        text = re.sub(r'антисептик минер fungi stop конц 1 л', 'удалитель плесени fungi clean 1 л', text)
        text = re.sub(r'огнебиозащита древесины огнебио prof 1 высшая 1 ая группа красно коричневая гот состав 10 л 12 кг', 'огнебиозащита 1-ая группа красный 10 кг', text)
        text = re.sub(r'краска резиновая резиновая до 50° акриловая дисперсия матовое покрытие 3 кг черный', 'краска резиновая черный ral 9004 3 кг', text)
        text = re.sub(r'удаления ржавчины минеральных отложений щадящего действия bath acid 1 л', 'bath acid 1 л', text)
        text = re.sub(r'краска osb 7 кг', 'краска резиновая черный ral 9004 12 кг', text)
        text = re.sub(r'посуды cooky 1 л', 'посуды вручную без запаха cooky 500 мл', text)
        text = re.sub(r'удаленияплесени 10 л', 'антисептик против грибка плесени антиплесень 10 л', text)
        text = re.sub(r'удаления ржавчины минеральных отложений щадящего действия bath acid 5 л|удаления ржавчины минеральных отложений щадящее bath acid 5 л', 'bath acid 5 л', text)
        text = re.sub(r'строительный антисептик отбеливающий 19900 г', 'отбеливатель древесины 50 20 л', text)
        text = re.sub(r'кондиционер ополаскиватель белья crystal rinser лепестки сакуры 5 л', 'кондиционер белья лепестков сакуры crystal rinser 5 л', text)
    
        # Удаление всех двойных пробелов и пробелов в начале/конце строки
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
    
        return text
    #endregion clean_description
