import cPickle as pickle
#import stanford_parser_test as stan
from bllipparser.ModelFetcher import download_and_install_model
from bllipparser import RerankingParser
import multiprocessing as mp
import time
import sys
import re


def parseRecipes(all_recipes_file):
    with open(all_recipes_file,'rb') as f:
        unparsed_recipes = pickle.load(f)
    # toTxtFileProcess(unparsed_recipes, 'parsed.p', 'tmp.txt')
    onekey = unparsed_recipes.keys()[0]
    oneelt= unparsed_recipes[onekey]
    keys = []
    small_set = {}
    for key in unparsed_recipes.keys()[:25]:
        small_set[key] = unparsed_recipes[key]


    parseRecipeProcess(small_set, 'tmp.txt', 'parsed.p',0)
    return
    cpus = mp.cpu_count()
    chunksize = len(unparsed_recipes)/cpus
    pool = mp.Pool(processes=cpus)
    recipe_keys = unparsed_recipes.keys()
    processes = []
    for cpu in xrange(cpus):
        chunk = {}
        if cpu < cpus-1:
            recipe_key_chunk = recipe_keys[cpu*chunksize:(cpu+1)*chunksize]
        else:
            recipe_key_chunk = recipe_keys[cpu*chunksize:]
        for key in recipe_key_chunk:
            chunk[key] = unparsed_recipes[key]
        output_file = "parsed_%d.p"%cpu
        tmp_file = "tmp_%d.txt"%cpu
        #p = pool.apply_async(toTxtFileProcess, [chunk, output_file, tmp_file])
        p = pool.apply_async(parseRecipeProcess, [chunk, tmp_file, output_file, cpu])
        processes.append(p)
    [p.get() for p in processes]

def toTxtFileProcess(unparsed_recipes, output_file, txt_file):
    with open(txt_file, 'wb') as f:
        for i,recipe in enumerate(unparsed_recipes):
            if i == 3:
                return
            try:
                reviews = []
                for review in unparsed_recipes[recipe]['reviews']:
                    if type(review) == dict:
                        reviews.append(review['text'])
                    else:
                        reviews.append(review)
                recipe_list = unparsed_recipes[recipe]['instructions']
                recipe_text = ' '.join(recipe_list)
                f.write(recipe+'. <RECIPE_TITLE_BREAK>. ')
                f.write(recipe_text+'. <RECIPE_TEXT_BREAK>. ')
                for review in reviews:
                    f.write(review+'. <RECIPE_REVIEW_BREAK>. ')
                f.write('. <RECIPE_BREAK>. ')
            except:
                continue
def parseSentence(sent):
    split_p = re.split(r' and | or | but |, | thought | although', sent)
    phrases = []
    for i,p in enumerate(split_p):
        if len(p.split(' ')) < 5 and len(phrases)>0:
            phrases[-1] = phrases[-1]+', '+p
        else:
            phrases.append(p)
    return phrases

def parseReviewPhrases(reviews):
    parsed = []
    for r in reviews:
        parsed.append([])
        for sent in re.split(r'\.|\?|!|;|:', r):
            try:
                sent = str(sent)
            except:
                continue
            else:
                if len(sent) > 2:
                    parsed[-1].append(parseSentence(sent))
                    #if parsed_sent:
                    #    parsed[-1].append(parsed_sent)
    return parsed

def parseRecipePhrases(recipe):
    parsed = []
    for step in recipe:
        for sent in re.split(r'\.|\?|!', step):
            try:
                sent = str(sent)
            except:
                continue
            else:
                if len(sent) > 2:
                    parsed.append(parseSentence(sent))
                    # if parsed_sent:
                        # parsed.append(parsed_sent)
    return parsed
def parseRecipeProcess(unparsed_recipes, txt_file, output_file,cpu):

    # sys.stdout = open("%d.out"%cpu, "a", buffering=0)
    # sys.stderr = open("%d.out"%cpu, "a", buffering=0)
    # stan.parse(txt_file, output_file,cpu)
    parsed = {}
    begin = time.time()
    for i,recipe in enumerate(unparsed_recipes):
        reviews = []
        for review in unparsed_recipes[recipe]['reviews']:
            if type(review) == dict:
                reviews.append(review['text'])
            else:
                reviews.append(review)
        reviews = parseReviewPhrases(reviews)
        recipe_text = parseRecipePhrases(unparsed_recipes[recipe]['instructions'])
        if i == 0:
            for review in reviews:
                print review
            print recipe_text
        parsed[recipe] = {'instructions':recipe_text, 'reviews':reviews}
        # except:
            # continue
        if i%100 == 0:
            with open(output_file, 'wb') as f:
                pickle.dump(parsed, f)
    end = time.time()
    print 'time per recipe:', (end-begin)/25.

    with open(output_file, 'wb') as f:
        pickle.dump(parsed, f)
if __name__=='__main__':
    parseRecipes('../scraper/pickle_files/full_dataset.p')
