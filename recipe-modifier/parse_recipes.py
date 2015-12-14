import cPickle as pickle
#import stanford_parser_test as stan
from bllipparser.ModelFetcher import download_and_install_model
from bllipparser import RerankingParser
import multiprocessing as mp
import time
import sys
import re


def parseRecipes(all_recipes_file, breakIntoPhrases):
    with open(all_recipes_file,'rb') as f:
        unparsed_recipes = pickle.load(f)

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
        output_file = "pck_dataset/full_sentences%d.p"%cpu
        p = pool.apply_async(parseRecipeProcess, [chunk, output_file, cpu, breakIntoPhrases])
        processes.append(p)
    [p.get() for p in processes]

# def toTxtFileProcess(unparsed_recipes, output_file, txt_file):
    # with open(txt_file, 'wb') as f:
        # for i,recipe in enumerate(unparsed_recipes):
            # if i == 3:
                # return
            # try:
                # reviews = []
                # for review in unparsed_recipes[recipe]['reviews']:
                    # if type(review) == dict:
                        # reviews.append(review['text'])
                    # else:
                        # reviews.append(review)
                # recipe_list = unparsed_recipes[recipe]['instructions']
                # recipe_text = ' '.join(recipe_list)
                # f.write(recipe+'. <RECIPE_TITLE_BREAK>. ')
                # f.write(recipe_text+'. <RECIPE_TEXT_BREAK>. ')
                # for review in reviews:
                    # f.write(review+'. <RECIPE_REVIEW_BREAK>. ')
                # f.write('. <RECIPE_BREAK>. ')
            # except:
                # continue
def parseSentence(sent):
    split_p = re.split(r' and | or | but |, | thought | although', sent)
    phrases = []
    for i,p in enumerate(split_p):
        if len(p.split(' ')) < 5 and len(phrases)>0:
            phrases[-1] = phrases[-1]+', '+p
        else:
            phrases.append(p)
    return phrases

def parseReviewPhrases(reviews,breakIntoPhrases=True):
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
                    if breakIntoPhrases:
                        parsed[-1].extend(parseSentence(sent))
                    else:
                        parsed[-1].append(sent)
                    #if parsed_sent:
                    #    parsed[-1].append(parsed_sent)
    return parsed

def parseRecipePhrases(recipe, breakIntoPhrases=True):
    parsed = []
    for step in recipe:
        for sent in re.split(r'\.|\?|!', step):
            try:
                sent = str(sent)
            except:
                continue
            else:
                if len(sent) > 2:
                    if breakIntoPhrases:
                        parsed.extend(parseSentence(sent))
                    else:
                        parsed.append(sent)
                    # if parsed_sent:
                        # parsed.append(parsed_sent)
    return parsed
def parseRecipeProcess(unparsed_recipes, output_file,cpu, breakIntoPhrases):

    parsed = {}
    for i,recipe in enumerate(unparsed_recipes):
        reviews = []
        for review in unparsed_recipes[recipe]['reviews']:
            if type(review) == dict:
                reviews.append(review['text'])
            else:
                reviews.append(review)
        reviews = parseReviewPhrases(reviews, breakIntoPhrases=breakIntoPhrases)
        recipe_text = parseRecipePhrases(unparsed_recipes[recipe]['instructions'],breakIntoPhrases=breakIntoPhrases)
        parsed[recipe] = {'instructions':recipe_text, 'reviews':reviews}
        # except:
            # continue
        if i%100 == 0:
            with open(output_file, 'wb') as f:
                pickle.dump(parsed, f)

    with open(output_file, 'wb') as f:
        pickle.dump(parsed, f)
if __name__=='__main__':
    parseRecipes('../scraper/pickle_files/full_dataset.p', False)
