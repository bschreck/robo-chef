import cPickle as pickle
import stanford_parser_test as stan
import multiprocessing as mp
import sys



def parseRecipes(all_recipes_file):
    # with open(all_recipes_file,'rb') as f:
        # unparsed_recipes = pickle.load(f)
    parseRecipeProcess('tmp.txt', 'parsed.p')
    return
    cpus = mp.cpu_count()
    #chunksize = len(unparsed_recipes)/cpus
    pool = mp.Pool(processes=cpus)
    #recipe_keys = unparsed_recipes.keys()
    processes = []
    for cpu in xrange(cpus):
        # chunk = {}
        # if cpu < cpus-1:
            # recipe_key_chunk = recipe_keys[cpu*chunksize:(cpu+1)*chunksize]
        # else:
            # recipe_key_chunk = recipe_keys[cpu*chunksize:]
        # for key in recipe_key_chunk:
            # chunk[key] = unparsed_recipes[key]
        output_file = "parsed_%d.p"%cpu
        tmp_file = "tmp_%d.txt"%cpu
        #p = pool.apply_async(toTxtFileProcess, [chunk, output_file, tmp_file])
        p = pool.apply_async(parseRecipeProcess, [output_file, tmp_file])
        processes.append(p)
    [p.get() for p in processes]

def toTxtFileProcess(unparsed_recipes, output_file, txt_file):
    with open(txt_file, 'wb') as f:
        for i,recipe in enumerate(unparsed_recipes):
            try:
                reviews = []
                for review in unparsed_recipes[recipe]['reviews']:
                    if type(review) == dict:
                        reviews.append(review['text'])
                    else:
                        reviews.append(review)
                recipe_list = unparsed_recipes[recipe]['instructions']
                recipe_text = ' '.join(recipe_list)
                f.write(recipe+'\n<RECIPE_TITLE_BREAK>\n')
                f.write(recipe_text+'\n<RECIPE_TEXT_BREAK>\n')
                for review in reviews:
                    f.write(review+'\n<RECIPE_REVIEW_BREAK>\n')
                f.write('\n<RECIPE_BREAK>\n')
            except:
                continue

def parseRecipeProcess(txt_file, output_file):
    stan.parse(txt_file, output_file)
        # try:
            # reviews = []
            # for review in unparsed_recipes[recipe]['reviews']:
                # if type(review) == dict:
                    # reviews.append(review['text'])
                # else:
                    # reviews.append(review)
            # reviews = parseReviewPhrases(reviews, tmp_file)
            # recipe_text = parseRecipePhrases(unparsed_recipes[recipe]['instructions'], tmp_file)
            # parsed[recipe] = {'instructions':recipe_text, 'reviews':reviews}
        # except:
            # continue
        # if i%100 == 0:
            # with open(label_file, 'wb') as f:
                # pickle.dump(parsed, f)

    # with open(label_file, 'wb') as f:
        # pickle.dump(parsed, f)
if __name__=='__main__':
    parseRecipes('../scraper/pickle_files/full_dataset.p')
