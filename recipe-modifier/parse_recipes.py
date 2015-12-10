import cPickle as pickle
import stanford_parser_test as stan
import multiprocessing as mp
import sys



def parseRecipes(all_recipes_file):
    with open(all_recipes_file,'rb') as f:
        unparsed_recipes = pickle.load(f)
    # toTxtFileProcess(unparsed_recipes, 'parsed.p', 'tmp.txt')
    # parseRecipeProcess('tmp.txt', 'parsed.p',0)
    # return
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
        p = pool.apply_async(toTxtFileProcess, [chunk, output_file, tmp_file])
        #p = pool.apply_async(parseRecipeProcess, [tmp_file, output_file, cpu])
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

def parseRecipeProcess(txt_file, output_file,cpu):
    sys.stdout = open("%d.out"%cpu, "a", buffering=0)
    sys.stderr = open("%d.out"%cpu, "a", buffering=0)
    stan.parse(txt_file, output_file,cpu)
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
