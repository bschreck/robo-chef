import pck_to_txt
import stanford_parser_test as stan
import cPickle as pickle
import os,sys

def loadRecipes(all_recipes_file):
    with open(all_recipes_file,'rb') as f:
        recipes = pickle.load(f)
    # for recipe in recipes:
        # for i,review in enumerate(recipes[recipe]['reviews']):
            # recipes[recipe]['reviews'][i] = review.split('. ')
    # with open(all_recipes_file, 'wb') as f:
        # pickle.dump(recipes, f)
    # return
    labeled_file = 'labels.p'
    labeler(recipes, labeled_file)

def openLabelFile(labeled_file):
    labeled_recipes = None
    try:
        with open(labeled_file, 'rb') as f:
            labeled_recipes = pickle.load(f)
    except:
        return {}
    if labeled_recipes:
        print labeled_recipes
        return labeled_recipes
    return {}
def labeler(recipes, labeled_file):
    labeled_recipes = openLabelFile(labeled_file)
    for r_i, recipe in enumerate(recipes):
        recipe_text = recipes[recipe]['instructions']
        if recipe in labeled_recipes:
            continue
        reviews = []
        for review in recipes[recipe]['reviews']:
            reviews.append(review)
        current_label = {}
        for i,review in enumerate(reviews):
            current_label[i] = {}
            print "CURRENT RECIPE"
            print '\n'
            print "="*50
            for k,step in enumerate(recipe_text):
                print "STEP %d."%k, step
            print "-"*50
            print "REVIEW %d"%i
            for j, phrase in enumerate(review):
                print "\n"
                print "Phrase %d\n>>> \"%s\""%(j, phrase)
                check_refinement = str(raw_input("-->Refinement in phrase? "))
                if check_refinement == "y":
                    while True:
                        indexing = str(raw_input("-->Line + Modification or Insertion? "))
                        if indexing.startswith('q'):
                            pickle.dump( labeled_recipes, open( labeled_file, "wb" ) )
                            sys.exit()
                        else:
                            info = str(indexing[-1])
                            try:
                                indexing = int(indexing[:-1])
                            except:
                                print 'Enter an integer plus m or i'
                            else:
                                if indexing < len(recipe_text):
                                    current_label[i][j] = (indexing,info)
                                    break
                                else:
                                    print 'Integer must be less than length of recipe steps'
                elif check_refinement.startswith('q'):
                    pickle.dump( labeled_recipes, open( labeled_file, "wb" ) )
                    sys.exit()
                else:
                    current_label[i][j] = None
        labeled_recipes[recipe] = current_label

        if r_i%10 == 0:
            pickle.dump( labeled_recipes, open( labeled_file, "wb" ) )

    return labeled_recipes



# for key in recipes:
#     instructions
#     reviews
#     RECIPE:
#         1. ]asdkjsd
#         2. askdjf
#         3. sdf

#     REVIEW 1:
#         1. jaksd
#         2. lkjasdf
#         3, lskdj
#         4. jdkdjf

#     question1: is there a refinement contained in review? y/n
#     question2: index + modification/insertion (3m, 1i, 5m, 5i, etc.):


if __name__ == '__main__':
    dataset = '../scraper/pickle_files/full_dataset.p'
    dataset = '10_recipes.p'
    loadRecipes(dataset)

