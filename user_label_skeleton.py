import cPickle as pickle
import os

def loadRecipes(directory):
    files = [os.path.join(directory,f) for f in os.listdir(directory) if (
                                    os.path.isfile(os.path.join(directory,f)) and
                                    os.path.join(directory,f).endswith('.p'))]
    for path in files:
        with open(path,'rb') as f:
            _,_,recipes = pickle.load(f)
        labeled_file = 'labels_'+path
        labeler(recipes, labeled_file)
        break

def parseReviewPhrases(reviews):
    pass
def parseRecipePhrases(recipe):
    pass

def labeler(recipes, labeled_file):
    labeled_recipes = recipes
    for recipe in recipes:
        reviews = parseReviewPhrases([r['text'] for r in recipes[recipe]['reviews']])
        recipe_text = parseRecipePhrases(recipes[recipe]['instructions'])
        labeled_recipes[recipe]["labels"] = {}
        review_index = 0
        for reviews in recipes[recipe]['reviews']:
            print reviews
            labeled_recipes[recipe]["labels"][review_index] = {}
            for i in range(len(reviews)):
                print i,".", reviews[i]
                check_refinement = str(raw_input("Refinement in review? "))
                if check_refinement == "y":
                    print "Refinement: ", reviews[i].upper()
                    for j in range(len(recipes[recipe]['instructions'])):
                        print j, ".", recipes[recipe]['instructions'][j]
                    indexing = str(raw_input("Line + Modification or insertion? "))
                    labeled_recipes[recipe]["labels"][review_index][i] = indexing
                else:
                    labeled_recipes[recipe]["labels"][review_index][i] = None

                pickle.dump( labeled_recipes, open( labeled_file, "wb" ) )

            review_index += 1

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
    saved_directory = 'scraper/pickle_files/all_recipes'
    loadRecipes(saved_directory)
	# recipes = {
        # 'Chicken Soup': {
            # 'instructions': ['chop carrots', 'put in bowl'],
            # 'reviews': [['i liked this recipe', 'instead of carrots i used broccoli'], []]
            # }
        # }

	# print labeler(recipes)
