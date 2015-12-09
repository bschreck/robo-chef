import pck_to_txt
import stanford_parser_test as stan
import cPickle as pickle
import os

def loadRecipes(all_recipes_file):
    with open(all_recipes_file,'rb') as f:
        recipes = pickle.load(f)
    labeled_file = 'labels.p'
    labeler(recipes, labeled_file)

def parseReviewPhrases(reviews):
    phrases = [p for r in reviews
                    for p in stan.parse(r)]
    return phrases
def parseRecipePhrases(recipe):
    phrases = [p for s in recipe
                    for p in stan.parse(s)]
    return phrases

def labeler(recipes, labeled_file):
    labeled_recipes = {}
    for r_i, recipe in enumerate(recipes):
        if recipe in labeled_recipes:
            continue
        reviews = []
        for review in recipes[recipe]['reviews']:
            try:
                reviews.append(r['text'])
            except:
                reviews.append(review)
        reviews = parseReviewPhrases(reviews)
        recipe_text = parseRecipePhrases(recipes[recipe]['instructions'])
        current_label = {}
        review_index = 0
        for i,review in enumerate(reviews):
            current_label[review_index] = {}
            print "Review %d."%i, review
            check_refinement = str(raw_input("Refinement in review? "))
            if check_refinement == "y":
                print "Refinement: ", reviews[i].upper()
                for j,step in enumerate(recipe_text):
                    print j, ".", step
                while True:
                    indexing = str(raw_input("Line + Modification or Insertion? "))
                    if indexing.startswith('q'):
                        labeled_recipes[recipe] = current_label
                        pickle.dump( labeled_recipes, open( labeled_file, "wb" ) )
                        sys.exit()
                    else:
                        try:
                            indexing = int(indexing)
                        except:
                            print 'Enter an integer'
                        else:
                            if indexing < len(recipe_text):
                                current_label[review_index][i] = int(indexing)
                                break
                            else:
                                print 'Integer must be less than length of recipe steps'
            elif check_refinement.startswith('q'):
                labeled_recipes[recipe] = current_label
                pickle.dump( labeled_recipes, open( labeled_file, "wb" ) )
                sys.exit()
            else:
                current_label[review_index][i] = None
        labeled_recipes[recipe] = current_label
        review_index += 1

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
    saved_directory = '../scraper/pickle_files/full_dataset.p'
    loadRecipes(saved_directory)
	# recipes = {
        # 'Chicken Soup': {
            # 'instructions': ['chop carrots', 'put in bowl'],
            # 'reviews': [['i liked this recipe', 'instead of carrots i used broccoli'], []]
            # }
        # }

	# print labeler(recipes)
