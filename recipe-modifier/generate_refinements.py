#functions to read in batched data and generate refinements

def removals(phrases):
    #labels are the index in the new
    #list of phrases, separated by blanks,
    #where the phrase
    #should be inserted
    print phrases
    labels, refinements = zip(*[(2*i+1, phrase) for i,phrase in enumerate(phrases)])
    new_recipes = [phrases[:i]+phrases[i+1:] for i in xrange(len(phrases))]
    print new_recipes
    l = len(new_recipes[0][0])
    print l
    for i,r in enumerate(new_recipes):
        for j,p in enumerate(r):
            if len(r) != l:
                pass
                #print 'wtf'
                #print len(r)
    sys.exit()
    return labels, refinements, new_recipes

def _generate_modifications(phrase):
    #possibilities:
    #mess with grammer, syntax
    #replace verbs
    #replace nouns
    return [phrase, phrase]

def modifications(phrases):
    targets = [2*i for i in xrange(len(phrases))]
    for i,phrase in enumerate(phrases):
        new_phrases = _generate_modifications(phrase)
        targets[i] = (targets[i], new_phrases)
    return targets
if __name__ == '__main__':
    phrases = ['do this', 'do that', 'eat butter', 'cook food', 'sacrifice infants']
    print removals(phrases)
