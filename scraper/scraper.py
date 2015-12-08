from bs4 import BeautifulSoup
# from pyvirtualdisplay import Display
# from selenium import webdriver
from utils import web_functions as web
from utils import gen_utils
import os,sys,csv,re

class RecipeWebsiteScraper(object):
    def __init__(self, home_page, browse_href, save_directory, pages_per_pickle=100):
        self.home_page = home_page
        self.browse_href = browse_href
        self.save_directory = save_directory
        self.pages_per_pickle = pages_per_pickle
    def formUrl(self,href):
        return self.home_page + href
    def formatText(self,text):
        return gen_utils.replaceNonAscii(text.lstrip().rstrip())
    def newPickleFile(self):
        os.rename(self.pickle_file, os.path.join(self.save_directory, "pages_up_to_%s.p"%(int(self.page)-1)))
        self.pickle_file = os.path.join(self.save_directory, "latest.p")
        self.recipes = {}
        self.page_in_file = 1
    def scrape(self, page_arguments = "?page=%s", num_pages=None):
        statusCode = 200
        # self.display = Display(visible=0, size=(1024, 768))
        # self.display.start()
        # self.driver = webdriver.Firefox()

        while statusCode == 200 and (not num_pages or self.page <= num_pages):
            print "SCRAPING PAGE #",self.page
            if self.page_in_file > self.pages_per_pickle:
                self.newPickleFile()
            current_page_url = self.formUrl(self.browse_href + page_arguments % self.page)
            print current_page_url
            statusCode = self.scrapePage(current_page_url)
            sys.stdout.flush()
            gen_utils.updatePickleFile([self.page, self.page_in_file, self.recipes],self.pickle_file)
            print "updated pickle file"
            sys.stdout.flush()

            self.page += 1
            #try next page to see if just a problem with current page
            if statusCode != 200:
                print "TRYING NEXT PAGE"
                current_page = self.formUrl(self.browse_href + page_arguments%self.page)
                statusCode = self.scrapePage(current_page_url)
                gen_utils.updatePickleFile([self.page, self.page_in_file, self.recipes],self.pickle_file)
                sys.stdout.flush()
            self.page_in_file += 1
        # self.driver.quit()
        # self.display.stop()
    def loadRecipes(self):
        self.pickle_file = os.path.join(self.save_directory, "latest.p")
        savedInfo = gen_utils.loadObjectFromPickleFile(self.pickle_file)

        if savedInfo:
            self.page, self.page_in_file, self.recipes = savedInfo
            self.page += 1
            self.page_in_file += 1
        else:
            self.recipes = {}
            self.page = 1
            self.page_in_file = 1
    def get_selenium_text_excluding_children(self,element):
        # return self.driver.execute_script("""
                    # return jQuery(arguments[0]).contents().filter(function() {
                            # return this.nodeType == Node.TEXT_NODE;
                                # }).text();
            # """, element)
        return self.driver.execute_script("""
                var parent = arguments[0];
                var child = parent.firstChild;
                var ret = "";
                while(child) {
                        if (child.nodeType === Node.TEXT_NODE)
                                ret += child.textContent;
                                    child = child.nextSibling;
                                    }
                return ret;
                """, element)
class AllRecipesScraper(RecipeWebsiteScraper):
    def __init__(self, save_directory):
        super(AllRecipesScraper,self).__init__('http://allrecipes.com',
                                                '/recipes/?grouping=all',
                                                save_directory)
        self.loadRecipes()
    def scrape(self, num_pages=None):
        super(AllRecipesScraper,self).scrape(page_arguments="&page=%s", num_pages = num_pages)

    def scrapePage(self,url):
        page = web.getPage(url)
        if not page:
            return 0
        elif not page.text or page.status_code != 200:
            return page.status_code
        parsed = BeautifulSoup(page.text)
        for section in parsed.findAll('section'):
            if 'class' in section.attrs and 'recipe_hub' in section['class']:
                for subsection in section.findAll('section'):
                    if 'class' in subsection.attrs and 'grid' in subsection['class']:
                        for article in subsection.findAll('article'):
                            for link in article.findAll('a'):
                                if 'href' in link.attrs:
                                    href = link.get('href')
                                    break
                            if not href:
                                continue
                            url = self.formUrl(href)
                            if url not in self.recipes:
                                self.scrapeRecipePage(self.formUrl(href))
        return page.status_code

    def scrapeRecipePage(self,url):
        page = web.getPage(url)
        #self.driver.get(url)
        if not page:
            print "no page:", url
            return
        elif not page.text or page.status_code != 200:
            return
        else:
            print "success:", url
            parsed = BeautifulSoup(page.text)
            metadata = {}
            instructions = []
            notes = []
            footnotes = []
            for div in parsed.findAll('div'):
                if 'class' in div.attrs and 'recipe-container-outer' in div['class']:
                    for ul in div.findAll('ul'):
                        if 'class' in ul.attrs and 'breadcrumbs' in ul['class']:
                            startRecipeCategory = False
                            categories = []
                            for li in ul.findAll('li'):
                                if startRecipeCategory == False:
                                    try:
                                        breadcrumb_name = li.a.span.text
                                    except:
                                        continue
                                    else:
                                        breadcrumb_name = self.formatText(breadcrumb_name)
                                        if breadcrumb_name.startswith('Recipe'):
                                            startRecipeCategory = True
                                else:
                                    try:
                                        breadcrumb_name = li.a.span.text
                                    except:
                                        continue
                                    else:
                                        subcategory = self.formatText(breadcrumb_name)
                                        categories.append(subcategory)
                            metadata['categories'] = categories
                    for subdiv in div.findAll('div'):
                        if 'class' in subdiv.attrs and 'summaryGroup' in subdiv['class']:
                            for section in subdiv.findAll('section'):
                                if 'class' in section.attrs and 'recipe-summary' in section['class']:
                                    summary = self.formatText(section.h1.text)
                                    for meta in section.findAll('meta'):
                                        if 'itemprop' in meta.attrs and 'ratingValue' in meta['itemprop']:
                                            rating = meta['content']
                                            metadata['rating'] = rating
                                            break
                                    for span in section.findAll('span'):
                                        if 'class' in span.attrs and 'review-count' in span['class']:
                                            review_count = int(span.text)
                                            metadata['review_count'] = review_count
                                        elif 'class' in span.attrs and 'made-it-count' in span['class']:
                                            try:
                                                made_it_count = int(span.text)
                                            except:
                                                pass
                                            else:
                                                metadata['made_it_count'] = made_it_count
                                        elif 'class' in span.attrs and 'submitter__name' in span['class']:
                                            author = span.text
                                            metadata['author'] = author
                                    for subsubdiv in section.findAll('div'):
                                        if 'class' in subsubdiv.attrs and 'submitter__description' in subsubdiv['class']:
                                            description = self.formatText(subsubdiv.text)
                                            metadata['description'] = description
                                            break
                    for section in div.findAll('section'):
                        if 'class' in section.attrs and 'recipe-ingredients' in section['class']:
                            metaRecipeServings = section.find(id="metaRecipeServings")
                            metadata['recipe_servings'] = metaRecipeServings.get('content')
                            nutritionButton = section.find(id="nutrition-button")
                            if nutritionButton:
                                for span in nutritionButton.findAll('span'):
                                    if 'class' in span.attrs and 'calorie-count' in span['class']:
                                        for subspan in span.findAll('span'):
                                            if 'class' not in span.attrs:
                                                metadata['calorie_count'] = int(span.text)
                                                break
                                        break
                            metadata['ingredients'] = []
                            for ul in section.findAll('ul'):
                                column = 1
                                if 'class' in ul.attrs and "list-ingredients-%s"%column in ul['class']:
                                    for li in ul.findAll('li'):
                                        for span in li.findAll('span'):
                                            if 'itemprop' in span.attrs and 'ingredients' in span['itemprop']:
                                                metadata['ingredients'].append(self.formatText(span.text))
                                    column += 1
                        elif 'class' in section.attrs and 'recipe-directions' in section['class']:
                            for time in section.findAll('time'):
                                if 'itemprop' in time.attrs and 'prepTime' in time['itemprop']:
                                    metadata['prep_time'] = time.get('datetime')
                                elif 'itemprop' in time.attrs and 'cookTime' in time['itemprop']:
                                    metadata['cook_time'] = time.get('datetime')
                                elif 'itemprop' in time.attrs and 'totalTime' in time['itemprop']:
                                    metadata['total_time'] = time.get('datetime')
                            for ol in section.findAll('ol'):
                                if 'itemprop' in ol.attrs and 'recipeInstructions' in ol['itemprop']:
                                    for li in ol.findAll('li'):
                                        step = self.formatText(li.span.text)
                                        instructions.append(step)
                                elif 'class' in ol.attrs and 'recipeNotes' in ol['class']:
                                    for li in ol.findAll('li'):
                                        try:
                                            step = self.formatText(li.span.text)
                                        except:
                                            continue
                                        else:
                                            notes.append(step)
                                    break
                        elif 'class' in section.attrs and 'recipe-footnotes' in section['class']:
                            for ul in section.findAll('ul'):
                                for li in ul.findAll('li'):
                                    header = None
                                    new_footnote = None
                                    if li.span:
                                        if new_footnote:
                                            footnotes.append(new_footnote)
                                        header = self.formatText(li.span.text)
                                        new_footnote = {header:[]}
                                    else:
                                        if new_footnote:
                                            new_footnote[header].append(self.formatText(li.text))
                                        else:
                                            header = self.formatText(li.text)
                                            new_footnote = {header:[]}
                                footnotes.append(new_footnote)
                    reviewLinks = []
                    review_section = div.find(id='reviews')
                    for a in review_section.findAll('a'):
                        if 'class' in a.attrs and 'review-detail__link' in a['class']:
                            href = a.get('href')
                            reviewLinks.append(url+'/'+href)
                    reviews = []
                    for i,review_link in enumerate(reviewLinks):
                        review_page = web.getPage(review_link)
                        if not review_page or not review_page.text:
                            continue
                        else:
                            review = {}
                            parsed_review_page = BeautifulSoup(review_page.text)
                            for p in parsed_review_page.findAll('p'):
                                if 'itemprop' in p.attrs and 'reviewBody' in p['itemprop']:
                                    review['text'] = self.formatText(p.text)
                                    break
                            for rdiv in parsed_review_page.findAll('div'):
                                if 'class' in rdiv.attrs and 'review-detail__stars' in rdiv['class']:
                                    for rrdiv in rdiv.findAll('div'):
                                        if 'class' in rrdiv.attrs and 'rating-stars' in rrdiv['class']:
                                            try:
                                                review['rating'] = float(rrdiv['data-ratingstars'])
                                            except:
                                                break
                                    break
                                elif 'class' in rdiv.attrs and 'helpful-count' in rdiv['class']:
                                    for num in rdiv.findAll('format-large-number'):
                                        try:
                                            review['helpful_count'] = int(num.get('number'))
                                        except:
                                            continue
                                        else:
                                            break
                            reviews.append(review)


            self.recipes[url] = {'metadata': metadata, 'instructions': instructions, 'notes': notes, 'footnotes': footnotes, 'reviews': reviews}

class FoodDotComScraper(RecipeWebsiteScraper):
    def __init__(self, save_directory):
        super(FoodDotComScraper,self).__init__('http://food.com',
                                                '/recipe/',
                                                save_directory)
        self.loadRecipes()
        self.max_review_pages = 15
    def scrape(self, num_pages=None):
        super(FoodDotComScraper,self).scrape(page_arguments="?pn=%s", num_pages = num_pages)

    def scrapePage(self,url):
        page = web.getPage(url)
        if not page:
            return 0
        elif not page.text or page.status_code != 200:
            return page.status_code
        parsed = BeautifulSoup(page.text)
        for div in parsed.findAll('div'):
            if 'class' in div.attrs and 'fd-content' in div['class']:
                print 'fd-content'
                for sdiv in div.findAll('div'):
                    print 'fd-recipe'
                    print sdiv
                    for a in sdiv.findAll('a'):
                        print a['href']
                        if 'recipe' in a['href']:
                            url = a['href']
                            print 'url'
                            if url not in self.recipes:
                                print url
                            #self.scrapeRecipePage(url)
        return page.status_code

    def scrapeRecipePage(self,url):
        page = web.getPage(url)
        if not page:
            print "no page:", url
            return
        elif not page.text or page.status_code != 200:
            return
        else:
            print "success:", url
            parsed = BeautifulSoup(page.text)
            metadata = {}
            instructions = []
            notes = []
            footnotes = []
            for article in parsed.findAll('article'):
                if 'data-page-name' in article.attrs:
                    for div in article.findAll('div'):
                        if 'class' in div.attrs and 'recipe-detail' in div['class']:
                            for sdiv in div.findAll('div'):
                                if 'class' in div.attrs and 'directions' in div['class']:
                                    for ol in div.findAll('ol'):
                                        if 'itemprop' in ol.attrs and 'recipeInstructions' in ol['itemprop']:
                                            for li in ol.findAll('li'):
                                                instructions.append(li.text)
                                            break
                                    break

                            break
                    break
            review_url = url + '/review'
            review_page = web.getPage(review_url)
            reviews = []
            if not review_page or not review_page.text or review_page.status_code != 200:
                self.recipes[url] = {'instructions': instructions, 'reviews': reviews}
                return
            else:
                total_number_of_reviews_index = review_page.text.find('totalReviews = ')
                total_number_of_reviews = review_page.text[total_number_of_reviews_index+1: total_number_of_reviews_index+4]
                try:
                    int(total_number_of_reviews)
                except:
                    try:
                        int(total_number_of_reviews[:-1])
                    except:
                        try:
                            total_number_of_reviews = int(total_number_of_reviews[:-2])
                        except:
                            total_number_of_reviews = 25
                num_pages = total_number_of_reviews/25
                pages_to_check = min(num_pages, self.max_review_pages)
            print 'num_pages:', num_pages
            for page in xrange(2, pages_to_check+1):
                parsed = BeautifulSoup(review_page.text)
                for section in parsed.findAll('section'):
                    if 'class' in section.attrs and 'reviews' in section['class']:
                        for article in section.findAll('article'):
                            review_text = article.p.text
                            if review_text:
                                reviews.append(review_text)
                        break
                if page < pages_to_check:
                    review_page = web.getPage(review_url+'?pn=%d'%page)
            self.recipes[url] = {'instructions': instructions, 'reviews': reviews}
            print self.recipes
            sys.exit()

if __name__ == '__main__':
    scraper = AllRecipesScraper('pickle_files/all_recipes')
    scraper = FoodDotComScraper('pickle_files/food_dot_com')
    scraper.scrape()
