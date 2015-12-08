import boto
import warc
from StringIO import StringIO
import os
import json
from bs4 import BeautifulSoup
from utils import gen_utils

from boto.s3.key import Key
from gzipstream import GzipStreamFile

import sys
import gzip
import multiprocessing as mp
import datetime
import re

class Process:
    def __init__(self, save_directory, max_recipes_per_file = 2500):
        global run
        self.recipes = {}

        self.count = 0
        self.start = datetime.datetime.now()

        self.workers = []
        directory = 'cdx-index-client/allrecipes_index2'
        paths = []
        for path in os.listdir(directory):
            paths.append(os.path.join(directory, path))
        paths = sorted(paths)

        pool = mp.Pool(processes=mp.cpu_count())
        cpus = mp.cpu_count()
        if len(paths) > cpus:
            chunks = [[paths[i]] for i in xrange(cpus)]
            if len(paths) <= 2*cpus:
                for i in xrange(len(paths)-cpus):
                    chunks[i].append(paths[cpus+i])
            else:
                raise ValueError, "not implemented yet"

        else:
            raise ValueError, "not implemented yet"
        for i,chunk in enumerate(chunks):
            p = pool.apply_async(run, [chunk, save_directory, i, max_recipes_per_file])
            self.workers.append(p)
        [w.get() for w in self.workers]

def initialize_logging():
    outfile = str(os.getpid()) + ".out"
    errfile = str(os.getpid()) + "_error.out"
    sys.stdout = open(outfile, "a", buffering=0)
    sys.stderr = open(errfile, "a", buffering=0)

def loadRecipes(save_directory, index):
    pickle_file = os.path.join(save_directory, "latest_%d.p"%index)
    saved_info = gen_utils.loadObjectFromPickleFile(pickle_file)
    if saved_info:
        file_num, recipes, old_recipes = saved_info
        return file_num, recipes, old_recipes
    else:
        return 0, {}, set()

def genNewPickleFile(pickle_file, file_num):
    os.rename(pickle_file, pickle_file.replace('latest', "page_%d_index"%file_num))
def run(path_chunk,save_directory, index, max_recipes_per_file):
    initialize_logging()
    pickle_file = os.path.join(save_directory, "latest_%d.p"%index)
    file_num, recipes, old_recipes = loadRecipes(save_directory, index)
    for path in path_chunk:
        sys.stdout.flush()
        f = gzip.GzipFile(path, 'r')

        for line_index, line in enumerate(f.readlines()):
            num_spaces = 0
            json_string = ''
            for i,c in enumerate(line):
                if num_spaces == 2:
                    json_string = line[i:]
                    break
                if c == ' ':
                    num_spaces += 1
            d = json.loads(json_string)
            offset = int(d['offset'])
            length = int(d['length'])
            url = d['url']
            filename = d['filename']

            title = url.lower().split('/recipe/')[1].split('/')[0]
            if title in old_recipes:
                continue

            recipe = parse_archive(filename, url, offset, length)
            if recipe and 'instructions' in recipe:
                print "adding:", title
                recipes[title] = recipe
                old_recipes.add(title)
                sys.stdout.flush()
                if len(recipes) % 25 == 0:
                    gen_utils.updatePickleFile([file_num, recipes, old_recipes],pickle_file)
                    if len(recipes) > max_recipes_per_file:
                        genNewPickleFile(pickle_file, file_num)
                        recipes = {}
                        file_num += 1


def parse_archive(filename, url, offset, length):
    # Connect to Amazon S3 using anonymous credentials
    conn = boto.connect_s3(anon=True)
    bucket = conn.lookup('aws-publicdatasets')
    key = bucket.lookup(filename)
    end = offset + length - 1

    headers={'Range' : 'bytes={}-{}'.format(offset, end)}

    chunk = StringIO(
                key.get_contents_as_string(headers=headers)
            )

    page =  gzip.GzipFile(fileobj=chunk).read()
    return process_record(page)



def process_record(page):
    parsed = BeautifulSoup(page)
    item_id_elt = parsed.find(id='metaItemId')
    if item_id_elt and 'content' in item_id_elt.attrs:
        item_id = item_id_elt['content']
    else:
        return
    instructions = []
    for div in parsed.findAll('div'):
        if 'class' in div.attrs and ('directions' in div['class'] or 'recipe-directions' in div['class']):
            for ol in div.findAll('ol'):
                for li in ol.findAll('li'):
                    if li.span:
                        instructions.append(li.span.text)
                    elif li.text:
                        instructions.append(li.text)
                break
            break
    reviews = []
    for review in parsed.findAll(id='pReviewText'):
        reviews.append(review.text)
    return {'_id': item_id, 'reviews':reviews, 'instructions':instructions}

if __name__ == '__main__':
    Process('pickle_files/cc_all_recipes', max_recipes_per_file=2500)
