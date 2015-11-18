import requests

def getPage(url):
    try:
        r = requests.get(
            url=url,
            timeout=5
            #headers={
            #    'X-Requested-With': 'XMLHttpRequest',
            #}
        )
        return r
    except:
        return None
def savePageToFile(url, filepath):
    with open(filepath, 'wb') as handle:
        response = requests.get(url, stream=True)

        if not response.ok:
            return None

        for i,block in enumerate(response.iter_content(1024)):
            handle.write(block)
