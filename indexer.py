import operator
import ujson

papers = {}
authors_index = []
conflicts_index = []
keywords_index = []


def make_item_index(key):
    """
    make list sorted by value
    :param key: authors, conflicts, keywords
    :param indexes: dictionary
    :return:
    """
    indexes = {}
    for paper_id in papers:
        for item in papers[paper_id][key]:
            if item not in indexes:
                indexes[item] = set()
            indexes[item].add(paper_id)
    return sorted(indexes.items(), key=operator.itemgetter(0))


def make_global_index():
    global authors_index, conflicts_index, keywords_index
    with open('openreview.txt') as f:
        notes = ujson.load(f)
    for note in notes['notes']:
        papers[note['id']] = {
            'pdf': note['content']['pdf'],
            'conflicts': note['content']['conflicts'],
            'authors': note['content']['authors'],
            'title': note['content']['title'],
            'TL;DR': note['content']['TL;DR'],
            'keywords': note['content']['keywords']
        }
    authors_index = make_item_index('authors')
    conflicts_index = make_item_index('conflicts')
    keywords_index = make_item_index('keywords')


def query_note(id):
    return '[%s](http://openreview.net/forum?id=%s), %s, "%s", [[pdf](http://openreview.net/pdf?id=%s)]' %\
           (papers[id]['title'], id, ', '.join(papers[id]['authors']), papers[id]['TL;DR'], id)


def make_index_markdown(title, indexes, filename=''):
    if title == 'keywords':
        char_index = sorted(set([c[0] for c in indexes]))
    else:
        char_index = sorted(set([c[0][0] for c in indexes]))

    if not filename:
        filename = title
    with open('%s.md' % filename, 'w') as f:
        f.write('## %s Index\n' % title.capitalize())

        for c in char_index:
            f.write('[%s](#%s) ' % (c.capitalize(), c.lower().replace(' ', '-')))
        f.write('\n')

        prev_sub_title = ''
        for c in indexes:
            if c[0][0] != prev_sub_title:
                prev_sub_title = c[0][0]
                f.write('\n### %s\n' % prev_sub_title.upper())

            f.write('\n#### %s\n' % c[0])
            for paper_id in c[1]:
                f.write('* %s\n' % query_note(paper_id))


if __name__ == '__main__':
    make_global_index()
    make_index_markdown('authors', authors_index, 'readme')
    make_index_markdown('conflicts', conflicts_index)
    make_index_markdown('keywords', keywords_index)