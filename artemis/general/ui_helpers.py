from six.moves import input

from artemis.fileman.directory_crawl import DirectoryCrawler
from artemis.general.display import surround_with_header


def select_item_ui(item_name_list, prompt = 'Select Items>>'):

    print('\n'.join('{}: {}'.format(i, item) for i, item in enumerate(item_name_list)))
    cmd = input(prompt)
    if cmd=='all':
        ixs = range(item_name_list)
    elif '-' in cmd:
        start, end = cmd.split('-')
        ixs = range(int(start), int(end)+1)
    else:
        ixs = [int(cmd)]
    return ixs

