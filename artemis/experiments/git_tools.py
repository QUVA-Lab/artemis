
from git import Git

import git
from git import Repo
from sys import argv
import sys
from git.exc import InvalidGitRepositoryError


def get_git_repo_for_current_run():
    path = sys.argv[0]
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    print git_root


def save_working_copy(message = 'autocommit'):

    g = Git(get_git_repo_for_current_run())

    g.stash()

    g.checkout('-B', 'artemis-experiments')

    g.stash('apply')

    unmerged_files = g.diff('--name-only', '--diff-filter=U')

    print unmerged_files

    g.commit('-am', message)

    g.checkout('-')

    g.stash('pop')


print 'ee'


save_working_copy()
