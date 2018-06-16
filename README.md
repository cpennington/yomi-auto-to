Yomi Automatic Tournament Organizer (AutoTO)
============================================

Installation
------------

* [Install Python](https://www.python.org/downloads/)
  * Why? Python is the program that you will use to run AutoTO
* [Install Git]( https://help.github.com/desktop/guides/getting-started-with-github-desktop/)
  * Why? Git is the tool used to store the AutoTO files
* Clone [AutoTO](https://github.com/cpennington/yomi-auto-to)
  * How? [Cloning a repository](https://help.github.com/articles/cloning-a-repository/)
  * Why? This will copy AutoTO onto your computer prior to use
* [Get a Challonge API Key](https://challonge.com/settings/developer)
  * Why? AutoTO will use this to look up the current state of tournaments you are running
* Install AutoTO
  * On a commandline in the `yomi-auto-to` directory, run `pip install -r requirements.txt`
    * Why? This will install the python packages used by AutoTO
    * How?
      * On Windows: [Python on Windows FAQ](https://docs.python.org/3/faq/windows.html)
          * In GitHub Desktop: Repository > Open in Command Prompt
      * On Mac OS X: [Using Python on a Macintosh](https://docs.python.org/3.6/using/mac.html)

Usage
-----

AutoTO is run from the commandline, using [Python](https://www.python.org).

From the `yomi-auto-to` directory, run `python auto_to.py --help` or `./auto_to.py --help` to see
a list of commands and their various options.
