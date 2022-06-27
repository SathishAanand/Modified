# color definitions from SO:
# https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python
class bcolors:
    #HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    HEADER = '\033[93m'
    WARNING = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def printHeader():
    print(bcolors.HEADER + '======================================================================')
    print(bcolors.HEADER + ' ')
    print(bcolors.HEADER + ' Learning LES Closures...')
    print(bcolors.HEADER + ' ')
    print(bcolors.HEADER + '======================================================================'+bcolors.ENDC)


def printBanner(string):
    print(bcolors.HEADER + '\n======================================================================')
    print(bcolors.HEADER + ' '+string)
    print(bcolors.HEADER + '======================================================================'+bcolors.ENDC)


def printWarning(string):
    print(bcolors.WARNING + '\n !! '+string+' !! \n'+bcolors.ENDC)


def printNotice(string):
    print('\n# '+string)
