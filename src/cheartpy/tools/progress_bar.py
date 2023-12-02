def printProgressBar(
    iteration: int,
    total: int,
    prefix="",
    suffix="Complete",
    decimals=1,
    length=50,
    fill="*",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 if (total == 0) else 100 * (iteration / float(total))
    )
    filledLength = int(length if (total == 0) else length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end=printEnd)
    if iteration == total:
        print()


class progress_bar:
    __slots__ = ["n", "i", "msg"]
    n: int
    i: int
    msg: str

    def __init__(self, message, max=100):
        self.n, self.i, self.msg = max, 0, message
        printProgressBar(self.i, self.n, prefix=self.msg)

    def reset(self) -> None:
        self.i = 0

    def next(self):
        self.i = self.i + 1
        printProgressBar(self.i, self.n, prefix=self.msg)

    def finish(self):
        printProgressBar(self.n, self.n, prefix=self.msg)
