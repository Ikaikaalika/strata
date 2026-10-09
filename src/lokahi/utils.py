import codecs, io, time

def file_put_contents(filename, st):
    file = codecs.open(filename, "w", "utf-8")
    file.write(st)
    file.close()

def file_get_contents(name):
    f = io.open(name, mode="r", encoding="utf-8") #utf-8 | Windows-1252
    return f.read()

class Stats:
    def __init__(self):
        self.d = {}

    def set(self, name, t1):
        if name not in self.d: self.d[name] = []
        self.d[name].append( round(time.perf_counter() - t1, 3) )

    def print_and_clean(self):
        st = "Stats:"
        for name, a in self.d.items():
            st+=f" {name}: {a[:5]} t:{round(sum(a), 3)},"
        self.d = {}
        return st


