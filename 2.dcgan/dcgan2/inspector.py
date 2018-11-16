import abc

class Inspector(abc.ABC):
    pass


class TBSummary(Inspector):
    def __init__(self, out_path):
        self.out_path = out_path
#class DCGANTBSummary(Inspector):
#    pass

#class DCGANConsolLogger(Inspector):
#    pass