from __future__ import print_function


class Options(object):
    """
        Class for representing command-line options.
    """

    def __init__(self, 
                 global_model_name, 
                 xtype, 
                 xnum, 
                 global_model_path, 
                 proj_name, 
                 data_path):
        """
            Constructor.
        """
        self.global_model_name=global_model_name
        self.xtype = xtype
        self.xnum = xnum
        self.classifier = global_model_path
        self.proj_name = proj_name
        self.data_path = data_path
        self.files = ['openstack', 'RF']
        self.in_jupyter = False

        self.accmin = 0.95
        self.files = None
        self.inst = None
        self.mapfile = None
        self.maxdepth = 3
        self.n_estimators = 30
        self.output = 'global_model'
        self.reduce = 'lin'
        self.seed = 7
        self.separator = ','
        self.smallest = False
        self.solver = 'g3'
        self.testsplit = 0.3
        self.train = False
        self.unit_mcs = False
        self.use_categorical = False
        self.validate = False
        self.verb = 1
        self.encode = None
        
