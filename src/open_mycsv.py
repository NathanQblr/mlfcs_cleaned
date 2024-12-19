"""Extract information in folder names that follow this type : FCS:motion_thread_name_Wx_wxy_Wz_w_Illu_illu.csv"""
class CSVNameParser():
    def __init__(self, name):
        self.fields = name.split('_')
    @property
    def name(self):
        #file name
        return int(self.fields[2])

    @property
    def motion(self):
        #type of motion
        return str(self.fields[0][4:])

    @property
    def wxy(self):
        #focal volume width
        return int(self.fields[5])

    @property
    def w(self):
        #focal volume depth
        return int(self.fields[7][:-4])

    @property
    def illu(self):
        #illumination parameter
        return int(1)


"""class CSVNameParser():
    def __init__(self, name):
        self.fields = name.split('_')
    @property
    def name(self):
        #file name
        return int(self.fields[3])

    @property
    def motion(self):
        #type of motion
        return str(self.fields[1])#[4:])

    @property
    def wxy(self):
        #focal volume width
        return int(self.fields[5])

    @property
    def w(self):
        #focal volume depth
        return int(self.fields[7])

    @property
    def illu(self):
        #illumination parameter
        return int(self.fields[8][:-4])"""