import edibles.edibles_settings as Settings
import numpy as np
import pandas as pd

class file_info():
    def __init__(self, str, mode="dr4"):
        self.sightline = ""
        self.obsdate = ""
        self.setting = ""
        self.order = ""
        self.wave_min = None
        self.wave_min = None
        self.filename = ""
        if mode == "dr4": self.__loadDR4__(str)

        self.selected = True
        self.b_idx = 0

    def __loadDR4__(self,str):
        str = str.replace("\n","")
        str = str.replace(" ", "")
        str_split = str.split(",")
        self.sightline = str_split[0]
        self.filename = str_split[-1]
        self.setting = str_split[4]
        self.wave_min = float(str_split[5])
        self.wave_max = float(str_split[6])
        filename = str_split[-1]
        self.obsdate = filename.split("_")[-2]
        self.year = self.obsdate[0:4]
        order = filename.split("_")[-1]
        self.order = order.split(".")[0]

    def applyfilter(self,filter):
        filter.__checkMinMax__()
        self.selected = True
        while True:
            if filter.sightline is not "any" and self.sightline not in filter.sightline:
                self.selected = False
                break
            if filter.obsdate is not "any" and self.obsdate not in filter.obsdate:
                self.selected = False
                break
            if filter.year is not "any" and self.year not in filter.year:
                self.selected = False
                break
            if filter.setting is not "any" and self.setting not in filter.setting:
                self.selected = False
                break
            if filter.order is not "any" and self.order not in filter.order:
                self.selected = False
                break

            mid = []
            if filter.wave_min is not "any":
                mid.append(filter.wave_min)
                if filter.wave_min < self.wave_min or filter.wave_min > self.wave_max:
                    self.selected = False
                    break
            if filter.wave_max is not "any":
                mid.append(filter.wave_max)
                if filter.wave_max < self.wave_min or filter.wave_max > self.wave_max:
                    self.selected = False
                    break

            if len(mid) > 0:
                mid = np.mean(mid)
                self.b_idx = abs(np.mean([self.wave_max, self.wave_min]) - mid) / (self.wave_max - self.wave_min) * 2
                if filter.b_idx is not "any" and self.b_idx > filter.b_idx:
                    self.selected = False
                    break
            break

        return self.selected

    def __str__(self):
        wav_min = "{wav:.2f}".format(wav = self.wave_min)
        wav_max = "{wav:.2f}".format(wav = self.wave_max)
        selected = self.selected.__str__()
        b_idx = "{idx:.3f}".format(idx=self.b_idx)
        str_out = ", ".join([self.sightline, self.obsdate, self.setting,wav_min, wav_max, self.order,
                              self.filename, selected, b_idx])

        return str_out

class filter():
    def __init__(self):
        self.sightline="any"
        self.obsdate = "any"
        self.year = "any"
        self.setting = "any"
        self.order = "any"
        self.wave_min = "any"
        self.wave_max = "any"
        self.b_idx = "any"

    def resetFilter(self):
        self.__init__()

    def setFilter(self,*params,**kwargs):
        if "help" in params:
            self.__filterHelp__()
            return None

        kw_lower = dict((k.lower(), kwargs[k]) for k in kwargs.__iter__())
        if "sightline" in kw_lower.keys():
            sightline = kw_lower["sightline"]
            if sightline != "any" and type(sightline) is not list:
                sightline = [sightline]
            if type(sightline) is list:
                for idx, item in enumerate(sightline):
                    if type(item) is not str: sightline[idx] = "HD"+str(item)
            self.sightline = sightline

        if "obsdate" in kw_lower.keys():
            obsdate = kw_lower["obsdate"]
            if obsdate != "any":
                if type(obsdate) is not list: obsdate = [obsdate]
                obsdate = [str(i) for i in obsdate]
            self.obsdate = obsdate

        if "year" in kw_lower.keys():
            year = kw_lower["year"]
            if year != "any":
                if type(year) is not list: year = [year]
                year = [str(i) for i in year]
            self.year = year

        if "setting" in kw_lower.keys():
            setting = kw_lower["setting"]
            if setting != "any":
                if type(setting) is not list: setting = [setting]
                for i, item in enumerate(setting):
                    item = str(item)
                    assert item in ["346","564","437","860"], "Invalid setting input"
                    setting[i] = item
            self.setting = setting

        if "order" in kw_lower.keys():
            order = kw_lower["order"]
            if order != "any":
                if type(order) is not list: order = [order]
                for i, item in enumerate(order):
                    item = str(item)
                    if "O" not in "item": item = "O"+item
                    order[i] = item
            self.order = order

        if "center" in kw_lower.keys():
            center = kw_lower["center"]
            if center is not "any":
                center = float(center)
                if "span" in kw_lower.keys():
                    span = float(kw_lower["span"])
                else:
                    span = 0.0
                self.wave_min = center - span
                self.wave_max = center + span
            else:
                self.wave_max="any"
                self.wave_min="any"

        if "xmin" in kw_lower.keys(): self.wave_min = kw_lower["xmin"]
        if "xmax" in kw_lower.keys(): self.wave_max = kw_lower["xmax"]
        if "b_idx" in kw_lower.keys(): self.b_idx = kw_lower["b_idx"]

        self.__checkMinMax__()

    def __filterHelp__(self):
        print("="*40)
        print("Allowed Key Words and Formats:")
        print("Sightline, str like 'HD183143'")
        print("ObsDate, int or str, in the format of yyyymmdd")
        print("Year of Observation, int or str, yyyy")
        print("Setting, among 346, 564, 437 and/or 860")
        print("Order, order number, int or str")
        print("Center, float, center wavelength in AA")
        print("Span, float, span on each direction from center in AA")
        print("xMax/xMin, float, override center + span")
        print("b_idx, float between 0-1 and 1 means on the edge of the spectrum order")
        print("All key words can be set to 'any', i.e. no filter")
        print("Other than xMax/xMin and b_idx, all other key words can be a list for multiple")

    def __checkMinMax__(self):
        if self.wave_max is not "any" and self.wave_min is not "any":
            if self.wave_min > self.wave_max:
                self.wave_min = self.wave_max + self.wave_min
                self.wave_max = self.wave_min - self.wave_max
                self.wave_min = self.wave_min - self.wave_max

        return 1 * (self.wave_max != "any") + 2 * (self.wave_min != "any")

    def __str__(self):
        str_out = ""
        wav_flag = self.__checkMinMax__()
        if self.sightline != "any": str_out+= "Sightline: " + ", ".join(self.sightline) + "\n"
        if self.obsdate != "any": str_out+= "ObsDate: " + ", ".join(self.obsdate) + "\n"
        if self.year != "any": str+= "Year: " + ", ".join(self.year) + "\n"
        if self.setting != "any": str_out+= "Setting: " + ", ".join(self.setting) + "\n"
        if self.order != "any": str_out+= "Order: " + ", ".join(self.order) + "\n"
        if wav_flag == 1: str_out+= "Wavelength: < {wav:.2f}\n".format(wav = self.wave_max)
        if wav_flag == 2: str_out += "Wavelength: > {wav:.2f}\n".format(wav=self.wave_min)
        if wav_flag == 3: str_out += "Wavelength: {wav1:.2f} to {wav2:.2f}\n"\
            .format(wav1=self.wave_min, wav2=self.wave_max)
        if self.b_idx != "any": str_out+= "b_idx: < {b_idx:.2f}\n".format(b_idx=self.b_idx)

        if len(str_out) == 0: str_out = "No filter set"
        else: str_out = "Filters: \n" + str_out

        return str_out

class ObsLog():
    def __init__(self, LogFile=None):
        if LogFile is None: LogFile = Settings.ObsLog
        self.obslog = pd.read_csv(LogFile)
        self.fileinfo=[]
        self.filter = filter()

        with open(LogFile) as f:
            header = f.readline()
            while True:
                line = f.readline()
                if not line: break
                self.fileinfo.append(file_info(line))

        self.counter = len(self.fileinfo)

    def setFilter(self,*params,**kwargs):
        self.filter.setFilter(*params,**kwargs)

    def resetFilter(self):
        self.filter.resetFilter()

    def printFilter(self):
        print(self.filter)

    def applyFilter(self):
        counter = 0
        for item in self.fileinfo:
            counter+= item.applyfilter(self.filter)

        print("The following filter has been applied:")
        print(self.filter)
        print("{n} files selected".format(n=counter))

    def outputResult(self, *kword, filename=None, mode="multipanel"):
        if filename is None: filename = "selected_file.txt"
        filename = Settings.resultdir + filename
        mode_all = ["str", "singlepanel", "multipanel"]
        if "help" in kword:
            print("mode available: " + ",".join(mode_all))
            return
        assert mode in mode_all, "Available modes include: {modes}".format(modes=", ".join(mode_all))

        if mode == "str": return self.__outputStr__()
        if mode == "singlepanel": self.__outputSingle__(filename=filename)
        if mode == "multipanel": self.__outputMultiple__(filename=filename)

    def __outputStr__(self):
        str_out_lst = []
        for item in self.fileinfo:
            if item.selected: str_out_lst.append(item.filename)
        return str_out_lst

    def __outputSingle__(self, filename=None):
        if filename is None: filename = Settings.resultdir + "selected_file.txt"
        f = open(filename,"w")
        for item in self.fileinfo:
            if item.selected: f.write(",".join([item.sightline,item.filename] ) + "\n")
        f.close()

    def __outputMultiple__(self, filename=None):
        if filename is None: filename = Settings.resultdir + "selected_file.txt"
        sightline_lst = []
        filename_lst = []
        for item in self.fileinfo:
            if item.selected:
                if item.sightline not in sightline_lst:
                    sightline_lst.append(item.sightline)
                    filename_lst.append(item.filename)
                else:
                    idx = sightline_lst.index(item.sightline)
                    filename_lst[idx] = ",".join([filename_lst[idx],item.filename])

        f = open(filename,"w")
        for idx, sightline in enumerate(sightline_lst):
            f.write(",".join([sightline, filename_lst[idx]]) + "\n")
        f.close()

    def printResult(self):
        for item in self.fileinfo:
            if item.selected: print(item)


class ISM_Info:

    def __init__(self,datafile=None):
        if datafile is None: datafile = Settings.edibles_ISMinfo
        self.sightline_velocities = {}
        with open(datafile) as f:
            while True:
                line = f.readline().replace("\n","")
                if not line: break
                line_split = line.split(",")
                sightline = line_split[0]
                velocities_str = line_split[1]
                velocities_str = velocities_str.split(";")
                velocities=[]
                for velocity_str in velocities_str:
                    velocities.append(float(velocity_str))
                velocities = np.array(velocities)
                self.sightline_velocities[sightline] = velocities

    def lookupVelocity(self,sightline, mode="average"):
        assert mode.lower() in ["average", "all"], "Only 'average' and 'all' modes are allowed!"

        if sightline not in self.sightline_velocities.keys():
            print("ISM velocity for sightline {name} is unknown".format(name=sightline))
            return 0.0
        else:
            result = self.sightline_velocities[sightline]
            if mode.lower() == "all":
                return result
            else:
                return np.mean(result)

if __name__ == "__main__":
    "Getting spectra arond C3 line at 4051 AA in sightlines HD73882, 147889, 154368, 210121, and 169454"
    a = ObsLog()
    a.setFilter(sightline = [73882, 147889, 154368, 210121, 169454])
    a.setFilter(center=4051, span=5)
    a.printFilter()
    a.applyFilter()
    a.printResult()
    filenames = a.outputResult(mode="str")