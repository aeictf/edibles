import edibles.edibles_settings as Settings
import numpy as np
import pandas as pd
import copy

class filter():
    def __init__(self):
        self.object= None
        self.date = None
        self.year = None
        self.setting = None
        self.order = "all"
        self.wave_min = None
        self.wave_max = None
        self.b_idx = 1.0

    def reset(self):
        self.__init__()

    def setFilter(self,*params,**kwargs):
        if "help" in params:
            self.__filterHelp__()
            return None
        kw_lower = dict((k.lower(), kwargs[k]) for k in kwargs.__iter__())
        if "object" in kw_lower.keys(): self.__setObject__(kw_lower["object"])
        if "date" in kw_lower.keys(): self.__setDate__(kw_lower["date"])
        if "year" in kw_lower.keys(): self.__setYear__(kw_lower["year"])
        if "setting" in kw_lower.keys(): self.__setSetting__(kw_lower["setting"])
        if "order" in kw_lower.keys(): self.__setOrder__(kw_lower["order"])
        if "wave" in kw_lower.keys():
            if "span" in kw_lower.keys():
                self.__setWave__(kw_lower["wave"], span=kw_lower["span"])
            else:
                self.__setWave__(kw_lower["wave"])
        if "b_idx" in kw_lower.keys(): self.b_idx = kw_lower["b_idx"]
        self.__checkFilter__()

    def __setObject__(self, object):
        if object is not None:
            if type(object) not in [type([1,2]), type(np.array([1,2]))]: object = [object]
            for i, item in enumerate(object):
                if type(item) not in [type("a"), type(np.array(["a", "b"])[0])]:
                    object[i] = "HD" + str(item)
            if len(object) == 0: object = None
        self.object = object

    def __setDate__(self, date):
        if date is not None:
            if type(date) not in [type([1,2]), type(np.array([1,2]))]: date = [date]
            date = [str(item) for item in date]
            date = [item for item in date if len(item) == 8 or len(item) == 10]
            for i, item in enumerate(date):
                if "-" not in item: date[i] = "-".join([item[0:4], item[4:6], item[6:8]])
            if len(date) == 0: date=None
        self.date = date

    def __setYear__(self, year):
        if year is not None:
            if type(year) not in [type([1,2]), type(np.array([1,2]))]: year = [year]
            year = [str(item) for item in year]
            year = [item for item in year if len(item) == 4]
            if len(year) == 0: year=None
        self.year = year

    def __setSetting__(self, setting):
        if setting is not None:
            if type(setting) not in [type([1,2]), type(np.array([1,2]))]: setting = [setting]
            setting = [item for item in setting if item in [346,564,437,860]]
            if len(setting) == 0: setting = None
        self.setting = setting

    def __setOrder__(self, order):
        if order is not None:
            if order.lower() not in ["merged", "order", "all"]:
                print("Cannot process Order setting, showing ALL results")
                self.order = "all"
            else:
                self.order = order.lower()
        else:
            self.order = "all"

    def __setWave__(self, wave, span=0):
        if wave is not None:
            if type(wave) not in [type([1,2]), type(np.array([1,2]))]:
                self.wave_min = wave - 0.5*span
                self.wave_max = wave + 0.5*span
            else:
                self.wave_min = np.min(wave)
                self.wave_max = np.max(wave)
        else:
            self.wave_min = None
            self.wave_max = None

    def __filterHelp__(self):
        print("="*40)
        print("Allowed Key Words and Formats:")
        print("Object, str or int, can be array")
        print("Date, str or int array, yyyymmdd or 'yyyy-mm-dd', can be array")
        print("Year, four digit, str or int, can be array")
        print("Setting, int, any combination of 346, 564, 437 and 860")
        print("Order, 'merged', 'order', OR 'all'")
        print("Wave, float for center wavelength, or array of two elements for min and max")
        print("Span, float, total span, only used when wave is float")
        print("b_idx, between 0 and 1, e.g. 10% of data will be ignored when set to 0.1")
        print("All key words can be set to None for no filter")

    def __checkFilter__(self):
        if self.wave_max is not None and self.wave_min is not None:
            if self.wave_min > self.wave_max:
                self.wave_min = self.wave_max + self.wave_min
                self.wave_max = self.wave_min - self.wave_max
                self.wave_min = self.wave_min - self.wave_max
        if self.b_idx < 0  or self.b_idx > 1:
            self.b_idx = 1

    def __str__(self):
        str_out = ""
        if self.object is not None: str_out+= "Object: " + ", ".join(self.object) + "\n"
        if self.date is not None: str_out+= "Date: " + ", ".join(self.date) + "\n"
        if self.year is not None: str_out+= "Year: " + ", ".join(self.year) + "\n"
        if self.setting is not None: str_out+= "Setting: " + \
                                               ", ".join([str(item) for item in self.setting]) + "\n"
        if self.order == "merged": str_out+= "Merged only\n"
        if self.order == "order": str_out+= "Orders only\n"
        if self.wave_min is not None:
            if self.wave_min != self.wave_max:
                str_out+= "Wavelength: {wav1:.2f} to {wav2:.2f} AA\n"\
                    .format(wav1=self.wave_min, wav2=self.wave_max)
            else:
                str_out+= "Wavelength: contain {wav} AA\n".format(wav=self.wave_min)
        if self.b_idx != 1.0: str_out+= "b_idx: consider central {b:.2f} percent spectra data\n"\
            .format(b=(self.b_idx) * 100)

        if len(str_out) == 0: str_out = "No filter set"
        else: str_out = "Filters: \n" + str_out

        return str_out

class EdiblesOracle:
    def __init__(self, LogFile=None):
        if LogFile is None: LogFile = Settings.ObsLog
        self.obslog = pd.read_csv(LogFile)
        self.filter = filter()

    def setFilter(self, *params,**kwargs):
        self.filter.setFilter(*params,**kwargs)
        if "help" not in params and "silence" not in params: print(self.filter)

    def resetFilter(self):
        self.filter.reset()

    def applyFilter(self):
        bool_select = ~self.obslog.Filename.isna()

        if self.filter.object is not None:
            bool_object = self.obslog.Object == "Zzz"
            for item in self.filter.object:
                bool_object = bool_object | self.obslog.Filename.str.contains(item)
            bool_select = bool_select & bool_object

        if self.filter.date is not None:
            bool_date = self.obslog.Object == "Zzz"
            for item in self.filter.date:
                bool_date = bool_date | self.obslog.DateObs.str.contains(item)
            bool_select = bool_select & bool_date

        if self.filter.year is not None:
            bool_year = self.obslog.Object == "Zzz"
            for item in self.filter.year:
                bool_year = bool_year | self.obslog.DateObs.str.contains(item)
            bool_select = bool_select & bool_year

        if self.filter.setting is not None:
            bool_setting = self.obslog.Setting.isin(self.filter.setting)
            bool_select = bool_select & bool_setting

        if self.filter.order == "merged":
            bool_order = self.obslog.Order == "ALL"
            bool_select = bool_select & bool_order
        if self.filter.order == "order":
            bool_order = self.obslog.Order != "ALL"
            bool_select = bool_select & bool_order

        if self.filter.wave_max is not None:
            b = self.filter.b_idx
            bool_min = 0.5*((1-b)*self.obslog.WaveMax + (1+b)*self.obslog.WaveMin) < self.filter.wave_min
            bool_max = 0.5*((1+b)*self.obslog.WaveMax + (1-b)*self.obslog.WaveMin) > self.filter.wave_max
            bool_select = bool_select & bool_min & bool_max

        ind = np.where(bool_select)
        ind = ind[0]
        return ind

    def selectFile(self, *params, out="single", **kwargs):
        self.filter.reset()
        self.setFilter(*params,**kwargs)
        if "help" in kwargs: return None
        ind = self.applyFilter()
        if out == "single": return self.__returnSingle__(ind)
        if out == "multiple": return self.__returnMultiple__(ind)

    def __returnSingle__(self, ind):
        out = []
        for i in ind:
            out.append(self.obslog.iloc[i].Filename)
        return out

    def __returnMultiple__(self, ind):
        out=[]
        log_tmp = self.obslog.iloc[ind]
        object_unique = log_tmp.Object.unique()
        for item in object_unique:
            out_tmp = []
            bool_select = log_tmp.Object == item
            ind = np.where(bool_select)[0]
            for i in ind:
                out_tmp.append(log_tmp.iloc[i].Filename)
            out.append(out_tmp)

        return out

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

        if type(sightline) not in [type("str"), type(np.array(["123"])[0])]:
            sightline = "HD"+str(sightline)

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
    log = EdiblesOracle()
    file_list = log.selectFile(object=[73882, 147889, 154368, 210121, 169454],
                               order="order",b_idx=0.6,wave=4051,out="multiple")
    print(file_list)