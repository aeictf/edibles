import numpy as np
import pandas as pd
from astropy.io import fits
import astropy.constants as cst
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import os
import math
import copy

from sherpa.stats import LeastSq
from sherpa.optmethods import LevMar, NelderMead
from sherpa.data import Data1D
from sherpa.fit import Fit

import edibles.edibles_settings as Setting
import edibles.src.datahandling as DataHandling
import edibles.src.math as eMath
import edibles.src.model as eM


class SingleSpectrum:
    """
    This class is for single fits data, major attributes are the spectrum and some markers
    """

    # init, load data, rename, reset
    def __init__(self, filename, *kword, name=None):
        self.delimiter = None
        if filename[-5:] != ".fits":
            self.__loadASICII__(filename, name=name)
        else:
            self.__loadFits__(filename, name=name)
        # if "ASICII" in kword or "asicii" in kword:
        if "barycentric" in kword:
            self.baryCorrection()

        self.estimateSNR()

    def __loadFits__(self, filename, name=None):
        if filename[0] == "/": filename = filename[1:]
        self.filename = filename
        spec_name = filename.split('/')[-1]
        if name is None: self.specname = spec_name.split("_")[0]
        else: self.specname=str(name)

        hdu = fits.open(os.path.join(Setting.datadir, filename))
        header = hdu[0].header
        self.header = header
        self.starname = header["OBJECT"]
        self.date = header["DATE-OBS"]
        self.v_bary = header["HIERARCH ESO QC VRAD BARYCOR"]

        # wave, flux, and labels
        flux = hdu[0].data
        crval1 = header["CRVAL1"]
        cdelt1 = header["CDELT1"]
        nwave = len(flux)
        grid = np.arange(0, nwave, 1)
        wave = (grid) * cdelt1 + crval1
        velocity = np.zeros_like(wave)
        masked = np.zeros_like(wave)
        self.df = pd.DataFrame(list(zip(wave, flux, velocity, masked)),
                          columns=["wave","flux","velocity","masked"])


        self.xlim = [np.min(wave), np.max(wave)]
        self.y_unit = "Flux"
        self.x_unit = "AA"
        self.reference_frame = "Geocentric"

        # processing markers
        self.normalized = False
        self.velocity_center = None
        self.SNR = None
        self.v_shift = 0.0

    def __loadASICII__(self, filename, name=None):
        self.filename = filename
        if name is not None:
            self.specname = str(name)
        else:
            name = filename.split("/")[-1]
            self.specname = name.split(".")[0]
        self.v_bary = None

        # try to determine delimiter automatically
        if self.delimiter is None:
            f = open(filename)
            header = f.readline()
            test_line = f.readline().replace("\n", " ")
            f.close()
            for item in "1234567890. ":
                test_line = test_line.replace(item, "")
            delimiter = list(set(test_line))
            if len(delimiter) == 0:
                self.delimiter = " "
            else:
                if len(delimiter) == 1:
                    self.delimiter = delimiter[0]
                else:
                    str_show = "Delimiter of " + filename.split("/")[-1] + "?\n"
                    str_show+= "Possible delimiters include: " + " ".join(delimiter)
                    self.delimiter = input(str_show)

        if self.delimiter == " ":
            df = pd.read_csv(filename, delim_whitespace=True)
        else:
            df = pd.read_csv(filename, delimiter=delimiter)

        columns = df.columns
        wave = pd.to_numeric(df[columns[0]], downcast="float").to_list()
        flux = pd.to_numeric(df[columns[-1]], downcast="float").to_list()
        velocity = np.zeros_like(wave)
        masked = np.zeros_like(wave)
        self.df = pd.DataFrame(list(zip(wave, flux, velocity, masked)),
                          columns=["wave","flux","velocity","masked"])

        self.xlim = [np.min(wave), np.max(wave)]
        self.y_unit = "Flux"
        self.x_unit = "AA"
        self.reference_frame = "Unknown"

        # processing markers
        self.normalized = False
        self.velocity_center = None
        self.SNR = None
        self.v_shift = 0.0

    def resetSpectrum(self):
        message = "Abort handling of spectrum {name}?[Y/N]".format(name=self.specname)
        if DataHandling.go_message(message):
            if self.delimiter is None:
                self.__loadFits__(filename=self.filename, name=self.specname)
            elif self.delimiter is not None:
                self.__loadASICII__(filename=self.filename, name=self.specname)
            return True
        else: return False

    def reName(self, name=""):
        while not name: name = input("Please type new name: ")
        self.specname = str(name)

    # Boundary related and data IO
    def __guessFrame__(self, left, right, x_in=None):
        if self.velocity_center is None or x_in == "wave": return "wave"
        else:
            boundary = [left, right]
            while None in boundary: boundary.pop(boundary.index(None))
            if x_in == "velocity" or np.min(boundary) <= 0: return "velocity"
            else:
                if len(boundary) == 2:
                    ind_wav = self.df.wave.between(left, right, inclusive=True)
                    ind_vel = self.df.velocity.between(left, right, inclusive=True)
                    if np.sum(ind_wav) < np.sum(ind_vel): return "velocity"
                    else: return "wave"
                if len(boundary) == 1 and \
                        np.min(self.df.veloct.to_list()) <= boundary[0] <= np.max(self.df.velocity.to_list()):
                    return "velocity"
                else: return "wave"

    def __getBoundary__(self, left, right, x_in=None):
        x_in = self.__guessFrame__(left, right, x_in=x_in)
        if left is None:
            left = self.xlim[0]
        elif x_in == "velocity":
            left = self.velocity_center * (1 + left / cst.c.to('km/s').value)

        if right is None:
            right = self.xlim[1]
        elif x_in == "velocity":
            right = self.velocity_center * (1 + right / cst.c.to('km/s').value)

        if not left < right: left, right = right, left

        return left, right

    def __getInd__(self, left, right, x_in=None):
        left, right = self.__getBoundary__(left, right, x_in=x_in)
        return self.df.wave.between(left, right, inclusive=True)

    def getData(self, x_in=None, x_out=None, apply_mask=False, xlim=None):
        """
        return tuple (X,Y). If apply_mask, will only return not-masked points
        :param apply_mask: emit masked points
        :type apply_mask: bool
        :param x_in: frame of x input, "wave" or "velocity"
        :param x_out: frame of x output, "wave" or "velocity"
        :param xlim: x boundaries
        :return: data_tuple = (X,Y)
        """
        if xlim is None:
            xlim = self.xlim
            x_in = "wave"
        if self.velocity_center is not None and x_out != "wave": x_out="velocity"
        else: x_out = "wave"

        ind = self.__getInd__(xlim[0], xlim[1], x_in=x_in)
        if apply_mask: ind = ind & (self.df.masked == 0)
        if x_out == "wave":
            x = np.array(self.df[ind].wave.to_list())
        else:
            x = np.array(self.df[ind].velocity.to_list())
        y = np.array(self.df[ind].flux.to_list())
        masked = np.array(self.df[ind].masked.to_list())
        return x, y, masked

    # wave and velocity
    def cutSpectrum(self, xmin=None, xmax=None, x_in=None):
        xmin, xmax = self.__getBoundary__(xmin, xmax, x_in=x_in)
        if not xmin < xmax: xmin, xmax = xmax, xmin
        if not self.df.wave.min() <= xmin <= self.df.wave.max(): xmin = self.df.wave.min()
        if not self.df.wave.min() <= xmax <= self.df.wave.max(): xmax = self.df.wave.max()
        self.xlim = [xmin, xmax]
        return self.xlim

    def setVelocityCenter(self, center=None):
        while center is None: center = input("Type center wavelength in AA:")
        center = float(center)
        if not self.xlim[0] <= center <= self.xlim[1]:
            message = "Warning! Center for spectrum {name} is outside working region, continue?[Y/N]" \
                .format(name=self.specname)
            if not DataHandling.go_message(message):
                return False

        self.df.velocity = (self.df.wave - center) / center * cst.c.to('km/s').value
        self.velocity_center = center
        self.x_unit = "km/s"
        return True

    def removeVelocityCenter(self):
        self.velocity_center = None
        self.df.velocity = self.df.velocity - self.df.velocity
        self.x_unit = "AA"

    def shiftSpectrum(self, v_offset=None):
        while v_offset is None:
            v_offset = input('Type velocity offset for {name} in km/s:'.format(name=self.specname))
        v_offset = float(v_offset)

        self.df.wave = self.df.wave * (1 + v_offset / cst.c.to('km/s').value)
        if self.velocity_center is not None:
            self.df.velocity = self.df.velocity + v_offset

        self.v_shift+= v_offset
        if self.v_bary is not None:
            if self.v_shift == 0: self.reference_frame = "Geocentric"
            elif self.v_shift == self.v_bary: self.reference_frame = "Barycentric"
            else: self.reference_frame = "Customized"

    def baryCorrection(self):
        if self.delimiter is not None:
            print("Spectrum {name} does not have barycentric information!".format(name=self.specname))
            return False
        else:
            if self.reference_frame == "Barycentric":
                print("{name} already in barycentric frame.".format(name=self.specname))
                return False
            else:
                if self.v_shift != 0:
                    self.resetVelocity()
                self.shiftSpectrum(v_offset = self.v_bary)
                self.reference_frame = "Barycentric"

    def resetVelocity(self):
        self.shiftSpectrum(v_offset = -1 * self.v_shift)

    # flux
    def fitContinuum(self, mode='spline', n=3,
                     lsigma=1, usigma=2, iterates=30, min_sigma = 0.5,
                     silence=False, apply_mask=False):

        x, y, masked = self.getData(apply_mask=False, x_out="wave")
        (cont, idx_cont) = DataHandling.iterate_continuum((x, y),
                                                          mode=mode, n=n, lsigma=lsigma, usigma=usigma,
                                                          iterates=iterates, min_sigma=min_sigma,
                                                          mask=masked * apply_mask)
        go_flag = True
        if not silence:
            cont_x, cont_y = x[idx_cont], y[idx_cont]
            go_flag = self.__vetContinuum__(cont, cont_x, cont_y)
        if go_flag:
            self.df.flux = self.df.flux / cont(self.df.wave)
            self.normalized = True
            self.y_unit = "Norm Flux"
            return True
        else:
            return False

    def fitSplineContinuum(self, xpoints=None, ypoints=None, silence=False):
        if len(xpoints) == len(ypoints):
            cont = CubicSpline(xpoints, ypoints)
            go_flag = True
            if not silence:
                go_flag = self.__vetContinuum__(cont, xpoints, ypoints)

            if go_flag:
                self.df.flux = self.df.flux / cont(self.df.wave)
                self.normalized = True
                self.y_unit = "Norm Flux"
                return True
            else:
                return False

    def __vetContinuum__(self, cont, cont_x, cont_y):
        x, y, masked = self.getData(apply_mask=False, x_out="wave")
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
        # origional
        ax1.plot(x, y, color="k")
        ax1.plot(x, cont(x), color='orange')
        ax1.set_ylabel("Original")

        # normalized
        ax2.plot(x, y / cont(x), color='k')
        ax2.plot(x, np.ones_like(x), linestyle='--', color='orange')
        ax2.set_ylabel("Normalized")
        ax2.set_xlabel("AA")

        # highlights
        ax1.scatter(cont_x, cont_y, marker="o", s=40, color="blue")
        ax2.scatter(cont_x, (cont_y / cont(cont_x)), marker="o", s=40, color="blue")
        if np.sum(masked) > 0:
            ax1.scatter(x[masked == 1], y[masked == 1], marker="x", s=40, color="red")
            ax2.scatter(x[masked == 1], (y / cont(x))[masked == 1], marker="x", s=40, color="red")

        # finalize
        ax1.grid()
        ax2.grid()
        plt.show()
        return DataHandling.go_message('Keep this continuum?[Y/N]')

    def convertToOpticalDepth(self):
        if not self.normalized:
            print("Spectrum {name} has not been normalized.".format(name=self.specname))
            return False
        if self.y_unit == "Opt Dep":
            print("Spectrum {name} already in optical depth.".format(name=self.specname))
            return False

        if np.min(self.flux) <= 0:
            print("Negative points detected in spectrum {name}, removed."
                  .format(name=self.specname))
            ind_delete = self.df[self.df["flux"] <= 0].index
            self.df = self.df.drop(ind_delete)
        self.df.flux = -1 * np.log(self.df.flux)
        self.y_unit = "Opt Dep"
        return True

    def convertToFlux(self):
        if self.y_unit != "Opt Dep":
            print("Spectrum {name} is not in optical depth.".format(name=self.specname))
            return False

        self.df.flux = np.exp(-1 * self.df.flux)
        self.y_unit = "Norm Flux"

    # mask
    def addMask(self, boundary_left, boundary_right, x_in=None, reset=True):
        if reset: self.resetMask()
        boundary_left, boundary_right = \
            DataHandling.parseInput(1, boundary_left, boundary_right, checklen=False)
        assert len(boundary_left) == len(boundary_right), "Boundaries must be in pairs."
        for i in range(len(boundary_left)):
            ind = self.__getInd__(boundary_left[i],boundary_right[i],x_in=x_in)
            self.df.loc[ind, "masked"] = 1

    def resetMask(self):
        self.df.masked = self.df.masked * 0
        return True

    # other utilities
    def estimateSNR(self, xlim=None):
        x,y,masked=self.getData(apply_mask=True, xlim=xlim, x_out="wave")
        SNR, LAM = DataHandling.measure_snr(x, y, do_plot=False)
        max_SNR = np.nanmax(DataHandling.smooth(SNR,10))
        self.SNR = max_SNR
        return self.SNR

    def showSpectrum(self):
        x,y,masked = self.getData(apply_mask=0)
        fig, ax = plt.subplots()
        ax.plot(x,y,color="k")
        ax.scatter(x[masked==1], y[masked==1], marker="x", s=50, color="red")
        if self.normalized:
            if self.y_unit == "Opt Dep":
                ax.plot(x, np.zeros_like(x), linestyle="--", color="orange")
                ax.invert_yaxis()
            else:
                ax.plot(x, np.ones_like(x), linestyle="--", color="orange")
        ax.set_ylabel(self.y_unit)
        ax.set_xlabel(self.x_unit)
        plt.show()

    def __str__(self):
        str_out = self.specname + ": \n"
        str_out+= "X: {min:.2f} to {max:.2f} ".format(min=self.xlim[0], max=self.xlim[1])\
                  + self.x_unit
        str_out+= ", " + self.reference_frame + " frame\n"
        str_out+= "Y: " + self.y_unit + "\n"
        return str_out


class EdiblesSpectrum:
    """
    A collection of tools for basic data handling.
    """
    # basic, load data and line list
    def __init__(self, filename, *kword, name=None):
        # log system
        self.log = ""
        # data
        self.specdata=[]
        self.loadSpectrum(filename, *kword, panel_name=name)
        # line list related
        self.linelist = {}
        self.color_map = ["m", "g", "c", "tan", "y",  "teal"]
        self.loadLineList()
        # model related
        self.model = None
        self.kernel = None
        self.model_fit = False

    def loadSpectrum(self, filename, *kword, clear_all=False, panel_name=None):
        filenames = DataHandling.parseInput(1, filename, checklen=False)
        panel_names = DataHandling.parseInput(len(filenames), panel_name)

        if clear_all and DataHandling.go_message("Abort all spectral data?"):
            self.__popAllPanels__()

        # basic info
        for i, filename in enumerate(filenames):
            if panel_names[i] is None: panel_name = "Panel {i}".format(i=len(self.specdata))
            else: panel_name = panel_names[i]
            newspec = SingleSpectrum(filename, *kword, name=panel_name)
            self.specdata.append(newspec)
            self.__addLog__(command="loadSpectrum",filename=filenames[i], name=panel_name, kword=kword)

    def loadLineList(self, list_name=None, reset=False):
        if reset: self.linelist = {}
        if list_name is None: list_name = Setting.edibles_linelist

        with open(list_name) as f:
            while True:
                line = f.readline().replace("\n", "")
                if not line: break
                (specie, wavelength) = line.split(": ")
                self.linelist[specie] = np.fromstring(wavelength, sep=",")

    def addHighLight(self, species="", wavelength=[0.0]):
        species = str(species)
        wavelength = DataHandling.parseInput(1, wavelength, checklen=0)
        self.linelist[species] = wavelength

    # common auxiliary methods and log commands
    def __parsePanelsInput__(self, panels=None):
        if panels is None:
            panels = np.arange(len(self.specdata))
        else:
            panels = DataHandling.parseInput(1, panels, checklen=False)
        panels = np.array(panels).astype("int32")
        panels = np.unique(panels)
        return panels

    def __addLog__(self, command="", **kwargs):
        log_str = ">>" + command + ": "
        while len(kwargs.keys()) > 0:
            item = kwargs.popitem()
            item_key = item[0]
            item_value = item[1]
            if type(item_value) == type(np.array([1,2,3])):
                value_str = np.array2string(item_value, separator=", ")
                log_str+= "=".join([item_key, value_str])
            else:
                log_str+= "=".join([item_key, str(item_value)])
            log_str+= ", "
        self.log+= log_str + "\n"
        return None

    def addComment(self, comment=""):
        if not comment:
            comment = input("addComment:")
        while len(comment)>2 and comment[0:2]==">>":
            comment = comment[2:]
        self.log+= comment + "\n"

    def printLog(self):
        print(self.log)

    # panel manipulate
    def popPanels(self,panels=None):
        panels = self.__parsePanelsInput__(panels)
        panels[::-1].sort()

        if len(panels) == len(self.specdata):
            if DataHandling.go_message("Pop ALL panels?[Y/N]"):
                self.__popAllPanels__()
                return None

        for panel in panels:
            message = "Pop panel {name}?[Y/N]".format(name=self.specdata[panel].specname)
            if DataHandling.go_message(message):
                self.specdata.pop(panel)
                self.__addLog__(command="popPanels", panels=panel)

    def __popAllPanels__(self):
        self.specdata = []
        self.__addLog__(command="popAllPanels")

    def duplicatePanels(self, ntimes=1, panels=None):
        panels = self.__parsePanelsInput__(panels)
        panels[::-1].sort()
        ntimes = DataHandling.parseInput(len(panels),ntimes)

        for i,panel in enumerate(panels):
            counter = 0
            while counter < ntimes[i]:
                self.specdata.insert(panel, copy.deepcopy(self.specdata[panel]))
                counter+= 1
        self.__addLog__(command="duplicatePanels", ntimes=ntimes, panels=panels)

    def renamePanels(self, names="", panels=None):
        panels = self.__parsePanelsInput__(panels)
        names = DataHandling.parseInput(len(panels), names)
        for i, panel in enumerate(panels):
            self.specdata[panel].reName(name=names[i])
            self.__addLog__(command="renamePanels", names=names[i], panels=panel)

    def resetSpectrum(self, panels=None):
        panels = self.__parsePanelsInput__(panels)
        for panel in panels:
            if self.specdata[panel].resetSpectrum():
                self.__addLog__(command="resetSpectrum", panels=panel)

    # change mask
    def addMask(self, n=None, reset=True, panels=None, boundary_left=None, boundary_right=None):
        panels = self.__parsePanelsInput__(panels)
        ns, resets = DataHandling.parseInput(len(panels), n, reset)
        if boundary_left is not None and boundary_right is not None:
            for i, panel in enumerate(panels):
                self.specdata[panel].addMask(boundary_left, boundary_right, reset=resets[i])
                self.__addLog__(command="addMask", reset=resets[i],
                                boundary_left=boundary_left, boundary_right=boundary_right, panels=panel)
            return None

        for i, panel in enumerate(panels):
            if resets[i]: self.resetMask(panels=panel)
            x,y,masked = self.specdata[panel].getData(apply_mask=False)
            n_now = ns[i]

            while True:
                while n_now is None:
                    n_now = np.int(input("Number of regions to be masked for {name}?"
                                         .format(name=self.specdata[panel].specname)))
                if n_now <= 0:
                    import matplotlib
                    matplotlib.use("module://backend_interagg")
                    import matplotlib.pyplot as plt
                    break
                else:
                    message = "Select boundaries of regions to mask"
                    x_select, y_select = self.__getPoints__(n=n_now*2, message=message, panel=panel)
                    boundary_left = x_select[0::2]
                    boundary_right = x_select[1::2]
                    idx_masked = np.zeros_like(x)
                    for j in range(len(boundary_left)):
                        idx = (boundary_left[j] <= x) & (x <= boundary_right[j])
                        idx_masked[idx] = 1

                    # why do I need to re-import?
                    import matplotlib.pyplot as plt
                    fig2, ax = plt.subplots(1,1)
                    ax.plot(x, y, color="k")
                    ax.scatter(x[idx_masked == 1], y[idx_masked == 1], marker="x", color="red")
                    ax.grid()
                    self.__drawXLabel__(ax=ax, panel=panel)
                    self.__drawYLabel__(ax=ax, panel=panel)
                    plt.show()

                    if DataHandling.go_message("Remove these points?[Y/N]"):
                        if self.specdata[panel].velocity_center is None: x_in = "wave"
                        else: x_in = "velocity"
                        self.specdata[panel].addMask(boundary_left, boundary_right, reset=resets[i], x_in=x_in)
                        self.__addLog__(command="addMask", reset=resets[i],
                                        boundary_left=boundary_left, boundary_right=boundary_right)
                        break
                    else:
                        n_now = None
                # if n_now > 0
            # while True
        # for panel in panels
        return None

    def resetMask(self, panels=None):
        panels = self.__parsePanelsInput__(panels)
        for panel in panels:
            if self.specdata[panel].resetMask():
                self.__addLog__(command="resetMask", panels=panel)

    # change wave:
    def cutSpectrum(self, xmin=None, xmax=None, center=None, span=None, x_in=None, panels=None):
        panels = self.__parsePanelsInput__(panels)
        (xmin, xmax, center, span, x_in) = DataHandling.parseInput(len(panels), xmin, xmax, center, span, x_in)

        for i, panel in enumerate(panels):
            xmin_now, xmax_now, center_now, span_now, x_in_now = xmin[i], xmax[i], center[i], span[i], x_in[i]
            if center_now is not None and span_now is not None:
                xmin_now = center_now - 0.5 * span_now
                xmax_now = center_now + 0.5 * span_now
            xlim = self.specdata[panel].cutSpectrum(xmin=xmin_now, xmax=xmax_now, x_in=x_in_now)
            # RESULT of the cutting, rather than the inputs, will be recorded
            self.__addLog__(command="cutSpectrum", xmin=xlim[0], xmax=xlim[1], x_in="wave", panels=panel)

    def cutSpectrum_visual(self, panels=None):
        panels = self.__parsePanelsInput__(panels)
        # an interactive backends will be needed
        import matplotlib
        matplotlib.use('Qt5Agg', warn=False, force=True)
        import matplotlib.pyplot as tmp_plt

        for i, panel in enumerate(panels):
            x,y,masked = self.specdata[panel].getData()
            fig1, ax = tmp_plt.subplots(1, 1)
            ax.plot(x, y, marker=".", linestyle="--", linewidth=0.5)
            ax.scatter(x[masked==1], y[masked==1], marker="x", color="red")
            ax.set_xlim(1.1 * np.min(x) - 0.1 * np.max(x), 1.1 * np.max(x) - 0.1 * np.min(x))
            ax.grid()
            self.__drawXLabel__(ax=ax, panel=panel)
            self.__drawYLabel__(ax=ax, panel=panel)

            print("Select boundaries of working region for spectrum {name}"
                  .format(name=self.specdata[panel].specname))
            points = tmp_plt.ginput(2, mouse_add=1, mouse_pop=3, mouse_stop=2)

            tmp_plt.close(fig1)
            points_tuple = ([points[0][0], points[1][0]], [points[0][1], points[1][1]])
            cut_idx = DataHandling.nearest_point(points_tuple, (x, y))
            (xmin, xmax) = (x[cut_idx[0]], x[cut_idx[1]])
            if self.specdata[panel].velocity_center is None: x_in="wave"
            else: x_in="velocity"
            self.cutSpectrum(xmin=xmin, xmax=xmax, x_in=x_in, panels=panel)

        # now let's switch back to the normal backend
        matplotlib.use("module://backend_interagg")
        import matplotlib.pyplot as plt
        return None

    def cutSpectrumTo(self, cut=None, to=0):
        cuts = self.__parsePanelsInput__(cut)
        tos = DataHandling.parseInput(len(cuts), to)

        for i, cut in enumerate(cuts):
            xlim = sp.specdata[to].xlim
            self.cutSpectrum(xmin=xlim[0], xmax=xlim[1], x_in="wave", panels=cut)

    def shiftSpectrum(self,v_offset = None, panels=None):
        panels = self.__parsePanelsInput__(panels)
        v_offset = DataHandling.parseInput(len(panels), v_offset)

        for i, panel in enumerate(panels):
            self.specdata[panel].shiftSpectrum(v_offset=v_offset[i])
            self.__addLog__(command="shiftSpectrum", v_offset=v_offset[i], panels=panel)

    def resetVelocity(self, panels=None):
        panels = self.__parsePanelsInput__(panels)
        for panel in panels:
            self.specdata[panel].resetVelocity()
        self.__addLog__(command="resetVelocity", panels=panels)

    def baryCorrection(self, panels=None):
        panels = self.__parsePanelsInput__(panels)
        for panel in panels:
            self.specdata[panel].baryCorrection()
        self.__addLog__(command="baryCorrection", panels=panels)

    def setVelocityCenter(self, center=None, panels=None):
        panels = self.__parsePanelsInput__(panels)
        center = DataHandling.parseInput(len(panels), center)

        for i, panel in enumerate(panels):
            self.specdata[panel].setVelocityCenter(center=center[i])
            self.__addLog__(command="setVelocityCenter", center=center[i], panels=panel)

    def removeVelocityCenter(self, panels=None):
        panels = self.__parsePanelsInput__(panels)
        for panel in panels:
            self.specdata[panel].removeVelocityCenter()
        self.__addLog__(command="removeVelocityCenter", panels=panels)

    # change flux
    def fitContinuum(self, mode='spline', n=3,
                     lsigma=1, usigma=2, iterates=30, min_sigma = 0.5,
                     silence=False, apply_mask=False, panels=None):

        panels = self.__parsePanelsInput__(panels)
        (mode, n, lsigma, usigma, iterates, min_sigma, silence, apply_mask) = \
            DataHandling.parseInput(len(panels), mode, n, lsigma, usigma, iterates, min_sigma, silence, apply_mask)

        for i, panel in enumerate(panels):
            if self.specdata[panel].fitContinuum(mode=mode[i], n=n[i], lsigma=lsigma[i], usigma=usigma[i],
                                              iterates=iterates[i], min_sigma=min_sigma[i],
                                              silence=silence[i], apply_mask=apply_mask[i]):
                self.__addLog__(command="fitContinuum", mode=mode[i], n=n[i], lsigma=lsigma[i], usigma=usigma[i],
                                iterates=iterates[i], min_sigma=min_sigma[i],silence=silence[i],
                                apply_mask=apply_mask[i], panel=panel)

    def fitSplineContinuum(self, step=1.0, silence=False, manual=False, xpoints=None, ypoints=None, apply_mask=True, panels=None):
        panels = self.__parsePanelsInput__(panels)
        step, silence, manual, apply_mask = DataHandling.parseInput(len(panels), step, silence, manual, apply_mask)
        for i, panel in enumerate(panels):
            if xpoints is None or ypoints is None:
                x, y, masked = self.specdata[panel].getData(x_out="wave", apply_mask=0)
                if not apply_mask[i]: masked = np.zeros_like(x)
                if xpoints is None:
                    if manual[i]:
                        xpoints, ypoints = self.__getPoints__(n=99, x_out="wave",
                                                              message="add as many points as you want", panel=panel)
                    else:
                        xpoints, ypoints = DataHandling.getSplinePoints(x, y, masked, step=step[i])
                x,y = x[np.where(masked==0)], y[np.where(masked==0)]
                spline_cont = eM.SplineContinuum(name="spline_cont", n_point=len(xpoints), x_anchor=xpoints)
                spline_cont.guess(x, y)
                data2fit = Data1D('data2fit', x, y)
                fit = Fit(data2fit, spline_cont, stat=LeastSq(), method=NelderMead())
                result = fit.fit()
                ypoints = spline_cont.y_used

            if self.specdata[panel].fitSplineContinuum(xpoints=xpoints, ypoints=ypoints, silence=silence[i]):
                    self.__addLog__(command="fitSpoineContinuum",
                                xpoints=xpoints, ypoints=ypoints, apply_mask=apply_mask[i],
                                silence=silence[i], panel=panel)

    def convertToOpticalDepth(self, panels=None):
        panels = self.__parsePanelsInput__(panels)
        for panel in panels:
            self.specdata[panel].convertToOpticalDepth()
        self.__addLog__(command="convertToOpticalDepth", panels=panels)

    def convertToFlux(self, panels=None):
        panels = self.__parsePanelsInput__(panels)
        for panel in panels:
            self.specdata[panel].convertToFlux()
        self.__addLog__(command="convertToFlux", panels=panels)

    # flux info
    def __getPoints__(self, n=2, x_out=None, message="", panel=0):
        if self.specdata[panel].velocity_center is not None and x_out != "wave": x_out="velocity"
        else: x_out = "wave"
        x, y, masked = self.specdata[panel].getData(x_out=x_out)

        import matplotlib
        matplotlib.use('Qt5Agg', warn=False, force=True)
        import matplotlib.pyplot as tmp_plt

        fig1, ax = tmp_plt.subplots(1, 1)
        ax.plot(x, y, marker=".", linestyle="--", linewidth=0.5)
        ax.scatter(x[masked == 1], y[masked == 1], marker="x", color="red")
        ax.set_xlim(1.1 * np.min(x) - 0.1 * np.max(x), 1.1 * np.max(x) - 0.1 * np.min(x))
        self.__drawYLabel__(ax=ax.axes, panel=panel)
        if x_out =="wave": x_label = "AA"
        else: x_label = "km/s"
        ax.set_xlabel(x_label)
        ax.set_ylabel(message)
        if message: print(message)
        print("Left = add; Right = pop; Middle = stop\n")
        timeout = np.max([n * 5, 30])
        points = tmp_plt.ginput(n, mouse_add=1, mouse_pop=3, mouse_stop=2, timeout=timeout)

        points_x, points_y = [], []
        for j in range(len(points)):
            points_x.append(points[j][0])
            points_y.append(points[j][1])
        points_idx = DataHandling.nearest_point((points_x, points_y), (x, y))
        x_select, y_select = x[points_idx], y[points_idx]
        x_select, y_select = x_select[np.argsort(x_select)], y_select[np.argsort(x_select)]

        tmp_plt.close(fig1)
        matplotlib.use("module://backend_interagg")
        import matplotlib.pyplot as plt

        return x_select, y_select

    def getWave(self, *kword, n=None, panel=0, message=None):
        while n is None:
            n = np.int(input("Number of target points?"))
        if message is None: message = "Select target points"
        x_select, y_select = self.__getPoints__(n=n, x_out="wave", message=message, panel=panel)

        if "voigt" in kword:
            y_out = []
            for i, x in enumerate(x_select):
                alpha = 1.5 * x / cst.c.to('km/s').value * np.sqrt(2. * np.log(2.))
                gamma = 0.0005
                voigt_peak = eMath.voigtMath(0, alpha, gamma)
                y_out.append(math.log(y_select[i]) / voigt_peak * (-1))
            return (x_select, np.array(y_out))
        elif "gaussian" in kword:
            y_out = []
            for y in y_select: y_out.append(1 - y)
            return (x_select, np.array(y_out))
        else:
            return x_select

    def getVelocity(self, *kword, n=None, wavelength=None, message=None, panel=0):
        while n is None:
            n = np.int(input("Number of target points?"))
        while wavelength is None:
            wavelength = np.float(input("Velocity center in AA?"))
        wavelength = np.array(DataHandling.parseInput(1,wavelength,checklen=False))
        if message is None: message = "Select target points"
        x_select, y_select = self.__getPoints__(n=n, x_out="wave", message=message, panel=panel)

        if len(wavelength) == 1:
            x_out = (x_select / wavelength[0] - 1) * cst.c.to('km/s').value
        else:
            sub_length = math.floor(len(x_select) / len(wavelength))
            n_subs = math.floor(len(x_select) / sub_length)
            boundary = (np.arange(n_subs) + 1) * sub_length
            x_split = np.split(x_select, boundary)
            x_out = []
            for i in range(n_subs):
                x_out.append((x_split[i] / wavelength[i] - 1) * cst.c.to('km/s').value)

        if "voigt" in kword:
            y_out = []
            for i, x in enumerate(x_select):
                alpha = 1.5 * x / cst.c.to('km/s').value * np.sqrt(2. * np.log(2.))
                gamma = 0.0005
                voigt_peak = eMath.voigtMath(0, alpha, gamma)
                y_out.append(math.log(y_select[i]) / voigt_peak * (-1))
            return (x_out, np.array(y_out))
        elif "gaussian" in kword:
            y_out = []
            for y in y_select: y_out.append(1 - y)
            return (x_out, np.array(y_out))
        else:
            return x_out

    def searchPeaks(self, n=None, prominence=3, panels=None):
        panels = self.__parsePanelsInput__(panels)
        x, y, SNR = self.getData(apply_mask=True, x_out="wave", panels=panels)
        peak_idx = DataHandling.searchPeak(y, n_peaks=n, normalized=True,
                                           SNR=np.max(SNR), prominence=prominence)
        peak_wavelengths = x[peak_idx]
        peak_fluxs = y[peak_idx]

        self.addHighLight(species="Peaks", wavelength=peak_wavelengths)
        self.showSpectrum(model=False, highlight_draw=True,highlight_species="Peaks", panels=panels)
        self.linelist.pop("Peaks")

        return peak_wavelengths

    def estimateSNR(self, xlim=None, panels=None):
        panels = self.__parsePanelsInput__(panels)
        xlim = DataHandling.parseInput(len(panels), xlim, checklen=False)
        SNR = []
        for i, panel in enumerate(panels):
            SNR.append(self.specdata[panel].estimateSNR(xlim=xlim[i]))
        return SNR

    # model fitting
    def getData(self, apply_mask=False, xlim=None, x_in=None, x_out=None, panels=None):
        panels = self.__parsePanelsInput__(panels)
        apply_mask, x_in = DataHandling.parseInput(len(panels), apply_mask, x_in)
        xlim = DataHandling.parseInput(len(panels), xlim, checklen=False)
        vc = []
        for panel in panels: vc.append(self.specdata[panel].velocity_center)
        if None not in vc and x_out != "wave": x_out = "velocity"
        else: x_out = "wave"

        if len(panels) == 1:
            x,y,masked = self.specdata[panels[0]].\
                getData(xlim=xlim[0], x_in=x_in[0], x_out=x_out, apply_mask=apply_mask[0])
            SNR = self.specdata[panels[0]].SNR
            return x,y,np.ones_like(x)*SNR
        else:
            if x_out == "wave":
                x, y, masked = self.specdata[panels[0]].\
                    getData(xlim=xlim[0], x_in=x_in[0], x_out=x_out, apply_mask=apply_mask[0])
                SNR = np.ones_like(x) * self.specdata[panels[0]].SNR
                for i, panel in enumerate(panels[1:]):
                    x_panel, y_panel, masked_panel = self.specdata[panel]. \
                        getData(xlim=xlim[i+1], x_in=x_in[i+1], x_out=x_out, apply_mask=apply_mask[i+1])
                    SNR_panel = np.ones_like(x_panel) * self.specdata[panel].SNR
                    x, y, SNR = DataHandling.MergeSpectra(x,y,x_panel,y_panel,SNR1=SNR,SNR2=SNR_panel)
                return x, y, SNR

            if x_out == "velocity":
                x, y, SNR = [], [], []
                for i, panel in enumerate(panels):
                    x_panel, y_panel, masked_panel = self.specdata[panel]. \
                        getData(xlim=xlim[i], x_in=x_in[i], x_out=x_out, apply_mask=apply_mask[i])
                    SNR_panel = np.ones_like(x) * self.specdata[panel].SNR
                    x, y,SNR = x.append(x_panel), y.append(y_panel), SNR.append(SNR_panel)
                return np.array(x), np.array(y), np.array(SNR)

    def importModel(self, model, silence=False):
        self.model = copy.deepcopy(model)
        if not silence:
            print("New model imported!")
            print(model)
            self.showSpectrum(masked=False, highlight_draw=False, linebyline=True)
        self.__addLog__(command="importModel")
        self.addComment(comment=str(model))
        return None

    def getKernel(self, resolution=80000, n_sigma=5, apply_mask=False, panels=None):
        panels = self.__parsePanelsInput__(panels)
        x,y,SNR = self.getData(apply_mask=apply_mask, panels=panels, x_out="wave")
        dx = np.median(x[1:] - x[0:-1])
        x_mean = np.median(x)

        k_sigma = x_mean / resolution / 2.35482
        n_steps = (n_sigma * k_sigma) // dx + 1
        k_x = np.arange(-n_steps, n_steps + 1, 1) * dx
        z = (k_x / k_sigma) ** 2
        kernel = np.exp(- z / 2)
        kernel = kernel / np.sum(kernel)
        #self.kernel = kernel
        return kernel

    def __getKernelOffset__(self, apply_mask=False, panels=None, kernel=None, xlength=None):
        panels = self.__parsePanelsInput__(panels)
        if kernel is None: kernel = self.getKernel(apply_mask=apply_mask, panels=panels)
        if xlength is None:
            x,y,SNR = self.getData(apply_mask=apply_mask, panels=panels)
            xlength = len(x)

        crval1 = self.specdata[panels[0]].header["CRVAL1"]
        cdelt1 = self.specdata[panels[0]].header["CDELT1"]
        x_test = np.arange(0, xlength, 1) * cdelt1 + crval1
        x_peak = x_test[math.floor(xlength/2)]

        model_test = eM.Cloud(name="Kernel_Test")
        model_test.addLines(x_peak, b=1.0)
        model_test.importInstrumental(kernel)
        conv_model = model_test.compileModel(add_instrumental=True)
        idx_peak_conv = np.argmin(conv_model(x_test))
        return idx_peak_conv - math.floor(xlength/2)

    def fitModel(self, link_b=None, link_d=None, freeze_d=None, opt="NelderMead",
                 apply_mask=False, panels=None):
        assert self.model is not None, "No model imported yet."
        stat_lib = ['LeastSq']
        opt_lib = ['LevMar', 'NelderMead']
        assert opt in opt_lib, "The opt you choose is not available"

        from sherpa.optmethods import LevMar, NelderMead
        from sherpa.data import Data1D
        from sherpa.fit import Fit

        opt_apply = eval(opt + "()")

        panels = self.__parsePanelsInput__(panels)
        x,y,SNR = self.getData(apply_mask=apply_mask, panels=panels)
        n_offset = self.__getKernelOffset__(apply_mask=apply_mask, panels=panels)
        #dx = np.median(x[1:-1] - x[0:-2])
        #v_offset = dx/np.median(x) * cst.c.to("km/s").value * n_offset
        v_offset = (x[1] - x[0]) / np.median(x) * cst.c.to("km/s").value * n_offset
        model_backup = copy.deepcopy(self.model)

        model2fit = self.model.compileModel(link_b=link_b, link_d=link_d,
                                            freeze_d=freeze_d, conv_correction=v_offset)
        data2fit = Data1D('data2fit', x, y, 1/SNR)
        fit = Fit(data2fit, model2fit, method=opt_apply)
        result = fit.fit()

        # make result plots
        n_rows = len(panels)
        v_centers = []
        for panel in panels: v_centers.append(self.specdata[panel].velocity_center)
        if None in v_centers: fig, axs = plt.subplots(nrows=n_rows, ncols=2)
        else: fig, axs = plt.subplots(nrows=n_rows, ncols=2, sharex="row")

        for i, panel in enumerate(panels):
            wave, fulx, masked = self.specdata[panel].getData(apply_mask=False, x_out="wave")
            x,y, masked = self.specdata[panel].getData(apply_mask=False)
            # result
            axs = fig.axes[i*2]
            axs.plot(x, y, color="k")

            # if the fitting uses data from two panels, then the offset of each panel and the combined
            # data would be different!
            #n_offset = self.__getKernelOffset__(xlength=len(x), apply_mask=False, panels=panel)
            #dx = np.median(x[1:-1] - x[0:-2])
            #v_offset = dx / np.median(x) * cst.c.to("km/s").value * n_offset

            n_offset = self.__getKernelOffset__(apply_mask=apply_mask, panels=panel)
            # dx = np.median(x[1:-1] - x[0:-2])
            # v_offset = dx/np.median(x) * cst.c.to("km/s").value * n_offset
            v_offset = (x[1] - x[0]) / np.median(x) * cst.c.to("km/s").value * n_offset

            model_plot_panel = self.model.compileModel(link_b=link_b, link_d=link_d,
                                    freeze_d=freeze_d, conv_correction=v_offset)
            axs.plot(x, model_plot_panel(wave), color="red")
            axs.plot(x, self.model.continuum(wave), linestyle='--', color="orange")

            axs.grid()
            self.__drawYLabel__(ax=axs, panel=panel)
            if apply_mask:
                idx = self.specdata[panel].masked == 1
                axs.scatter(x[idx], y[idx], marker="x", color="red")
            if i == n_rows - 1:
                axs.set_xlabel("Fitted Model")

            # residual
            axs = fig.axes[i*2 + 1]
            axs.plot(x, (model_plot_panel(wave) - y)/self.model.continuum(wave), color='k')
            axs.plot(x, np.zeros_like(x), linestyle='--', color="orange")
            if self.specdata[panel].SNR is None: self.specdata[panel].estimateSNR()
            axs.plot(x, np.ones_like(x)/self.specdata[panel].SNR, linestyle='--', color="blue")
            axs.plot(x, -1 * np.ones_like(x)/self.specdata[panel].SNR, linestyle='--', color="blue")
            if apply_mask:
                idx = self.specdata[panel].masked == 1
                axs.scatter(x[idx], model_plot_panel(wave[idx]) - y[idx], marker="x", color="red")
            axs.grid()
            if i == n_rows - 1:
                axs.set_xlabel("Residuals")
        plt.show()

        if DataHandling.go_message("Keep this fit?[Y/N]"):
            self.model_fit = True
            self.model.__afterFit__(link_b=link_b, link_d=link_d, freeze_d=freeze_d)
            self.__addLog__(command="fitModel", link_b=link_b, link_d=link_d, freeze_d=freeze_d,
                            opt=opt, apply_mask=apply_mask, panels=panels)
            return True, result
        else:
            self.model = model_backup
            return False, result

    def raiseParameter(self, line=None, par=None):
        if not self.model_fit:
            print("="*16)
            print("Warning! These values are not final!")
            print("=" * 16)

        name,val = self.model.raiseParameter(line=line, par=par)
        return name, val

    def reportEW(self, *kwords, nsigma = 5, dx = 0.01):
        assert self.model is not None, "No model imported!"
        if not self.model_fit:
            print("="*16)
            print("Warning! These values are not final!")
            print("=" * 16)

        line_EW = self.model.report_EW(*kwords, nsigma=nsigma, dx=dx)
        return line_EW

    # plotting
    def showSpectrum(self, x_label=True, y_label=True, continuum=True,
                     model=True, masked=True, linebyline=False, highlight_draw=True,
                     highlight_species="All", panels=None,
                     save=False, savepath=None, filename=None):
        panels = self.__parsePanelsInput__(panels)
        self.__makeBasicPlots__(x_label=x_label, y_label=y_label,
                                continuum=continuum, masked=masked,
                                model=model, linebyline=linebyline,
                                highlight_draw=highlight_draw,
                                highlight_species=highlight_species,
                                panels=panels)
        if save:
            if savepath is None: savepath = Setting.plotdir
            if not os.path.exists(savepath): os.makedirs(savepath)
            if filename is None: filename = input("Please type file name to be used:")
            plt.savefig(savepath + filename+".png")
        plt.show()
        return None

    def __makeBasicPlots__(self,x_label=True, y_label=True,
                           continuum=True, model=True, linebyline=False, masked=True,
                           highlight_draw=True, highlight_species="all",
                           panels=None):
        panels = self.__parsePanelsInput__(panels)
        n_plots = len(panels)
        x_labels, y_labels = DataHandling.parseInput(n_plots, x_label, y_label)
        continua, models, maskeds = DataHandling.parseInput(n_plots, continuum, model, masked)
        highlight_draws = DataHandling.parseInput(n_plots, highlight_draw)

        v_centers = []
        for panel in panels: v_centers.append(self.specdata[panel].velocity_center)
        if v_centers == [None] * n_plots:
            fig, axs = plt.subplots(nrows=n_plots)
            x_label_draw = [0] * (n_plots - 1) + [1]
        elif None in v_centers:
            fig, axs = plt.subplots(nrows=n_plots)
            x_label_draw = [1] * n_plots
        else:
            fig, axs = plt.subplots(nrows=n_plots, sharex=True)
            x_label_draw = [0] * (n_plots - 1) + [1]

        for i, panel in enumerate(panels):
            x,y, masked = self.specdata[panel].getData(apply_mask=False)
            fig = plt.gcf()
            axs = fig.axes[i]
            axs.plot(x, y, color="k")
            axs.grid()
            if x_label_draw[i]: self.__drawXLabel__(x_label=x_labels[i], ax=axs, panel=panel)
            if y_labels[i]: self.__drawYLabel__(y_label=y_labels[i], ax=axs, panel=panel)
            if continua[i]: self.__drawContinuum__(ax=axs, panel=panel)
            if maskeds[i]: self.__drawMasked__(ax=axs, panel=panel)
            if highlight_draws[i]: self.__drawHighlight__(ax=axs, panel=panel, highlight=highlight_species)
            if models[i]: self.__drawModel__(ax=axs, linebyline=linebyline, panel=panel)

        return None

    def __drawXLabel__(self,x_label=None, ax=None, panel=0):
        if ax is None: ax = plt.gca()
        if type(x_label) in [type("abc"), type(np.array(["abc"])[0])]:
            ax.set_xlabel(x_label)
        else: ax.set_xlabel(self.specdata[panel].x_unit)

    def __drawYLabel__(self,y_label=None, ax=None, panel=0):
        if self.specdata[panel].y_unit == "Opt Dep":
            ax.invert_yaxis()
        if type(y_label) in [type("abc"), type(np.array(["abc"])[0])]:
            ax.set_ylabel(y_label)
        else:
            ax.set_ylabel(", ".join([self.specdata[panel].specname, self.specdata[panel].y_unit]))

    def __drawContinuum__(self, ax=None, panel=0):
        if ax is None: ax = plt.gca()
        x,y,masked = self.specdata[panel].getData(apply_mask=0)
        if self.specdata[panel].normalized:
            if self.specdata[panel].y_unit == "Opt Dep":
                ax.plot(x, np.zeros_like(x), linestyle='--', color='orange')
            else:
                ax.plot(x, np.ones_like(x), linestyle='--', color='orange')

    def __drawMasked__(self, ax=None, panel=0):
        if ax is None: ax = plt.gca()
        x, y, masked = self.specdata[panel].getData(apply_mask=0)
        if np.sum(masked) > 0:
            ax.scatter(x[masked==1], y[masked==1], marker="x", color="red")

    def __drawHighlight__(self, ax=None, panel=0, highlight="all"):
        if ax is None: ax = plt.gca()
        x,y,masked = self.specdata[panel].getData(apply_mask=0, x_out="wave")
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        ymin, ymax = 0.25*ymax+0.75*ymin, 0.75*ymax+0.25*ymin

        if type(highlight) is str and highlight.lower() == "all": highlights = self.linelist.keys()
        elif type(highlight) is not list: highlights = [highlight]
        else: highlights = highlight

        for i,highlight in enumerate(highlights):
            color_idx = i % len(self.color_map)
            if highlight in self.linelist.keys():
                wavelengths = self.linelist[highlight]
                within, idx = DataHandling.within_boundaries(wavelengths, xmin, xmax)
                if np.sum(within) > 0:
                    wavelengths = wavelengths[idx]
                    if self.specdata[panel].velocity_center is not None:
                        wavelengths = (wavelengths - self.specdata[panel].velocity_center) / \
                                      self.specdata[panel].velocity_center * cst.c.to('km/s').value
                    for wavelength in wavelengths:
                        ax.plot([wavelength, wavelength], [ymax, ymin], color=self.color_map[color_idx])
                    y_text = ymin - 0.1 * (i%2 +1) * (ymax - ymin)
                    ax.text(np.mean(wavelengths), y_text, highlight, color=self.color_map[color_idx])
            else:
                print("{item} not found".format(item = str(highlight)))
                print("Available species include {list}".format(list = self.linelist.keys()))

    def __drawModel__(self, ax=None, linebyline=False, panel=0):
        if ax is None: ax = plt.gca()
        x,y,masked = self.specdata[panel].getData(apply_mask=0)
        if self.model is not None:
            n_offset = self.__getKernelOffset__(panels=panel, apply_mask=False)
            wave, flux, masked = self.specdata[panel].getData(apply_mask=False, x_out="wave")
            v_offset = (wave[1] - wave[0]) / np.median(wave) * cst.c.to("km/s").value * n_offset

            # plot individual lines
            if linebyline:
                model = copy.deepcopy(self.model)
                for line in model.lines:
                    if model.instrumental is not None:
                        line.kernel_offset.val = v_offset / cst.c.to('km/s').value * line.lam_0.val
                        line = model.instrumental(line) * model.continuum
                    ax.plot(x, line(wave), color="blue")
                for i, cloud in enumerate(model.clouds):
                    color = self.color_map[i]
                    for line in cloud.lines:
                        if model.instrumental is not None:
                            line.kernel_offset.val = v_offset
                            line = model.instrumental(line) * model.continuum
                        ax.plot(x, line(wave), color=color)
            # plot summed model only
            else:
                model = self.model.compileModel(link_b=None, link_d=None, freeze_d=None, conv_correction=v_offset)
                ax.plot(x, model(wave), color='blue')

    # other
    def __str__(self):
        str_out = "{n} panels:\n".format(n=len(self.specdata))
        panels = self.__parsePanelsInput__(panels=None)
        for panel in panels:
            str_out+= str(self.specdata[panel])
        return str_out


if __name__ == "__main__":
    filename = 'HD22951/BLUE_346/HD22951_w346_blue_20160829_O12.fits'
    sp = EdiblesSpectrum(filename)
    sp.baryCorrection()
    sp.cutSpectrum(xmin=3301, xmax=3304.5)
    sp.showSpectrum()

