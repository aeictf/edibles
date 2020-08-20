import numpy as np
import astropy.constants as cst
from scipy.interpolate import CubicSpline

from sherpa import models
from sherpa.models.model import ArithmeticModel
from sherpa.models.parameter import Parameter
from sherpa.data import Data1D
from sherpa.instrument import Kernel, ConvolutionKernel, ConvolutionModel
import sherpa.ui as UI

import edibles.src.math as eMath
import edibles.src.datahandling as DataHandling

class VoigtLine(ArithmeticModel):

    def __init__(self, name='VoigtLine',lam_0=5000, b=1.5, d=0.0005, N=999, f=999, N_mag=10, tau_0=0.1):

        self.N_mag = N_mag
        self.lam_0 = Parameter(name, 'lam_0', lam_0, frozen=False, min=0.0)
        self.b = Parameter(name, 'b', b, frozen=False, min=1e-12)
        self.d = Parameter(name, 'd', d, frozen=False, min=0)
        self.N = Parameter(name, 'N', N, frozen=True, hidden=True, min=0.0)
        self.f = Parameter(name, 'f', f, frozen=True, hidden=True, min=0.0)
        self.tau_0 = Parameter(name, 'tau_0', tau_0, frozen=False, min=0.0)
        self.kernel_offset = Parameter(name, 'kernel_offset', 0, frozen=True, hidden=True)

        ArithmeticModel.__init__(self, name,
                                 (self.lam_0, self.b, self.d,self.N, self.f, self.tau_0, self.kernel_offset))

        if N != 999 and f!= 999: self.__useNf__(N=N, f=f)

    def calc(self, pars, x, *args, **kwargs):
        lam_0, b, d, N, f, tau_0, kernel_offset = pars

        if N != 999 and f != 999:
            N = N * (10**self.N_mag)
            transmission = eMath.voigtOpticalDepthAbsorption(lam=x, lam_0=lam_0-kernel_offset, b=b, d=d, N=N, f=f)
        else:
            transmission = eMath.voigtOpticalDepthAbsorption(lam=x, lam_0=lam_0-kernel_offset, b=b, d=d, tau_0=tau_0)

        return transmission

    def FWHM(self):
        Gau_FWHM = 2.355 * self.b.val * self.lam_0.val / cst.c.to('km/s').value
        Lor_FWHM = self.d.val * 2
        Voigt_FWHM_tmp = 0.5346 * Lor_FWHM + np.sqrt(0.2166 * Lor_FWHM**2 + Gau_FWHM**2)
        n_point = round(Voigt_FWHM_tmp * 5 / 0.01)
        x = np.arange(n_point * 2 + 1)
        x = (x - np.median(x)) * 0.01 + self.lam_0.val
        y = self(x)
        CD = abs(1 - np.min(y))
        y = y[abs(1-y) >= 0.5*CD]
        Voigt_FWHM = (len(y) -1 ) * 0.01
        return Voigt_FWHM

    def report_EW(self, *kwords, nsigma = 10, dx = 0.01):
        Voigt_FWHM = self.FWHM()
        n_point = round(Voigt_FWHM * nsigma / dx)
        x = np.arange(n_point * 2 + 1)
        x = (x - np.median(x)) * dx + self.lam_0.val
        y = self(x)
        EW = eMath.integrateEW(x, y)
        if "mA" in kwords: EW = EW * 1000
        return EW

    def __useNf__(self, N=999, f=999):
        self.N.val = N
        self.N.thaw()
        self.N.hidden = False
        self.f.val = f
        self.f.hidden = False
        self.tau_0.freeze()
        self.tau_0.hidden = True

class VoigtLine_KnownWav(ArithmeticModel):
    # use lambda_0 and v_offset, for lines with known wavelength
    # by default, the line is restrict to pm 100 km/s

    def __init__(self, name='VoigtLine', lam_0=5000, v_max=1000, b=1.5, d=0.0005, N=999, f=999, N_mag=10, tau_0=0.1):

        self.N_mag = N_mag
        self.v_offset = Parameter(name, 'v_offset', 0.0, frozen=False, min=-v_max, max=v_max)
        self.lam_0 = Parameter(name, 'lam_0', lam_0, frozen=True)
        self.b = Parameter(name, 'b', b, frozen=False, min=1e-12)
        self.d = Parameter(name, 'd', d, frozen=False, min=0)
        self.N = Parameter(name, 'N', N, frozen=True, hidden=True, min=0.0)
        self.f = Parameter(name, 'f', f, frozen=True, hidden=True, min=0.0)
        self.tau_0 = Parameter(name, 'tau_0', tau_0, frozen=False, min=0.0)
        self.kernel_offset = Parameter(name, 'kernel_offset', 0, frozen=True, hidden=True)

        ArithmeticModel.__init__(self, name,
                                 (self.v_offset, self.lam_0, self.b, self.d, self.N, self.f, self.tau_0, self.kernel_offset))

        if N != 999 and f != 999: self.__useNf__(N=N, f=f)


    def calc(self, pars, x, *args, **kwargs):
        v_offset, lam_0, b, d, N, f, tau_0, kernel_offset = pars
        lam_0 = (1 + (v_offset - kernel_offset) / cst.c.to('km/s').value) * lam_0
        if N != 999:
            N = N * (10**self.N_mag)
            transmission = eMath.voigtOpticalDepthAbsorption(lam=x, lam_0=lam_0, b=b, d=d, N=N, f=f)
        else:
            transmission = eMath.voigtOpticalDepthAbsorption(lam=x, lam_0=lam_0, b=b, d=d, tau_0=tau_0)

        return transmission

    def FWHM(self):
        lam_0 = self.lam_0.val * (1 + self.v_offset.val / cst.c.to('km/s').value)
        Gau_FWHM = 2.355 * self.b.val * lam_0 / cst.c.to('km/s').value
        Lor_FWHM = self.d.val * 2
        Voigt_FWHM_tmp = 0.5346 * Lor_FWHM + np.sqrt(0.2166 * Lor_FWHM ** 2 + Gau_FWHM ** 2)
        n_point = round(Voigt_FWHM_tmp * 5 / 0.01)
        x = np.arange(n_point * 2 + 1)
        x = (x - np.median(x)) * 0.01 + self.lam_0.val
        y = self(x)
        CD = abs(1 - np.min(y))
        y = y[abs(1-y) >= 0.5*CD]
        Voigt_FWHM = (len(y) -1 ) * 0.01
        return Voigt_FWHM

    def report_EW(self, *kwords, nsigma = 5, dx = 0.01):
        Voigt_FWHM = self.FWHM()
        n_point = round(Voigt_FWHM * nsigma / dx)
        x = np.arange(n_point * 2 + 1)
        x = (x - np.median(x)) * dx + self.lam_0.val
        y = self(x)
        EW = eMath.integrateEW(x, y)
        if "mA" in kwords: EW = EW * 1000
        return EW

    def __useNf__(self, N=999, f=999):
        self.N.val = N
        self.N.thaw()
        self.N.hidden = False
        self.f.val = f
        self.f.hidden = False
        self.tau_0.freeze()
        self.tau_0.hidden = True

class GaussianLine(ArithmeticModel):

    def __init__(self, name="GaussianLine", lam_0=5000, b=1.5, N=999, f=999, N_mag=10, CD=0.1):
        self.N_mag = N_mag
        self.lam_0 = Parameter(name, 'lam_0', lam_0, frozen=False, min=0.0)
        self.b = Parameter(name, 'b', b, frozen=False, min=1e-12)
        self.N = Parameter(name, 'N', N, frozen=True, hidden=True, min=0.0)
        self.f = Parameter(name, 'f', f, frozen=True, hidden=True, min=0.0)
        self.CD = Parameter(name, 'CD', CD, frozen=False, min=0.0)
        self.kernel_offset = Parameter(name, 'kernel_offset', 0, frozen=True, hidden=True)

        ArithmeticModel.__init__(self, name,
                                 (self.lam_0, self.b, self.N, self.f, self.CD, self.kernel_offset))

        if N != 999  and f != 999: self.__useNf__(N=N, f=f)


    def calc(self, pars, x, *args, **kwargs):
        lam_0, b, N, f, CD, kernel_offset = pars
        if N !=999 and f != 999:
            N = N * (10**self.N_mag)
            transmission = eMath.gaussianAbsorption(lam=x, lam_0=lam_0 - kernel_offset, b=b, N=N, f=f)
        else:
            transmission = eMath.gaussianAbsorption(lam=x, lam_0=lam_0 - kernel_offset, b=b, CD=CD)
        return transmission

    def FWHM(self):
        Gau_FWHM = 2.355 * self.b.val * self.lam_0.val / cst.c.to('km/s').value
        return Gau_FWHM

    def report_EW(self, *kwords, nsigma = 5, dx = 0.01):
        sigma = self.b.val * self.lam_0.val / cst.c.to('km/s').value
        EW = self.CD.val * sigma * np.sqrt(2*np.pi)
        if "mA" in kwords: EW = EW * 1000
        return EW

    def __useNf__(self, N=999, f=999):
        self.N.val = N
        self.N.thaw()
        self.N.hidden = False
        self.f.val = f
        self.f.hidden = False
        self.CD.freeze()
        self.CD.hidden = True

class GaussianLine_KnownWav(ArithmeticModel):

    def __init__(self, name='VoigtLine', lam_0=5000, v_max = 1000, b=1.5, N=999, f=999, N_mag=10, CD=0.1):
        self.N_mag = N_mag
        self.v_offset = Parameter(name, 'v_offset', 0., frozen=False, min=-v_max, max=v_max)
        self.lam_0 = Parameter(name, 'lam_0', lam_0, frozen=True)
        self.b = Parameter(name, 'b', b, frozen=False, min=1e-12)
        self.N = Parameter(name, 'N', N, frozen=True, hidden=True, min=0.0)
        self.f = Parameter(name, 'f', f, frozen=True, hidden=True, min=0.0)
        self.CD = Parameter(name, 'CD', CD, frozen=False, min=0.0)
        self.kernel_offset = Parameter(name, 'kernel_offset', 0, frozen=True, hidden=True)

        ArithmeticModel.__init__(self, name,
                                 (self.v_offset, self.lam_0, self.b, self.N, self.f, self.CD, self.kernel_offset))

        if N != 999 and f != 999: self.__useNf__(N=N, f=f)

    def calc(self, pars, x, *args, **kwargs):
        v_offset, lam_0, b, N, f, CD, kernel_offset = pars
        lam_0 = (1 + (v_offset - kernel_offset) / cst.c.to('km/s').value) * lam_0
        if N !=999 and f != 999:
            N = N * (10**self.N_mag)
            transmission = eMath.gaussianAbsorption(lam=x, lam_0=lam_0, b=b, N=N, f=f)
        else:
            transmission = eMath.gaussianAbsorption(lam=x, lam_0=lam_0, b=b, CD=CD)
        return transmission

        return transmission

    def FWHM(self):
        Gau_FWHM = 2.355 * self.b.val * self.lam_0.val / cst.c.to('km/s').value
        return Gau_FWHM

    def report_EW(self, *kwords, nsigma=5, dx=0.01):
        sigma = self.b.val * self.lam_0.val / cst.c.to('km/s').value
        EW = self.CD.val * sigma * np.sqrt(2 * np.pi)
        if "mA" in kwords: EW = EW * 1000
        return EW

    def __useNf__(self, N=999, f=999):
        self.N.val = N
        self.N.thaw()
        self.N.hidden = False
        self.f.val = f
        self.f.hidden = False
        self.CD.freeze()
        self.CD.hidden = True

class SplineContinuum(ArithmeticModel):
    def __init__(self, name="", n_point=3, x_anchor=None):
        if not name: name = "sp_cont"
        self.name = name
        self.n_point = None
        self.x_anchor = None
        self.x_used = None
        self.y_used = None
        self.__parseNpoint__(n_point=n_point, x_anchor=x_anchor)

        y_str = []
        for i in np.arange(200):
            exec("self.y_{i} = Parameter(name, 'y_{i}', 1.0, frozen=True, hidden=True, min=0.0)".format(i=i))
            y_str.append("self.y_{i}".format(i=i))
        exec("ArithmeticModel.__init__(self, name, (" + ", ".join(y_str) + "))")

        self.__activateY__()

    def __parseNpoint__(self, n_point=3, x_anchor=None):
        try:
            self.n_point = len(x_anchor)
            self.x_anchor = x_anchor
        except:
            if n_point is None: self.n_point = 3
            else: self.n_point = n_point
            self.x_anchor = None

    def __activateY__(self):
        for i in np.arange(self.n_point):
            exec("self.y_{i}.frozen = False".format(i=i))
            exec("self.y_{i}.hidden = False".format(i=i))

    def set_xpoint(self, n_point=3, x_anchor=None):
        self.__parseNpoint__(n_point=n_point, x_anchor=x_anchor)
        self.__activateY__()

    def guess(self, x, y):
        assert len(x) == len(y), "Input X and Y must be of the same length!"
        idx = x.argsort()
        x, y = x[idx], y[idx]

        if self.x_anchor is None:
            n_piece = self.n_point - 1
            y_sections = np.array_split(y, n_piece * 2)
            self.y_0.val = np.median(y_sections[0])
            for i in range(1, len(y_sections), 2):
                y_idx = int((i+1)/2)
                if i == range(len(y_sections))[-1]:
                    span = y_sections[i]
                    exec("self.y_{i}.val = np.median(span)".format(i=y_idx))
                    exec("self.y_{i}.min = 0.99*np.median(span)".format(i=y_idx))
                    exec("self.y_{i}.max = 1.01*np.median(span)".format(i=y_idx))
                else:
                    span = np.append(y_sections[i], y_sections[i + 1])
                    exec("self.y_{i}.val = np.median(span)".format(i=y_idx))
                    exec("self.y_{i}.min = 0.99*np.median(span)".format(i=y_idx))
                    exec("self.y_{i}.max = 1.01*np.median(span)".format(i=y_idx))
        else:
            for i, anchor in enumerate(self.x_anchor):
                left = y[x <= anchor][-1]
                right = y[x >= anchor][0]
                exec("self.y_{i}.val = np.mean([left, right])".format(i=i))
                exec("self.y_{i}.min = 0.99*np.mean([left, right])".format(i=i))
                exec("self.y_{i}.max = 1.01*np.mean([left, right])".format(i=i))

    def calc(self, pars, x, *args, **kwargs):
        y_points = pars
        y_points = y_points[0:self.n_point]
        #print(y_points)
        if self.x_anchor is not None:
            x_points = self.x_anchor
        else:
            n_piece = self.n_point - 1
            x_sections = np.array_split(x, n_piece * 2)
            x_points = [np.min(x)]
            for i in range(1, len(x_sections), 2):
                if i == range(len(x_sections))[-1]:
                    x_points.append(np.max(x))
                else:
                    x_points.append(np.max(x_sections[i]))

        self.x_used = x_points
        self.y_used = y_points
        spline = CubicSpline(x_points, y_points)
        return spline(x)

class Cloud(ArithmeticModel):

    def __init__(self, name="cloud", velocity = 0.0):
        self.name = name
        self.lines = []
        self.velocity = velocity
        self.instrumental = None
        self.continuum = models.Const1D()
        self.continuum.c0.val = 1.0

        self.link_b = False
        self.link_d = False
        self.freeze_d = False
        #self.model = None
        #self.compiled = False

    def set_velocity(self, velocity):
        self.velocity = velocity
        if len(self.lines) > 0:
            self.lines[0].v_offset.val = velocity

    def addLines(self, lam_0, *kwords, line_name=None, b=1.5, d = 0.0005, strength=0.1, N=999, f=999, N_mag=10):
        lam_0 = DataHandling.parseInput(1,lam_0,checklen=False)
        line_name, b, d  = DataHandling.parseInput(len(lam_0), line_name, b, d)
        strength, N, f, N_mag = DataHandling.parseInput(len(lam_0), strength, N, f, N_mag)
        for i, name in enumerate(line_name):
            if name is None:
                name = self.name + "_known_wav_line"+str(len(self.lines))

            # Voigt is the default choice
            if "gaussian" in kwords:
                new_line = GaussianLine_KnownWav(name=name, lam_0=lam_0[i], b=b[i],
                                                 N=N[i], f=f[i], CD=strength[i], N_mag=N_mag[i])
            else:
                new_line = VoigtLine_KnownWav(name=name, lam_0=lam_0[i], b=b[i], d=d[i],
                                              N=N[i], f=f[i], tau_0=strength[i], N_mag=N_mag[i])

            if len(self.lines)>0:
                UI.link(new_line.v_offset, self.lines[0].v_offset)
            else:
                new_line.v_offset = self.velocity

            self.lines.append(new_line)

    def importInstrumental(self, kernel):
        x_g = np.ones_like(kernel)
        kernel = Data1D("kernel", x_g, kernel)
        self.instrumental = ConvolutionKernel(kernel, name="Conv")

    def importContinuum(self, continuum):
        self.continuum = continuum

    def compileModel(self, link_b=None, link_d=None, freeze_d=None, add_instrumental = True, sightline=False, conv_correction=None):
        if link_b is None: link_b=self.link_b
        if link_d is None: link_d=self.link_d
        if freeze_d is None: freeze_d=self.freeze_d

        #first_line = copy.deepcopy(self.lines[0])
        first_line = self.lines[0]
        first_line.v_offset.val = self.velocity
        if hasattr(first_line, "d") and freeze_d: first_line.d.frozen = True

        if conv_correction is not None and (sightline or (self.instrumental is not None)):
            first_line.kernel_offset.val = conv_correction
        model_out = first_line

        for i, line in enumerate(self.lines[1:]):
            if link_b: UI.link(line.b, first_line.b)
            else: line.b.link = None

            if hasattr(line, "d"):
                if freeze_d: line.d.frozen = True
                else: line.d.frozen = False

                if link_d: UI.link(line.d, first_line.d)
                else: line.d.link = None

            if conv_correction is not None and (sightline or (self.instrumental is not None)):
                line.kernel_offset = conv_correction

            UI.link(line.v_offset, first_line.v_offset)
            model_out = model_out * line

        # continuum and instrumental
        if not sightline: model_out = self.continuum * model_out
        if add_instrumental and self.instrumental is not None:
            model_out = self.instrumental(model_out)

        return model_out

    def report_EW(self, *kwords, nsigma = 5, dx=0.01, lines = None):
        if lines is None:
            lines = np.arange(len(self.lines))
        else:
            lines = DataHandling.parseInput(1, lines, checklen=False)

        print("Cloud velocity = {v:.2f} km/s".format(v = self.velocity))
        EW_all = []

        unit = "A"
        if "mA" in kwords: unit = "mA"

        for line_idx in lines:
            line = self.lines[line_idx]
            EW = line.report_EW(*kwords, nsigma=nsigma, dx=dx)
            lam_0 = line.lam_0.val
            str = "Line {i} at {lam_0:.2f}, EW = {EW:.2f} ".format(i=line_idx, lam_0=lam_0, EW=EW) + unit
            print(str)
            EW_all.append(EW)

        EW_all = np.array(EW_all)
        return EW_all

    def __afterFit__(self, link_b=None, link_d=None, freeze_d=None):
        self.velocity = self.lines[0].v_offset.val
        if link_b is not None: self.link_b=link_b
        if link_d is not None: self.link_d=link_d
        if freeze_d is not None: self.freeze_d=freeze_d

    def raiseParameter(self,line=None,par=None):
        name,val = [], []
        for item in self.lines:
            if line is None or line.lower() in item.name.lower():
                print(item.name)
                for p in item.pars:
                    if par is None or par.lower() in p.name:
                        print(p.name + ": {val:.2f}".format(val=p.val))
                        name.append(p.fullname)
                        val.append(p.val)
        return name, val

    def __str__(self):
        str_out = "="*15 + " Cloud: " + self.name + " " + "="*15 + "\n"
        str_out += "Velocity: {v:.2f} km/s\n".format(v = self.velocity)
        str_out += "{n} lines in this cloud\n".format(n = len(self.lines))
        for i,line in enumerate(self.lines):
            str_out += "Line {idx}: {name} at {wav}\n".format(idx=i, name=line.name, wav=line.lam_0.val)

        return str_out

class Sightline(ArithmeticModel):

    def __init__(self, name="Model"):
        self.name = name
        self.clouds = []
        self.cloud_names = []
        self.clouds_velocities = []
        self.lines = []
        self.instrumental = None
        self.continuum = models.Const1D()
        self.continuum.c0.val = 1.0

    def importCloud(self, new_cloud):
        self.clouds.append(new_cloud)
        self.clouds_velocities.append(new_cloud.velocity)
        if new_cloud.name.lower() == "cloud":
            self.cloud_names.append("Cloud_{i}".format(i=len(self.clouds)))
        else:
            self.cloud_names.append(new_cloud.name)

    def addLine_tolines(self, lam_0, *kwords, line_name=None, b=1.5, d=0.0005, N=999, f=999, N_mag=10, strength=0.01):
        # create a Voigtline instance and append to self.lines, i.e. lines defined by wavelength rather than velocity
        lam_0 = DataHandling.parseInput(1, lam_0, checklen=False)
        line_name, b, d = DataHandling.parseInput(len(lam_0), line_name, b, d)
        N, f, strength, N_mag = DataHandling.parseInput(len(lam_0), N, f, strength, N_mag)
        for i, name in enumerate(line_name):
            if name is None:
                name = "Line_{idx}".format(idx = len(self.lines))

            # Voigt is the default choice
            if "gaussian" in kwords:
                new_line = GaussianLine_KnownWav(name=name, lam_0=lam_0[i], b=b[i],
                                                 N=N[i], f=f[i], CD=strength[i], N_mag=N_mag[i])
            else:
                new_line = VoigtLine_KnownWav(name=name, lam_0=lam_0[i], b=b[i], d=d[i],
                                              N=N[i], f=f[i], tau_0=strength[i], N_mag=N_mag[i])
            self.lines.append(new_line)

    def addLine_toclouds(self, lam_0, *kwords, clouds=None, cloud_names=None, velocity=None,
                         line_name=None, b=1.5, d=0.0005, strength=0.1, N=999, f=999, N_mag=10):
        assert clouds is not None or velocity is not None or cloud_names is not None,\
            "Please specific which cloud"
        # velocity override cloud_name override cloud(idx)
        if clouds is not None:
            clouds = DataHandling.parseInput(1,clouds,checklen=False)
            clouds_existing = []
            for cloud in clouds:
                if cloud < len(self.clouds):
                    clouds_existing.append(cloud)
                else:
                    print("Cloud{idx} not found".format(idx=cloud))

        if cloud_names is not None:
            cloud_names = DataHandling.parseInput(1, cloud_names, checklen=False)
            clouds_existing = []
            for cloud_name in cloud_names:
                if cloud_name in self.cloud_names:
                    clouds_existing.append(self.cloud_names.index(cloud_name))
                else:
                    print("Cloud " + cloud_name + " not found")

        if velocity is not None:
            velocity = DataHandling.parseInput(1, velocity, checklen=False)
            clouds_existing, clouds_new = [], []
            for v in velocity:
                if v in self.clouds_velocities:
                    clouds_existing.append(self.clouds_velocities.index(v))
                else:
                    clouds_new.append(v)

        if len(clouds_existing) > 0:
            for cloud in clouds_existing:
                self.__addLine_existingcloud__(lam_0, *kwords, cloud=cloud, line_name=line_name,
                                               b=b, d=d, strength=strength, N=N, f=f, N_mag=N_mag)

        if len(clouds_new) > 0:
            for v in clouds_new:
                self.__addLine_newcloud__(lam_0, *kwords, velocity=v, line_name=line_name,
                                          b=b, d=d, strength=strength, N=N, f=f, N_mag=N_mag)
                print("New cloud at {v:.2f} km/s created".format(v=v))

    def __addLine_existingcloud__(self, *kwords, lam_0, cloud=None, line_name=None, b=1.5, d=0.0005,
                                  strength=0.1, N=999, f=999, N_mag=10):
        self.clouds[cloud].addLines(lam_0, *kwords, line_name=line_name, b=b, d=d,
                                    strength=strength, N=N, f=f, N_mag=N_mag)

    def __addLine_newcloud__(self, lam_0, *kwords, velocity=None, line_name=None, b=1.5, d=0.0005,
                             strength=0.1, N=999, f=999, N_mag=10):
        new_cloud_name = "Cloud{i}".format(i = len(self.clouds))
        new_cloud = Cloud(name=new_cloud_name, velocity=velocity)
        new_cloud.addLines(lam_0,*kwords, line_name=line_name, b=b, d=d,
                           strength=strength, N=N, f=f, N_mag=N_mag)
        self.importCloud(new_cloud)

    def importInstrumental(self, kernel):
        x_g = np.ones_like(kernel)
        kernel = Data1D("kernel", x_g, kernel)
        self.instrumental = ConvolutionKernel(kernel, name="conv")

    def importContinuum(self, continuum):
        self.continuum = continuum

    def compileModel(self, link_b=None, link_d=None, freeze_d=None,  add_instrumental = True, conv_correction=None):

        if conv_correction is not None and self.instrumental is None:
            conv_correction = None

        model_out = 1
        for line in self.lines:
            #if hasattr(line_to_add, "d"):
            #   if freeze_d: line_to_add.d.frozen = True
            #   else: line_to_add.d.frozen = False
            if conv_correction is not None:
                line.kernel_offset = conv_correction / cst.c.to('km/s').value * line.lam_0.val
            model_out *= line

        for cloud in self.clouds:
            model_out *= cloud.compileModel(link_b=link_b, link_d=link_d, freeze_d=freeze_d, sightline=True,
                                            add_instrumental=False, conv_correction=conv_correction)

        if add_instrumental and self.instrumental is not None:
            model_out = self.instrumental(model_out)

        model_out = self.continuum * model_out

        return model_out

    def report_EW(self, *kwords, clouds=None, lines=True, nsigma=5, dx=0.01):
        if clouds is None:
            clouds = np.arange(len(self.clouds))
        else:
            clouds = DataHandling.parseInput(1, clouds, checklen=False)

        unit = "A"
        if "mA" in kwords: unit="mA"
        EW_all = []

        if len(clouds) > 0:
            for cloud_idx in clouds:
                print("="*10 + " Cloud {i} ".format(i=cloud_idx) + "="*10)
                cloud = self.clouds[cloud_idx]
                EW = cloud.report_EW(*kwords, nsigma=nsigma, dx=dx)
                EW_all.append(EW)

        EW_lines = []
        if lines:
            print("="*10 + " Lines " + "="*10)
            for line_idx, line in enumerate(self.lines):
                lam_0 = line.lam_0.val
                EW = line.report_EW(*kwords, nsigma = 5, dx = 0.01)
                str = "Line {i} at {lam_0:.2f}, EW = {EW:.2f} ".format(i=line_idx, lam_0=lam_0, EW=EW) + unit
                print(str)
                EW_lines.append(EW)
            EW_lines = np.array(EW_lines)
            EW_all.append(EW_lines)

        return EW_all

    def __afterFit__(self, link_b=None, link_d=None, freeze_d=None):
        for item in self.clouds:
            item.__afterFit__(link_b=link_b, link_d=link_d, freeze_d=freeze_d)

    def raiseParameter(self,line=None,par=None):
        name,val = [], []

        print("="*10 + " Lines " + "="*10)
        for item in self.lines:
            if line is None or line.lower() in item.name.lower():
                print(item.name)
                for p in item.pars:
                    if par is None or par.lower() in p.name:
                        print(p.name + ": {val:.2f}".format(val=p.val))
                        name.append(p.fullname)
                        val.append(p.val)

        print("="*10 + " Clouds " + "="*10)
        for item in self.clouds:
            print("="*5 + " " + item.name + " " + "="*5)
            a,b = item.raiseParameter(line=line, par=par)
            name.append(a)
            val.append(b)

        return name, val

    def __str__(self):
        str_out = "="*15 + " Sightline: " + self.name + " " + "="*15 + "\n"
        str_out += "{n_line} lines and {n_cloud} clouds\n\n".format(n_line=len(self.lines), n_cloud=len(self.clouds))
        str_out += "\n"

        str_out += "=" * 15 + " Lines " + "="*15 + "\n"
        for i, line in enumerate(self.lines):
            str_out += "Line {idx}: {name} at {wav}\n".format(idx=i, name=line.name, wav=line.lam_0.val)

        str_out += "\n"

        for i, cloud in enumerate(self.clouds):
            str_out += cloud.__str__()

        return str_out

