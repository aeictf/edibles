import numpy as np
import copy
import astropy.constants as cst
from sherpa import models
from sherpa.models.model import ArithmeticModel
from sherpa.models.parameter import Parameter
from sherpa.data import Data1D
from sherpa.instrument import Kernel, ConvolutionKernel, ConvolutionModel
import sherpa.ui as UI
import edibles.src.math as eMath
import edibles.src.datahandling as DataHandling
import math

class VoigtLine(ArithmeticModel):

    def __init__(self, name='VoigtLine'):

        self.lam_0 = Parameter(name, 'lam_0', 5000., frozen=False, min=0.0)
        self.b = Parameter(name, 'b', 1.5, frozen=False, min=1e-12)
        self.d = Parameter(name, 'd', 0.0005, frozen=False, min=0)
        self.N = Parameter(name, 'N', 999, frozen=True, hidden=True, min=0.0)
        self.f = Parameter(name, 'f', 999, frozen=True, hidden=True, min=0.0)
        self.tau_0 = Parameter(name, 'tau_0', 0.1, frozen=False, min=0.0)

        ArithmeticModel.__init__(self, name,
                                 (self.lam_0, self.b, self.d,self.N, self.f, self.tau_0))

    def calc(self, pars, x, *args, **kwargs):
        lam_0, b, d, N, f, tau_0 = pars

        if N != 999:
            transmission = eMath.voigtOpticalDepthAbsorption(lam=x, lam_0=lam_0, b=b, d=d, N=N, f=f)
        else:
            transmission = eMath.voigtOpticalDepthAbsorption(lam=x, lam_0=lam_0, b=b, d=d, tau_0=tau_0)

        return transmission

    def FWHM(self):
        Gau_FWHM = 2.355 * self.b.val * self.lam_0.val / cst.c.to('km/s').value
        Lor_FWHM = self.d.val * 2
        Voigt_FWHM = 0.5346 * Lor_FWHM + np.sqrt(0.2166 * Lor_FWHM**2 + Gau_FWHM**2)
        return Voigt_FWHM

    def report_EW(self, *kwords, nsigma = 5, dx = 0.01, continuum=1.0):
        Voigt_FWHM = self.FWHM()
        n_point = round(Voigt_FWHM * nsigma / dx)
        x = np.arange(n_point * 2 + 1)
        x = (x - np.median(x)) * dx + self.lam_0.val
        y = self(x)
        EW = eMath.integrateEW(x, y, continuum=continuum)
        if "mA" in kwords: EW = EW * 1000
        return EW

class VoigtLine_KnownWav(ArithmeticModel):
    # use lambda_0 and v_offset, for lines with known wavelength
    # by default, the line is restrict to pm 100 km/s

    def __init__(self, name='VoigtLine', lam_0=0, v_max = 100):

        self.v_offset = Parameter(name, 'v_offset', 0., frozen=False, min=-v_max, max=v_max)
        self.lam_0 = Parameter(name, 'lam_0', lam_0, frozen=True)
        self.b = Parameter(name, 'b', 1.5, frozen=False, min=1e-12)
        self.d = Parameter(name, 'd', 0.0005, frozen=False, min=0)
        self.N = Parameter(name, 'N', 999, frozen=True, hidden=True, min=0.0)
        self.f = Parameter(name, 'f', 999, frozen=True, hidden=True, min=0.0)
        self.tau_0 = Parameter(name, 'tau_0', 0.1, frozen=False, min=0.0)

        ArithmeticModel.__init__(self, name,
                                 (self.v_offset, self.lam_0, self.b, self.d, self.N, self.f, self.tau_0))

    def calc(self, pars, x, *args, **kwargs):
        v_offset, lam_0, b, d, N, f, tau_0 = pars
        lam_0 = (1 + v_offset / cst.c.to('km/s').value) * lam_0
        if N != 999:
            transmission = eMath.voigtOpticalDepthAbsorption(lam=x, lam_0=lam_0, b=b, d=d, N=N, f=f)
        else:
            transmission = eMath.voigtOpticalDepthAbsorption(lam=x, lam_0=lam_0, b=b, d=d, tau_0=tau_0)

        return transmission

    def FWHM(self):
        lam_0 = self.lam_0.val * (1 + self.v_offset.val / cst.c.to('km/s').value)
        Gau_FWHM = 2.355 * self.b.val * lam_0 / cst.c.to('km/s').value
        Lor_FWHM = self.d.val * 2
        Voigt_FWHM = 0.5346 * Lor_FWHM + np.sqrt(0.2166 * Lor_FWHM**2 + Gau_FWHM**2)
        return Voigt_FWHM

    def report_EW(self, *kwords, nsigma = 5, dx = 0.01, continuum=1.0):
        Voigt_FWHM = self.FWHM()
        n_point = round(Voigt_FWHM * nsigma / dx)
        x = np.arange(n_point * 2 + 1)
        x = (x - np.median(x)) * dx + self.lam_0.val
        y = self(x)
        EW = eMath.integrateEW(x, y, continuum=continuum)
        if "mA" in kwords: EW = EW * 1000
        return EW

class Cloud(ArithmeticModel):

    def __init__(self, name="cloud", velocity = 0.0):
        self.name = name
        self.lines = []
        self.velocity = velocity
        self.instrumental = None
        #self.model = None
        #self.compiled = False

    def set_velocity(self, velocity):
        self.velocity = velocity
        if len(self.lines) > 0:
            self.lines[0].v_offset = velocity

    def addLines(self, lam_0, line_name=None, b=1.5, d = 0.0005, tau_0=0.1):
        lam_0 = DataHandling.parseInput(1,lam_0,checklen=False)
        line_name, b, d, tau_0 = DataHandling.parseInput(len(lam_0), line_name, b, d, tau_0)
        for i, name in enumerate(line_name):
            if name is None:
                name = self.name + "_known_wav_line"+str(len(self.lines))
            new_line = VoigtLine_KnownWav(name=name)
            new_line.lam_0 = lam_0[i]
            new_line.b = b[i]
            new_line.d = d[i]
            new_line.tau_0 = tau_0[i]

            if len(self.lines)>0:
                UI.link(new_line.v_offset, self.lines[0].v_offset)
            else:
                new_line.v_offset = self.velocity

            self.lines.append(new_line)

    def importInstrumental(self, kernel):
        x_g = np.ones_like(kernel)
        kernel = Data1D("kernel", x_g, kernel)
        self.instrumental = ConvolutionKernel(kernel, name="Conv")

    def compileModel(self, *kwords, add_instrumental = True, sightline=False, conv_correction=None):
        link_b = False
        freeze_d = False
        if "link_b" in kwords: link_b = True
        if "freeze_d" in kwords: freeze_d = True

        first_line = copy.deepcopy(self.lines[0])
        if conv_correction is not None and (sightline or self.instrumental is not None):
            first_line.v_offset = self.velocity - conv_correction
        model = first_line

        for i, line in enumerate(self.lines[1:]):
            if link_b: UI.link(line.b, first_line.b)
            else: line.b.link = None

            if freeze_d: line.d.frozen = True
            else: line.d.frozen = False

            UI.link(line.v_offset, first_line.v_offset)
            model = model * line

        if not sightline:
            model *= models.Const1D()

        if add_instrumental and self.instrumental is not None:
            model = self.instrumental(model)

        return model

    def report_EW(self, *kwords, nsigma = 5, dx=0.01, continuum = 1.0, lines = None):
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
            EW = line.report_EW(*kwords, nsigma=nsigma, dx=dx, continuum=continuum)
            lam_0 = line.lam_0.val
            str = "Line {i} at {lam_0:.2f}, EW = {EW:.2f} ".format(i=line_idx, lam_0=lam_0, EW=EW) + unit
            print(str)
            EW_all.append(EW)

        EW_all = np.array(EW_all)
        return EW_all


    def __str__(self):
        str_out = "="*15 + " Cloud: " + self.name + " " + "="*15 + "\n"
        str_out += "Velocity: {v:.2f} km/s\n".format(v = self.velocity)
        str_out += "{n} lines in this cloud\n".format(n = len(self.lines))
        str_out += "="*20 + "\n"
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

    def importCloud(self, new_cloud):
        self.clouds.append(new_cloud)
        self.clouds_velocities.append(new_cloud.velocity)
        if new_cloud.name.lower() == "cloud":
            self.cloud_names.append("Cloud_{i}".format(i=len(self.clouds)))
        else:
            self.cloud_names.append(new_cloud.name)

    def addLine_tolines(self, lam_0, line_name=None, b=1.5, d=0.0005, tau_0=0.01):
        # create a Voigtline instance and append to self.lines, i.e. lines defined by wavelength rather than velocity
        lam_0 = DataHandling.parseInput(1, lam_0, checklen=False)
        line_name, b, d, tau_0 = DataHandling.parseInput(len(lam_0), line_name, b, d, tau_0)
        for i, name in enumerate(line_name):
            if name is None:
                name = "Line_{idx}".format(idx = len(self.lines))
            new_line = VoigtLine(name=name)
            new_line.lam_0 = lam_0[i]
            new_line.b = b[i]
            new_line.d = d[i]
            new_line.tau_0 = tau_0[i]
            self.lines.append(new_line)

    def addLine_toclouds(self, lam_0, clouds=None, cloud_names=None, velocity=None, line_name=None, b=1.5, d=0.0005, tau_0=0.1):
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
                self.__addLine_existingcloud__(lam_0, cloud=cloud, line_name=line_name, b=b, d=d, tau_0=tau_0)

        if len(clouds_new) > 0:
            for v in clouds_new:
                self.__addLine_newcloud__(lam_0, velocity=v, line_name=line_name, b=b, d=d, tau_0=tau_0)
                print("New cloud at {v:.2f} km/s created".format(v=v))

    def __addLine_existingcloud__(self, lam_0, cloud=None, line_name=None, b=1.5, d=0.0005, tau_0=0.1):
        self.clouds[cloud].addLines(lam_0, line_name=line_name, b=b, d=d, tau_0=tau_0)

    def __addLine_newcloud__(self, lam_0, velocity=None, line_name=None, b=1.5, d=0.0005, tau_0=0.1):
        new_cloud_name = "Cloud{i}".format(i = len(self.clouds))
        new_cloud = Cloud(name=new_cloud_name, velocity=velocity)
        new_cloud.addLines(lam_0,line_name=line_name, b=b, d=d, tau_0=tau_0)
        self.importCloud(new_cloud)

    def importInstrumental(self, kernel):
        x_g = np.ones_like(kernel)
        kernel = Data1D("kernel", x_g, kernel)
        self.instrumental = ConvolutionKernel(kernel, name="conv")

    def compileModel(self, *kwords, add_instrumental = True, conv_correction=None):

        if conv_correction is not None and self.instrumental is None:
            conv_correction = None

        CstCont = models.Const1D()
        model = CstCont
        for line in self.lines:
            line_to_add = copy.deepcopy(line)
            if conv_correction is not None:
                line_to_add.lam_0 = (1 - conv_correction / cst.c.to('km/s').value) * copy.deepcopy(line_to_add.lam_0)
            model *= line_to_add

        for cloud in self.clouds:
            model *= cloud.compileModel(*kwords, sightline=True, add_instrumental=False,
                                        conv_correction=conv_correction)

        if add_instrumental and self.instrumental is not None:
            model = self.instrumental(model)

        return model

    def report_EW(self, *kwords, clouds=None, lines=True, nsigma=5, dx=0.01, continuum=1.0):
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
                EW = cloud.report_EW(*kwords, nsigma=nsigma, dx=dx, continuum=continuum)
                EW_all.append(EW)

        EW_lines = []
        if lines:
            print("="*10 + " Lines " + "="*10)
            for line_idx, line in enumerate(self.lines):
                lam_0 = line.lam_0.val
                EW = line.report_EW(*kwords, nsigma = 5, dx = 0.01, continuum=1.0)
                str = "Line {i} at {lam_0:.2f}, EW = {EW:.2f} ".format(i=line_idx, lam_0=lam_0, EW=EW) + unit
                print(str)
                EW_lines.append(EW)
            EW_lines = np.array(EW_lines)
            EW_all.append(EW_lines)

        return EW_all

    def __str__(self):
        str_out = "="*15 + " Cloud: " + self.name + " " + "="*15 + "\n"
        str_out += "{n_line} lines and {n_cloud} clouds\n\n".format(n_line=len(self.lines), n_cloud=len(self.clouds))
        str_out += "\n"

        str_out += "=" * 15 + " Lines " + "="*15 + "\n"
        for i, line in enumerate(self.lines):
            str_out += "Line {idx}: {name} at {wav}\n".format(idx=i, name=line.name, wav=line.lam_0.val)

        str_out += "\n"

        for i, cloud in enumerate(self.clouds):
            str_out += cloud.__str__()

        return str_out
