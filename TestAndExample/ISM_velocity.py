# this script aims to estimate the ISM velocity from sodium doublets at 3302 AA and around 5890AA

import edibles.src.edibles_spectrum as eS
import astropy.constants as cst
import edibles.src.model as eM
import os

filename = "/Users/haoyufan/EDIBLES_Spectrum/edibles/data/SpecList/Na_ISM.csv"
f = open(filename)

## load spectral data
#filename_3302 = "/HD23180/BLUE_346/HD23180_w346_blue_20140923_O12.fits"
#filename_5890 = "/HD23180/RED_564/HD23180_w564_redu_20140923_O4.fits"


## Step 1
line = f.readline().replace("\n","")
line_split = line.split(",")
line_split.append("")
sightline = line_split[0]
Dates = line_split[1:line_split.index("")]
filename_3302 = []
filename_5890 = []
for Date in Dates:
    filename_3302.append(os.path.join(sightline,"BLUE_346","_".join([sightline,"w346","blue",Date,"O12.fits"])))
    #filename_Na_5890 = os.path.join(sightline,"RED_564","_".join([sightline,"w564","redu",Date,"O4.fits"]))
sp = eS.EdiblesSpectrum(filename_3302, panel_name=Dates)
sodium_wavelength = sp.linelist["Na"][0:2]
sp.baryCorrection()
sp.cutSpectrum(xmin=3301.5, xmax=3304)
sp.showSpectrum()

## Step 2
sp.addMask()
sp.fitContinuum(mode="polynomial", n=3, apply_mask=True)
sp.resetMask()


sp.getPosition_visual(wavelength=sodium_wavelength,tau_0=1,panels=0)

model_working = eM.Sightline(name=sightline)
model_working.addLine_toclouds(sodium_wavelength,velocity=12,tau_0=[0.017,0.009])
#model_working.addLine_toclouds(sodium_wavelength,velocity=-13,tau_0=[0.013,0.009])
model_working.addLine_tolines(3302.18, tau_0 = 0.0014)



panel_now = 0
kernel = sp.getKernel(apply_mask=1, panels=panel_now)
model_working.importInstrumental(kernel)
conv_correction = sp.conv_offset * sp.header[panel_now]["CDELT1"] / sodium_wavelength[0] * cst.c.to('km/s').value
model2fit = model_working.compileModel("link_b", conv_correction=conv_correction)
sp.importModel(model2fit)


sp.fitModel(apply_mask=True,panels=panel_now)

a,v = sp.raiseParameter(par="tau_0")
a,v = sp.raiseParameter(par="v_offset")
for item in v:
    v_correct = item + conv_correction
    print(v_correct)


def flatten(T):
    if type(T) != tuple:
        T = (T,)
        return flatten(T)
    else:
        if hasattr(T[0],'parts'):
            if len(T) == 1: return flatten(T[0].parts)
            else: return flatten(T[0].parts) + flatten(T[1:])
        else:
            if len(T) == 1: return (T[0],)
            else: return (T[0],) + flatten(T[1:])



# C2 and C3


filename_C2_7719 = os.path.join(sightline,"RED_860","_".join([sightline,"w860","redl",Date,"O13.fits"]))
filename_C2_8757 = os.path.join(sightline,"RED_860","_".join([sightline,"w860","redu",Date,"O2.fits"]))
filename_C3 = os.path.join(sightline,"BLUE_437","_".join([sightline,"w437","blue",Date,"O10.fits"]))
#os.path.exists(os.path.join(Setting.datadir,filename_C2_7719))
sp2 = eS.EdiblesSpectrum([filename_C2_7719, filename_C2_8757, filename_C3], panel_name=["C2_7719", "C2_8757", "C3"])

sp2.cutSpectrum(xmin=7710, xmax=7745, panels=0)
sp2.cutSpectrum(xmin=8745, xmax=8787, panels=1)
sp2.cutSpectrum(xmin=4049, xmax=4055, panels=2)
sp2.baryCorrection()
sp2.shiftSpectrum(v_offset = -v[0])
sp2.showSpectrum()



