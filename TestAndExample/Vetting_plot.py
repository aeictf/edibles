import edibles.src.edibles_spectrum as eS
import edibles.edibles_settings as Setting
import edibles.src.file_selection as fS
import numpy as np
import astropy.constants as cst
import edibles.src.model as eM
import os

xmin = 3824.6
xmax = 3828.3
target_sightline = [61827, 63804, 73882, 147889, 148184, 154368, 161056, 169454, 179406, 203532, 210121]
log = fS.ObsLog()
log.setFilter(sightline=target_sightline,xMax=xmax, xMin=xmin)
log.applyFilter()
log.outputResult(filename="C3_120-000.txt",mode="singlepanel")
ISM_v = eS.ISM_Info()
ISM_template = ISM_v.lookupVelocity("HD169454")

list_file = Setting.resultdir + "C3_120-000.txt"
template = "HD169454/BLUE_437/HD169454_w437_blue_20160808_O10.fits"  #C3
#template = "HD169454/RED_860/HD169454_w860_redu_20160728_O2.fits"  #C2 2-0
#template = "HD169454/RED_860/HD169454_w860_redl_20160728_O13.fits"  #C3 3-0
template = "HD169454/BLUE_346/HD169454_w346_blue_20160714_O31.fits"

with open(list_file) as f:
    while True:
        line = f.readline().replace("\n", "")
        if not line: break
        #print(line)
        sightline = line.split(",")[0]
        spec_name = line.split(",")[1][1:]
        ISM_target = ISM_v.lookupVelocity(sightline)
        date = spec_name.split("/")[2]
        date = date.split("_")[3]

        sp = eS.EdiblesSpectrum(filename=[template, spec_name],panel_name=["HD169454", sightline])
        sp.baryCorrection()
        sp.shiftSpectrum(v_offset=-ISM_template,panels=0)
        sp.shiftSpectrum(v_offset=-ISM_target, panels=1)
        sp.cutSpectrum(xmin=xmin, xmax=xmax)

        savepath = Setting.plotdir + "C3_120-000/"
        savename = "_".join([sightline, date])
        #sp.showSpectrum(x_label=sightline, y_label=["169454 Template", date],
                        #highlight_species=["C3_R", "C3_P", "C3_Q"], highlight_draw=1,
                        #save=True,savepath=savepath, filename=savename)
        sp.showSpectrum(x_label=sightline, y_label=["169454 Template", date],
                        highlight_draw=1,
                        save=True,savepath=savepath, filename=savename)
