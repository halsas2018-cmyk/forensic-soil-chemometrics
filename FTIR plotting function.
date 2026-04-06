import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.signal import savgol_filter
from pybaselines import Baseline
from scipy.signal import find_peaks
	

def atr_correction(wavenumbers, absorbance):
    
    nu_ref = np.max(wavenumbers)
    print (nu_ref)
    # Apply the correction: A_corr = A_obs * (nu / nu_ref)
    absorbance_corrected = absorbance * (wavenumbers / nu_ref)
    
    return absorbance_corrected


# Baseline Correction implementation using Aspls algorithm 
def baseline_aspls(wavenumbers, absorbance,lam):
   w_fit = wavenumbers[::-1]
   a_fit = absorbance[::-1]
   
   fitter=Baseline(x_data=w_fit)
   baseline_curve,param=fitter.aspls(a_fit,lam)
   
   baseline_absor=a_fit-baseline_curve
   smoothed = savgol_filter(baseline_absor, window_length=13, polyorder=3,deriv=0)
   return smoothed[::-1]

# FTIR spectra plotting function 
def ftir_plot(files):
	plt.figure(figsize=(12,12))
	current_offset=0
	spacing=30
	
	#reading files and assigning values 
	for f in files:
		file1=pd.read_excel(f)
		transmittance=file1.iloc[:,0]
		wavenumbers=file1.iloc[:,1]
		
		#converting transmittance to absorbance 
		absorbance=2-np.log10(transmittance)
		
		#applying the baseline correction function 
		final_absorbance = baseline_aspls(wavenumbers, absorbance,1e7)
		
		#converting to transmittance 
		final_transmittance = 10**(2 - final_absorbance)
		
		# plotting the spectra 
		plt.plot(wavenumbers,final_transmittance+ current_offset,label=f)
		
		#labeling peaks 
		peaks,_= find_peaks(-final_transmittance, prominence=1, distance=10)
		for p in peaks:
			y_cord=final_transmittance[p]
			x_cord=wavenumbers[p]
			label=f"{int(x_cord)}"
			if x_cord < 4000:
				plt.text(x_cord,y_cord+current_offset-2,label,fontweight="bold", rotation=90,va="top",fontsize=12)
				
		current_offset+=spacing
	
	# Adding labels to the spectra	
	plt.ylabel("Transmittance",fontweight="bold",fontsize=12)
	plt.xlabel("Wavenumber (cm-¹)", fontweight="bold",fontsize=12)
	plt.title("FTIR Spectra of Hotoro", fontweight="bold", fontsize=20)
	plt.gca().invert_xaxis()
	plt.legend()
	plt.grid()
	plt.tight_layout()
	
	#saving option 
	save_path = "/storage/emulated/0/Download/hotoro num.png"
	plt.savefig(save_path)
	plt.show()


#enter files name
files=[]
ftir_plot(files)

