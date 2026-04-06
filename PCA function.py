import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn import datasets
from sklearn.decomposition import PCA
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
import re
from pybaselines import Baseline
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import mpl_toolkits.mplot3d


files = []

def baseline_aspls(wavenumbers, absorbance,lam):
   w_fit = wavenumbers[::-1]
   a_fit = absorbance[::-1]
   
   fitter=Baseline(x_data=w_fit)
   baseline_curve,param=fitter.aspls(a_fit,lam)
   
   baseline_absor=a_fit - baseline_curve
   smoothed = savgol_filter(baseline_absor, window_length=13, polyorder=3,deriv=0)
   return smoothed[::-1]
   

    
def pca(files):
	data_rows=[]
	samples=[]
	for f in files:
		file1=pd.read_excel(f)
		transmittance=file1.iloc[:,0]
		wavenumbers=file1.iloc[:,1]
		absorbance=2-np.log10(transmittance)
		#corec_atr = atr_correction (wavenumbers, absorbance )
		corrected_absorbance = baseline_aspls(wavenumbers, absorbance,1e7)
		data_rows.append(corrected_absorbance)
		samples.append(f.split('.')[0])
		
	wavenumbers=pd.read_excel(files[0]).iloc[:,1]. values 
	
	df_pca=pd.DataFrame(data_rows, columns=wavenumbers,index=samples)
	
	#select the particular Wavenumbers for pca
	df_pca = df_pca.loc[:, (df_pca.columns >= 650) & (df_pca.columns <= 4000)]

	scaler = StandardScaler()
	df_scaled = scaler.fit_transform(df_pca)
	
	pca=PCA(n_components=3)
	pca_reduced = pca.fit_transform(df_scaled)
	
	#pca_reduced =pca.fit_transform(df_pca)
	
	ex_var=pca.explained_variance_ratio_
	
	loadings_corr = pca.components_.T * np.sqrt(pca.explained_variance_)
	
	loadings_df=pd.DataFrame(loadings_corr, columns=["PC1","PC2","PC3"],index=df_pca.columns[(df_pca.columns >= 650) & (df_pca.columns <= 4000)]
	)
	
	top_indices1 = loadings_df["PC1"].abs().sort_values(ascending=False).head(10).index
	
	print (top_indices1.round(2))
	
	


	
	
	pc_columns = [f"PC{i+1}" for i in range(3)]
	
	df_selected_pcs=pd.DataFrame(pca_reduced,columns=pc_columns,index=samples)
	df_selected_pcs['Location'] = [re.sub(r'\s+\d+\.xlsx$|\.xlsx$', '', s) for s in files]
	#df_selected_pcs['Sample_ID'] = files
	
	
	markers = ['o', 's', 'p', 'D', 'v']

	fig=plt.figure(1,figsize=(12,10))
	ax=fig.add_subplot(111)


	for i, (location, group) in enumerate(df_selected_pcs.groupby('Location')):
	   
	   # Select marker using the index (loops back if you have more groups than markers)
	   
	   marker = markers[i % len(markers)]
	   
	   ax.scatter(group.iloc[:, 0],group.iloc[:, 1],label=location,s=150 ,marker=marker, edgecolors='k',)
	
	ax.set_xlabel(f'PC1 ({ex_var[0]:.2%} variance)')
	ax.set_ylabel(f'PC2 ({ex_var[1]:.2%} variance)')
	#ax.set_zlabel(f'PC3 ({ex_var[2]:.2%} variance)')
	ax.set_title(" PCA OF FULL DATA WITH 0 DERIVATIVE",fontweight="bold", fontsize=18)
	ax.legend(title="Classes")
	plt.grid()
	plt.tight_layout()
	save_path = "/storage/emulated/0/Download/ pca full data 0st deriv.png"
	#plt.savefig(save_path)
	plt.show()
	print(df_selected_pcs)
	
	
		
pca(files)

