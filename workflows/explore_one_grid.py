##there are some plots to explore one specific output csv
#make sure the cell indices written on the file path for cohort and ecosystem match

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.ticker import AutoMinorLocator

# fancy plot stuff
plt.rcParams.update({'figure.facecolor':  '#0d1117','axes.facecolor':    '#0d1117','axes.edgecolor':    '#30363d','axes.labelcolor':   '#c9d1d9','axes.grid':         True,'grid.color':        '#21262d','grid.linewidth':    0.6,'xtick.color':       '#8b949e','ytick.color':       '#8b949e','text.color':        '#c9d1d9','font.family':       'monospace'})

cohort_raw = pd.read_csv("gs://leap-persistent/samkuemmel/Model_data/output_amazonCerrado27/ESSPT_213150_Cohort_yearly.csv.gz",compression="gzip")
ecosystem_raw = pd.read_csv("gs://leap-persistent/samkuemmel/Model_data/output_amazonCerrado27/ESSPT_213150_Ecosystem_yearly.csv.gz",compression="gzip")

# Strip the column spaces dynamically to protect against name alignment changes
cohort = cohort_raw.rename(columns=lambda x: x.strip())
ecosystem = ecosystem_raw.rename(columns=lambda x: x.strip())

df  = cohort[(cohort["cNo."] == 1) & (cohort["yr"].between(1, 80))]
df2 = ecosystem[ecosystem["year"].between(1, 80)]

ts  = df.groupby("yr")["PFT"].sum()
gpp = df2["GPP"].values
yrs = df2["year"].values
lai = df2["LAI"].values

# 5-yr rolling mean gpp
gpp_smooth = pd.Series(gpp).rolling(5, center=True).mean().values

##plotting
fig = plt.figure(figsize=(13, 7), facecolor='#0d1117')
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
# this is making a 2x2 plot and filling it 
ax_alpha = fig.add_subplot(gs[0, 0]) 
ax_beta = fig.add_subplot(gs[0, 1])   
ax_gamma = fig.add_subplot(gs[1, 0])  
ax_delta = fig.add_subplot(gs[1, 1])

# this is the gpp one 
ax_alpha.fill_between(yrs, gpp, alpha=0.15, color='#3fb950')
ax_alpha.plot(yrs, gpp,        color='#3fb950', linewidth=1.2, alpha=0.6, label='GPP')
ax_alpha.plot(yrs, gpp_smooth, color='#58d68d', linewidth=2.2, label='five yr mean')
ax_alpha.set_title('Gross Primary Productivity', color='#e6edf3', fontsize=12, pad=10)
ax_alpha.set_xlabel('Year')
ax_alpha.set_ylabel('GPP  (KgC m⁻² yr⁻¹)')
ax_alpha.xaxis.set_minor_locator(AutoMinorLocator())
ax_alpha.legend(framealpha=0.15, edgecolor='#30363d', fontsize=9)
ax_alpha.spines[['top','right']].set_visible(False)

# annotate peak
peak_idx = np.argmax(gpp)
ax_alpha.annotate(f'peak {gpp[peak_idx]:.2f}',
             xy=(yrs[peak_idx], gpp[peak_idx]),
             xytext=(yrs[peak_idx]+2, gpp[peak_idx]+0.05),
             color='#f0f6fc', fontsize=8,
             arrowprops=dict(arrowstyle='->', color='#8b949e', lw=0.8))

# crown area 
ax_beta.fill_between(ts.index, ts.values, alpha=0.15, color='#58a6ff')
ax_beta.plot(ts.index, ts.values, color='#58a6ff', linewidth=1.8)
ax_beta.set_title('Crown area — cohort one', color='#e6edf3', fontsize=11, pad=10)
ax_beta.set_xlabel('Year')
ax_beta.set_ylabel('Acrown  (m²/m²)')
ax_beta.spines[['top','right']].set_visible(False)

# gpp histogram 
ax_gamma.hist(gpp, bins=20, color='#3fb950', alpha=0.75, edgecolor='#0d1117', linewidth=0.4)
ax_gamma.axvline(np.mean(gpp), color='#f0f6fc', linewidth=1.4, linestyle='--', label=f'mean {np.mean(gpp):.2f}')
ax_gamma.set_title('GPP distribution', color='#e6edf3', fontsize=11, pad=10)
ax_gamma.set_xlabel('GPP  (KgC m⁻² yr⁻¹)')
ax_gamma.set_ylabel('Years')
ax_gamma.legend(framealpha=0.15, edgecolor='#30363d', fontsize=9)
ax_gamma.spines[['top','right']].set_visible(False)

#lai 
ax_delta.plot(yrs, lai, alpha = 0.15, color = '#00ff66')
ax_delta.fill_between(yrs, lai, color='#3fb950', alpha=0.15)
ax_delta.set_title('LAI', color='#e6edf3', fontsize=11, pad=10)
ax_delta.set_xlabel('years')
ax_delta.set_ylabel('LAI')

fig.suptitle('Amazon grid ___ · ESSPT ecosystem diagnostics',
             color='#e6edf3', fontsize=14, y=1.01, fontweight='normal')
#plt.savefig('ecosystem_dashboard.png', dpi=180, bbox_inches='tight', facecolor='#0d1117')
plt.show()