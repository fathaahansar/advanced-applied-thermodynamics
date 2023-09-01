import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo
import scipy.integrate as spi
from sympy import var,solve

# Question 1: Reporting compound and inputs for EOS

Fluid = "Hydrogen Sulfide"

# Source: NIST Website
T_c = 373.1 # Critical Temp [K]
p_c = 9.0000 * (10**6) # Critical Pressure [Pa]
w = 0.100 # Acentric factor
R = 8.3144621 # Gas constant [SI Units]
M = 34.08088*(10**-3) # Molecular weight [mol/kg]
rho_c = 10.844*10**3 # Density [mol/m^3]

# Redlich-Kwong a,b values
a = 0.42748 * ((R**2)*(T_c**2.5))/p_c
b = 0.08664 * (R*T_c)/p_c
print("Redlich-Kwong Parameters:\na = ",a,"\nb = ",b,"\n")

# Redlich-Kwong Equation
def pressure(T,v):
    p = (R*T)/(v-b)-a/((T**0.5)*v*(v+b))
    return p

# Molar Volume Solver
def molar_volumes(p,T):
    v_r = var('v_r')
    solution = solve((R*T)/(v_r-b)-a/((T**0.5)*v_r*(v_r+b))-p)
    solution = [abs(n) for n in solution]
    return solution

# Question 2: Calculating Critical Molar Volume

v_c = molar_volumes(p_c,T_c)[0]
print("Critical Molar Volume: ",v_c * 1000,"[L/mol]")

v_c_NIST = 0.092217 * 0.001 # Critical Molar Volume [Units: m3/mol]
print("Critical Molar Volume (NIST): ",v_c_NIST * 1000,"[L/mol]")

v_c_error = (v_c-v_c_NIST)/v_c
print("Error: ",v_c_error,"%\n")

volume_set = np.linspace(0.00004,0.0005,191)
print("Molar Volume Array: ",volume_set,"\n")

# Question 3: Plotting 3 isotherms for R-K EOS

# Calculating the pressure for the molar volume array
def pressure_set(T, v_set):
    pressure = R * T / (v_set - b) - a / ((T ** 0.5) * v_set * (v_set + b))
    pressure = [abs(n) for n in pressure]
    return pressure

Temperature_set = [T_c*0.95,T_c,T_c*1.05]
Pressure_set = [pressure_set(Temperature_set[0],volume_set),pressure_set(Temperature_set[1],volume_set),pressure_set(Temperature_set[2],volume_set)]

def F_integral(T,p_trial,v_L,v_V):
    F_p,F_p_error = spi.quad(lambda v: ((R*T)/(v-b)-a/((T**0.5)*v*(v+b))-p_trial),v_L,v_V)
    return F_p

def tie_line_parameters(T,p_trial,dp,error):
    v_roots = molar_volumes(p_trial,T)
    F_p = F_integral(T, p_trial, min(v_roots), max(v_roots))

    while F_p > error:
        v_roots = molar_volumes(p_trial, T)
        F_p = F_integral(T,p_trial,min(v_roots),max(v_roots))
        F_p_dp = F_integral(T,p_trial+dp,min(v_roots),max(v_roots))
        F_dash = (F_p_dp - F_p)/dp
        p_trial -= F_p/F_dash

    return [p_trial,v_roots]

p_star,v_star = tie_line_parameters(Temperature_set[0], 3.5*1000000, 10, 0.1)

plt.figure(1)
plt.clf()

# Plotting the 3 isotherms for the values is Temperature_set
plt.plot([v/v_c for v in volume_set],[p/p_c for p in Pressure_set[0]],color='blue',label="{:0.2f} K (Sub-critical)".format(Temperature_set[0]))
plt.plot([v/v_c for v in volume_set],[p/p_c for p in Pressure_set[1]],color='green',label="{:0.2f} K (Critical)".format(Temperature_set[1]))
plt.plot([v/v_c for v in volume_set],[p/p_c for p in Pressure_set[2]],color='red',label="{:0.2f} K (Super-critical)".format(Temperature_set[2]))

# Plotting critical points
plt.plot(1,1,'o',color='black',label="Calculated Critical Point") # As in reduced pressure and volume p = p/p_c and v = v/v_c | p = v = 1
plt.plot(v_c_NIST/v_c,1,'s',color='black',label="Critical Point from NIST")

# Plotting maxwell tie line
plt.plot([v/v_c for v in v_star], [p_star/p_c]*3, '--', color='blue', label="Maxwell Tie-Line")

# Context additions to the plot
plt.title("Pressure-Volume Plot for "+Fluid)
plt.xlabel(r"$v_r$"+" (Reduced Volume)")
plt.ylabel(r"$p_r$"+" (Reduced Pressure)")
ax = plt.gca()
ax.set_xlim([0.4, 2.5])
ax.set_ylim([0.4, 2])
plt.legend()
plt.savefig("PVPlot_"+Fluid.replace(" ", "")+".png")

# Question 4: Coexistence Pressure calculation for additional terms

coexistence_temp = np.append((np.flip([T_c-2*(i+1) for i in range(0,5)])),T_c)

coexistence_p_v = [tie_line_parameters(coexistence_temp[i], 3.5*1000000, 10, 0.1) for i in range(0,5)]

# Creating a set of v and p points to plot the coexistence curve
v_star_set = np.append(np.append([min(coexistence_p_v[i][1]) for i in range(0,5)],v_c),[max(coexistence_p_v[i][1]) for i in range(0,5)])
p_star_set = np.append(np.append([coexistence_p_v[i][0] for i in range(0,5)],p_c),[coexistence_p_v[i][0] for i in range(0,5)])

print("Coexistence values:")
print("Volume: ",v_star_set)
print("Pressure: ",p_star_set)

v_star_set_NIST = [0.00004422959080,0.00004437583110,0.00004452410770,0.00004467447620,0.00004482699490,0.00004498172440,0.00004513872780,0.00004529807090,0.00004545982240,0.00004562405400,0.00004579084050,0.00004596026040,0.00004613239550,0.00004630733160,0.00004648515860,0.00004666597070,0.00004684986670,0.00004703695050,0.00004722733120,0.00004742112360,0.00004761844860,0.00004781943350,0.00004802421270,0.00004823292830,0.00004844573010,0.00004866277710,0.00004888423740,0.00004911028960,0.00004934112320,0.00004957693980,0.00004981795430,0.00005006439550,0.00005031650830,0.00005057455420,0.00005083881380,0.00005110958810,0.00005138720090,0.00005167200100,0.00005196436510,0.00005226470070,0.00005257344980,0.00005289109270,0.00005321815320,0.00005355520330,0.00005390287010,0.00005426184300,0.00005463288210,0.00005501682900,0.00005541461860,0.00005582729360,0.00005625602260,0.00005670212090,0.00005716707690,0.00005765258440,0.00005816058270,0.00005869330720,0.00005925335390,0.00005984376280,0.00006046812630,0.00006113073440,0.00006183677010,0.00006259257940,0.00006340605310,0.00006428717920,0.00006524887080,0.00006630824950,0.00006748872690,0.00006882357280,0.00007036246410,0.00007218466030,0.00007442907200,0.00007737726920,0.00008177387820,0.00009230684930,0.00010480168600,0.00012266252400,0.00013372651400,0.00014299376100,0.00015135472300,0.00015917312400,0.00016663984500,0.00017386975400,0.00018093843900,0.00018789894000,0.00019479035300,0.00020164266500,0.00020847964400,0.00021532067000,0.00022218193800,0.00022907728200,0.00023601875200,0.00024301703100,0.00025008174300,0.00025722168300,0.00026444499800,0.00027175932200,0.00027917188900,0.00028668961500,0.00029431917800,0.00030206706700,0.00030993964000,0.00031794315900,0.00032608383000,0.00033436783000,0.00034280133600,0.00035139055000,0.00036014171700,0.00036906114900,0.00037815523600,0.00038743047100,0.00039689346000,0.00040655093800,0.00041640978300,0.00042647703300,0.00043675989300,0.00044726575400,0.00045800220600,0.00046897704600,0.00048019829500,0.00049167421400,0.00050341331000,0.00051542435600,0.00052771640400,0.00054029879700,0.00055318118500,0.00056637354200,0.00057988618100,0.00059372976800,0.00060791534100,0.00062245433000,0.00063735856900,0.00065264032300,0.00066831230200,0.00068438768300,0.00070088013600,0.00071780384100,0.00073517351400,0.00075300443400,0.00077131246700,0.00079011409300,0.00080942643700,0.00082926729700,0.00084965517900,0.00087060932400,0.00089214975000,0.00091429728600,0.00093707360900,0.00096050128600]
p_star_set_NIST = [2110257.6300,2161766.0400,2214178.7300,2267505.1500,2321754.7800,2376937.1400,2433061.8400,2490138.5000,2548176.8300,2607186.5700,2667177.5500,2728159.6400,2790142.7900,2853137.0200,2917152.4200,2982199.1600,3048287.4800,3115427.7300,3183630.3300,3252905.8100,3323264.7800,3394717.9800,3467276.2400,3540950.5300,3615751.9100,3691691.6100,3768780.9800,3847031.5200,3926454.8800,4007062.8900,4088867.5400,4171881.0100,4256115.6900,4341584.1700,4428299.2800,4516274.0700,4605521.8700,4696056.2500,4787891.1200,4881040.6800,4975519.4700,5071342.3900,5168524.7500,5267082.2800,5367031.1700,5468388.0800,5571170.2600,5675395.5200,5781082.3000,5888249.7700,5996917.8600,6107107.3400,6218839.9300,6332138.3900,6447026.6200,6563529.8400,6681674.7100,6801489.5300,6923004.4900,7046251.8900,7171266.5200,7298086.0500,7426751.5400,7557308.0500,7689805.5200,7824299.8500,7960854.4000,8099542.0300,8240448.2100,8383675.5800,8529351.4500,8677640.9500,8828773.4000,8983104.4700,8983104.4700,8828773.4000,8677640.9500,8529351.4500,8383675.5800,8240448.2100,8099542.0300,7960854.4000,7824299.8500,7689805.5200,7557308.0500,7426751.5400,7298086.0500,7171266.5200,7046251.8900,6923004.4900,6801489.5300,6681674.7100,6563529.8400,6447026.6200,6332138.3900,6218839.9300,6107107.3400,5996917.8600,5888249.7700,5781082.3000,5675395.5200,5571170.2600,5468388.0800,5367031.1700,5267082.2800,5168524.7500,5071342.3900,4975519.4700,4881040.6800,4787891.1200,4696056.2500,4605521.8700,4516274.0700,4428299.2800,4341584.1700,4256115.6900,4171881.0100,4088867.5400,4007062.8900,3926454.8800,3847031.5200,3768780.9800,3691691.6100,3615751.9100,3540950.5300,3467276.2400,3394717.9800,3323264.7800,3252905.8100,3183630.3300,3115427.7300,3048287.4800,2982199.1600,2917152.4200,2853137.0200,2790142.7900,2728159.6400,2667177.5500,2607186.5700,2548176.8300,2490138.5000,2433061.8400,2376937.1400,2321754.7800,2267505.1500,2214178.7300,2161766.0400,2110257.6300]

plt.figure(2)
plt.clf()
plt.plot([v/v_c for v in v_star_set_NIST],[p/p_c for p in p_star_set_NIST],'--',color='violet',label='Actual Co-existence Curve (NIST Data)')
plt.plot([v/v_c for v in v_star_set],[p/p_c for p in p_star_set],'o',color='black',label='Calculated Co-existence Curve')

plt.title("Co-existence Curves $(p_r-v_r)$")
plt.xlabel(r"$v_r$"+" (Reduced Volume)")
plt.ylabel(r"$p_r$"+" (Reduced Pressure)")
plt.legend()
ax = plt.gca()
ax.set_xlim([0.5, 2])
ax.set_ylim([0.8, 1.05])
plt.savefig("CoexistenceCurve_"+Fluid.replace(" ", "")+".png")

# Question 5: Plotting coexistence curve in P_r-T_r space

coexistence_temp_NIST = [300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373]

plt.figure(3)
plt.clf()
plt.plot([T/T_c for T in coexistence_temp_NIST],[p/p_c for p in p_star_set_NIST[0:74]],'--',color='indianred',label='Actual Co-existence Curve (NIST Data)')
plt.plot([T/T_c for T in coexistence_temp],[p/p_c for p in p_star_set[0:6]],'o',color='black',label='Calculated Co-existence Curve')
plt.title("Co-existence Curves $(p_r-T_r)$")
plt.xlabel(r"$T_r$"+" (Reduced Temperature)")
plt.ylabel(r"$p_r$"+" (Reduced Pressure)")
plt.legend()
ax = plt.gca()
ax.set_xlim([0.965, 1.005])
ax.set_ylim([0.800, 1.025])
plt.savefig("CoexistenceCurve_PvsT_"+Fluid.replace(" ", "")+".png")

# Question 6: Plotting coexistence curve in ln P_r - 1/T_r space

# Curve fitting using SciPy

# August Equation
def august_equation(x,A,B):
    return A - B * x

p_opt,p_cov = spo.curve_fit(august_equation,[T_c/T for T in coexistence_temp],[np.log(p/p_c) for p in p_star_set[0:6]])

plt.figure(4)
plt.clf()
plt.plot([T_c/T for T in coexistence_temp],[np.log(p/p_c) for p in p_star_set[0:6]],'o',color='black',label='Calculated Co-existence Curve')
plt.plot([T_c/T for T in coexistence_temp],[p_opt[0] - p_opt[1] * T_c/T for T in coexistence_temp],'r-',label='August Eq. Fit [A=%5.5f, B=%5.5f]' % tuple(p_opt))

plt.title("Co-existence Curves $(ln(p_r)-1/T_r)$")
plt.xlabel(r"$1/T_r$"+" (1/Reduced Temperature)")
plt.ylabel(r"$ln(p_r)$"+" (Logarithmic Reduced Pressure)")
plt.legend()
ax = plt.gca()
ax.set_xlim([0.995, 1.04])
ax.set_ylim([-0.2, 0.05])
plt.savefig("CoexistenceCurve_lnPvsinvT_"+Fluid.replace(" ", "")+".png")

# Enthalpy of vaporization
delta_h = p_opt[0] * R

print('Calculated Enthalpy of vaporization: ',delta_h)

# Repeating calculations and plotting for NIST data
p_opt_NIST,p_cov_NIST = spo.curve_fit(august_equation,[T_c/T for T in coexistence_temp_NIST],[np.log(p/p_c) for p in p_star_set_NIST[0:74]])

plt.figure(5)
plt.clf()
plt.plot([T_c/T for T in coexistence_temp_NIST],[np.log(p/p_c) for p in p_star_set_NIST[0:74]],'--',color='mediumturquoise',label='Actual Co-existence Curve (NIST Data)')
plt.plot([T_c/T for T in coexistence_temp_NIST],[p_opt_NIST[0] - p_opt_NIST[1] * T_c/T for T in coexistence_temp_NIST],'r-',label='August Eq. Fit [A=%5.5f, B=%5.5f]' % tuple(p_opt_NIST))

plt.title("Actual Co-existence Curves $(ln(p_r)-1/T_r)$ (NIST Data)")
plt.xlabel(r"$1/T_r$"+" (1/Reduced Temperature)")
plt.ylabel(r"$ln(p_r)$"+" (Logarithmic Reduced Pressure)")
plt.legend()
ax = plt.gca()
ax.set_xlim([0.995, 1.04])
ax.set_ylim([-0.2, 0.05])
plt.savefig("ActualCoexistenceCurve_lnPvsinvT_"+Fluid.replace(" ", "")+".png")

# Enthalpy of vaporization (NIST)
delta_h_NIST = p_opt_NIST[0] * R

print('Actual Enthalpy of vaporization: ',delta_h_NIST)

print('Error: ',(delta_h_NIST-delta_h)/delta_h_NIST)