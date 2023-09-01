import matplotlib.pyplot as plt
import numpy as np
from sympy import *
from shapely.geometry import LineString

# Properties for Propane and Butane [From NIST]
A_propane, B_propane, C_propane = [4.53678, 1149.36, 24.906] # bar, bar*K, K

A_butane, B_butane, C_butane = [4.35576, 1175.581, -2.071] # bar, bar*K, K

T = 273.15 + 40 # Given temp, Units: K

# Antoine equation computes the pure pressure at certain temperature
def pure_pressure(A,B,C,T):
    p_star = 10**(A-B/(T+C)) # Antoine equation
    return p_star

# Pure pressure values
p_star_propane = pure_pressure(A_propane,B_propane,C_propane,T)
p_star_butane = pure_pressure(A_butane,B_butane,C_butane,T)

print("Pure Pressure (Propane): "+str(p_star_propane)+" bar")
print("Pure Pressure (Butane): "+str(p_star_butane)+" bar")

# Problem 1.1 Plotting the bubble and dew curves

# Defining line space for propane mole fraction
x_propane = np.linspace(0,1,201)

# Using Raoult's Law and the fact that x_propane + x_butane = 1, we get bubble curve pressure equation
p_bubble = p_star_butane + (p_star_propane - p_star_butane) * x_propane

# Using Raoult's Law and the fact that y_propane + y_butane = 1, we get dew curve pressure equation (here both x and y are represented on the same x-axis), hence y_propane = x_propane
p_dew = p_star_propane * p_star_butane / (p_star_propane+(p_star_butane-p_star_propane) * x_propane)

# Plotting the two curves
plt.figure(1)
plt.clf()
plt.plot(x_propane,p_bubble,color='blue',label='Bubble Curve')
plt.plot(x_propane,p_dew,color='red',label='Dew Curve')
plt.title("Bubble/Dew Curves at $40 \degree C$")
plt.xlabel(r"$x_3$"+" (Mole Fraction of Propane)")
plt.ylabel(r"$p$"+" (Pressure) [in Bar]")
plt.legend()
plt.savefig("BubbleDewCurve.png")

# Problem 1.2 Determine pressure where x = y

# Equating number of moles of liquid and vapor, and solving for p
p = var('p')
x_p = (p - p_star_butane)/(p_star_propane - p_star_butane)
y_p = x_p * (p_star_propane/p)
p_equal_moles = solve(1 - (1 - 2*y_p)/(2*x_p - 1))

print("Pressure for Equal Liquid and Vapor Mole Fractions:")
print(p_equal_moles) # Positive root is the pressure we seek

# Problem 1.3 Plotting liquid and vapor mole fractions

p_set = np.linspace(5,10,201)

x_p = (p_set - p_star_butane)/(p_star_propane - p_star_butane)
y_p = x_p * (p_star_propane/p_set)
L_by_V = (1 - 2*y_p)/(2*x_p - 1)

liq_mol_frac = 1 / (1 + 1/L_by_V)
vap_mol_frac = 1 - liq_mol_frac

# Plotting the mole fraction data sets
plt.figure(2)
plt.clf()
plt.plot(p_set,liq_mol_frac,color='red',label='Liquid Fraction')
plt.plot(p_set,vap_mol_frac,color='blue',label='Vapor Fraction')
plt.title("Liquid/Vapor Mole Fraction v/s Pressure")
plt.ylabel("Mole Fraction")
plt.xlabel(r"$p$"+" (Pressure) [Bar]")
plt.legend()
ax = plt.gca()
ax.set_ylim([0, 1])
plt.savefig("LiqGasMoleFractionvsPressure_Problem1_3.png")

# Problem 1.4: Plotting mole fraction of propane in liq and vapor as a function of pressure

# Computing pressure limits where vapor and liquid phases coexist
liq_curve = LineString(np.column_stack((p_set,liq_mol_frac)))
vap_curve = LineString(np.column_stack((p_set,vap_mol_frac)))
x_axis = LineString(np.column_stack((p_set,np.zeros(p_set.size))))

pressure_value_low = liq_curve.intersection(x_axis).xy[0][0]
pressure_value_high = vap_curve.intersection(x_axis).xy[0][0]

print('Pressure range for V-L coexistence is '+str(pressure_value_low)+' bar to '+str(pressure_value_high)+' bar')

# Pressure set with values a bit beyond the coexistence regions
p_set_coexistence = np.linspace(pressure_value_low,pressure_value_high,201)
x_p = (p_set_coexistence - p_star_butane)/(p_star_propane - p_star_butane)
y_p = x_p * (p_star_propane/p_set_coexistence)

plt.figure(3)
plt.clf()
plt.plot(p_set_coexistence,x_p,color='red',label='Liquid Mole Fraction')
plt.plot(p_set_coexistence,y_p,color='blue',label='Vapor Mole Fraction')
plt.plot([p_set_coexistence[0],p_set_coexistence[0]],[x_p[0],y_p[0]],'g--',color='grey')
plt.plot([p_set_coexistence[p_set_coexistence.size-1],p_set_coexistence[p_set_coexistence.size-1]],[x_p[p_set_coexistence.size-1],y_p[p_set_coexistence.size-1]],'g--',color='grey')
plt.plot([5.5,pressure_value_low],[0.5,y_p[0]], color='black',label='Single Phase Region')
plt.plot([9,pressure_value_high],[0.5,x_p[p_set_coexistence.size-1]],color='black')
plt.title("Liquid/Vapor Mole Fraction v/s Pressure for Propane")
plt.ylabel("Mole Fraction")
plt.xlabel(r"$p$"+" (Pressure) [Bar]")
plt.legend()
ax = plt.gca()
plt.savefig("LiqGasMoleFractionvsPressurePropane_Problem1_4.png")

# Problem 2: Construct T-x phase diagram

# Defining a function to solve for the T values for a given x value
def bubble_and_dew_temps(x1):
    T = var('T')

    # Defining intermediate variables
    p = 101.3 # kPa
    p1_star = exp(16.1952 - (3423.53/(T - 55.7172)))
    p2_star = exp(14.0568 - (2825.42/(T - 42.7089)))

    x2 = 1 - x1

    Gamma12 = 0.0952
    Gamma21 = 0.2713

    gamma_1 = exp(-log(x1 + x2*Gamma12) + x2*(Gamma12/(x1 + x2*Gamma12) - Gamma21/(x2 + x1*Gamma21)))
    gamma_2 = exp(-log(x2 + x1*Gamma21) + x1*(Gamma21/(x2 + x1*Gamma21) - Gamma12/(x1 + x2*Gamma12)))

    bubble_temp_value = nsolve(p - x1*gamma_1*p1_star - x2*gamma_2*p2_star,(300,500), solver = 'bisect')

    # Redefining to avoid confusion, plot is to be on the same x-axis linespace
    y1 = x1
    y2 = 1 - y1
    dew_temp_value = nsolve(1/p - y1/(gamma_1*p1_star) - y2/(gamma_2*p2_star),(300,500), solver = 'bisect')

    return bubble_temp_value, dew_temp_value

# Computing the T value set for the mole fraction set
x1 = np.linspace(0,1,201)
T_bubble = np.zeros(x1.size)
T_dew = np.zeros(x1.size)

for i in range(x1.size):
    T_bubble[i], T_dew[i] = bubble_and_dew_temps(x1[i])

# Finding the Azeotrope
azeotrope_min = []
azeotrope_avg = []
azeotrope_mole_frac = []

for i in range(x1.size):
    if x1[i] >= 0.3 and x1[i] <= 0.4: # visually, the range we can expect an azeotrope
        azeotrope_min += [abs(T_bubble[i] - T_dew[i])]
        azeotrope_avg += [(T_bubble[i] + T_dew[i])/2]
        azeotrope_mole_frac += [x1[i]]

azeotrope_min = np.asarray(azeotrope_min) # Converting to numpy array
index_min = np.argmin(azeotrope_min) # Finding the index of the minimum value, which we will use for the plot

print('Azeotrope coordinates: ['+str(azeotrope_mole_frac[index_min])+', '+str(azeotrope_avg[index_min])+']')

# Plotting the two curves
plt.figure(4)
plt.clf()
plt.plot(x1,T_bubble,color='blue',label='Bubble Curve')
plt.plot(x1,T_dew,color='red',label='Dew Curve')
plt.plot(azeotrope_mole_frac[index_min],azeotrope_avg[index_min],'ro',color='black', markersize=5, label="Azeotrope")
plt.title("Bubble/Dew Curves at 101.3 kPa")
plt.xlabel(r"$x_1$"+" (Mole Fraction of Ethanol)")
plt.ylabel(r"$T$"+" (Temperature) [K]")
plt.legend()
ax = plt.gca()
ax.set_xlim([0, 1])
plt.savefig("TempvsMoleFrac_Problem2.png")