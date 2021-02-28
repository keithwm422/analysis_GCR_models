import numpy as np
x_loc=24
y_loc=22
z_loc=20
# now all of the ratios we want need to be included are listed below
# B/C
# He-3/H3-4
# Be-10/Be-9
# deuterium/protons (H-2/H-1)
# H-2/He-4 (pamela did this)
# protons 
# from the machine learning paper
# and we want their fluxes first and take the ratio l8r

### master location index for GALPROP storing the isotope flux
# add more at the end
sec_proton_loc=7
prim_proton_loc=8
deuterium_loc=9
he3_loc=10
he4_loc=11
li6_loc=12
li7_loc=13
be7_loc=14
be9_loc=15
be10_loc=16
boron10_loc=17
boron11_loc=18
carbon12_loc=19
carbon13_loc=20
oxygen16_loc=23
oxygen17_loc=24
oxygen18_loc=25
#by nuclei
#add more nuclei if you like
# keep the ordering in mass
proton_list=[prim_proton_loc,sec_proton_loc,deuterium_loc]
helium_list=[he3_loc,he4_loc]
lithium_list=[li6_loc,li7_loc]
beryllium_list=[be7_loc,be9_loc,be10_loc]
boron_list=[boron10_loc,boron11_loc]
carbon_list=[carbon12_loc,carbon13_loc]
oxygen_list=[oxygen16_loc,oxygen17_loc,oxygen18_loc]

#element_index=[sec_proton_loc,prim_proton_loc,deuterium_loc,he3_loc,he4_loc,li6_loc,li7_loc,be7_loc,be9_loc,be10_loc,
#               boron10_loc,boron11_loc,carbon12_loc,carbon13_loc,oxygen16_loc,oxygen17_loc,oxygen18_loc]

#total list, append if you add more above
element_index=proton_list+helium_list+lithium_list+beryllium_list+boron_list+carbon_list+oxygen_list

# Kinetic energy in GeV/nucleon and charge+num_nucleons need to be integers
def rigidity_calc(kin_energy, num_nucleons, charge):
    mass_P=0.938 # mass of a nucleon in GeV
    return (((kin_energy*num_nucleons)**(2)+2*(kin_energy*(num_nucleons**(2))*mass_P))**(0.5))/charge
    
def undo_log_energy(energy):
    i=0
    new=[]
    while i<len(energy):
        new.append(10**(energy[i]))
        i+=1
    return new

def log_energy(energy):
    i=0
    new=[]
    while i<len(energy):
        new.append(np.log10(energy[i]))
        i+=1
    return new