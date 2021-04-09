from cosmic_ray_nuclei_index import rigidity_calc, undo_log_energy, log_energy
import numpy as np
from get_splines import *
from scipy.optimize import curve_fit

def flux_ratio_versus_x(numerator_x,numerator_y,denominator_x,denominator_y,spline_steps):
    num_x=np.array(log_energy(numerator_x.copy()))
    den_x=np.array(log_energy(denominator_x.copy()))
    x_list=[num_x,den_x]
    num_y=np.array(log_energy(numerator_y.copy()))
    den_y=np.array(log_energy(denominator_y.copy()))
    y_list=[num_y,den_y]
    #common x range
    spline_min_x,spline_max_x=find_interpolation_range(*x_list)
    #numerator
    num_x_rtn,num_y_rtn=spline(num_x,num_y, spline_steps,spline_min_x,spline_max_x)
    num_x_rtn=np.array(undo_log_energy(num_x_rtn))
    num_y_rtn=np.array(undo_log_energy(num_y_rtn))
    #denominator
    den_x_rtn,den_y_rtn=spline(den_x,den_y, spline_steps,spline_min_x,spline_max_x)
    den_x_rtn=np.array(undo_log_energy(den_x_rtn))
    den_y_rtn=np.array(undo_log_energy(den_y_rtn))
    ratio=np.true_divide(num_y_rtn,den_y_rtn)
    return num_x_rtn,ratio

def make_oxygen_nuclei(name,charge,energy_per_nuc,model,solar_phi,spline_steps,fluxes_per_element_full):
    o_obj=Nuclei(name,charge)
    o_obj.add_isotopes('O-16',16,charge)
    o_obj.add_isotopes('O-17',17,charge)
    o_obj.add_isotopes('O-18',18,charge)
    j=0
    #print(len(o_obj.list_isotopes))
    while j< len(o_obj.list_isotopes):
        o_obj.list_isotopes[j].add_energy_per_nucleon(energy_per_nuc)
        #print(cosmic_ray_nuclei_index.oxygen_list[j])
        #print(j)
        o_obj.list_isotopes[j].add_flux(fluxes_per_element_full[model][
            cosmic_ray_nuclei_index.element_index.index(cosmic_ray_nuclei_index.oxygen_list[j])])
        o_obj.list_isotopes[j].add_modulation(solar_phi)
        j+=1
    o_obj.add_isotope_fluxes()
    o_obj.calc_total_flux(spline_steps)
    return o_obj
def make_carbon_nuclei(name,charge,energy_per_nuc,model,solar_phi,spline_steps,fluxes_per_element_full):
    c_obj=Nuclei(name,charge)
    c_obj.add_isotopes('C-12',12,charge)
    c_obj.add_isotopes('C-13',13,charge)
    j=0
    #print(len(c_obj.list_isotopes))
    while j< len(c_obj.list_isotopes):
        c_obj.list_isotopes[j].add_energy_per_nucleon(energy_per_nuc)
        c_obj.list_isotopes[j].add_flux(fluxes_per_element_full[model][
            cosmic_ray_nuclei_index.element_index.index(cosmic_ray_nuclei_index.carbon_list[j])])
        c_obj.list_isotopes[j].add_modulation(solar_phi)
        j+=1
    c_obj.add_isotope_fluxes()
    c_obj.calc_total_flux(spline_steps)
    return c_obj
def make_boron_nuclei(name,charge,energy_per_nuc,model,solar_phi,spline_steps,fluxes_per_element_full):
    n_obj=Nuclei(name,charge)
    n_obj.add_isotopes('B-10',10,charge)
    n_obj.add_isotopes('B-11',11,charge)
    j=0
    print(len(n_obj.list_isotopes))
    while j< len(n_obj.list_isotopes):
        n_obj.list_isotopes[j].add_energy_per_nucleon(energy_per_nuc.copy())
        #print(cosmic_ray_nuclei_index.boron_list[j])
        #print(j)
        n_obj.list_isotopes[j].add_flux(np.array(fluxes_per_element_full[model][
            cosmic_ray_nuclei_index.element_index.index(cosmic_ray_nuclei_index.boron_list[j])].copy()))
        n_obj.list_isotopes[j].add_modulation(solar_phi)
        j+=1
    #print(n_obj.list_isotopes[-1].flux)
    n_obj.add_isotope_fluxes()
    n_obj.calc_total_flux(spline_steps)
    return n_obj

def make_boron_nuclei(name,charge,energy_per_nuc,model,solar_phi,spline_steps,fluxes_per_element_full):
    n_obj=Nuclei(name,charge)
    n_obj.add_isotopes('Be-10',10,charge)
    n_obj.add_isotopes('Be-9',9,charge)
    n_obj.add_isotopes('Be-7',7,charge)
    j=0
    print(len(n_obj.list_isotopes))
    while j< len(n_obj.list_isotopes):
        n_obj.list_isotopes[j].add_energy_per_nucleon(energy_per_nuc.copy())
        #print(cosmic_ray_nuclei_index.boron_list[j])
        #print(j)
        n_obj.list_isotopes[j].add_flux(np.array(fluxes_per_element_full[model][
            cosmic_ray_nuclei_index.element_index.index(cosmic_ray_nuclei_index.be_list[j])].copy()))
        n_obj.list_isotopes[j].add_modulation(solar_phi)
        j+=1
    #print(n_obj.list_isotopes[-1].flux)
    n_obj.add_isotope_fluxes()
    n_obj.calc_total_flux(spline_steps)
    return n_obj

# TRY THE FORCE_FIELD
#try the force-field approx
#companion function, EK is a column array
def force_field_factor(Z,A,phi,EK):
    proton_mass= 0.938 #In GeV
    PHI=Z*phi/A #really this is (Z e phi)/A but e=1 in the units I want
    #print(PHI)
    EK_shifted=EK.copy()
    EK_shifted=EK_shifted-PHI
    EK_sum=EK_shifted.copy()+2*A*proton_mass
    EK_num=EK_shifted*EK_sum
    EK_orig=EK.copy()
    EK_sum_2=EK_orig+2*A*proton_mass
    EK_denom=EK_orig*EK_sum_2
    return np.true_divide(EK_num,EK_denom), EK_shifted
# this function returns the spectra at Earth of the isotope provided (charge, Z, and mass, A) and its shifted energy values
#(not per nucleon)
#given the solar modulation potential phi and the Kinetic energy of the particle
#phi needs to be in units of GV not MV
#EK in GeV (not per nucleon!)
def force_field_approx(LIS,Z,A,phi,EK):
    factor,EK_shifted=force_field_factor(Z,A,phi,EK)
    return LIS*(factor),EK_shifted

def rigidity_power_law(R, A, GAMMA):
    return A * (R)**(GAMMA)

class Ratio:
    def __init__(self,name):
        self.name = name
        self.energy = []
        self.rigidity = []
        self.energy_per_nucleon = []
        self.ratio_energy_per_nucleon = []
        self.ratio_rigidity = []
        self.ratio_energy = []
        self.ratio_energy_per_nucleon_modulated = []
        self.ratio_rigidity_modulated = []
        self.ratio_energy_modulated = []
        self.rigidity_modulated = []
        self.energy_modulated = []
        self.energy_per_nucleon_modulated=[]   
        self.phi = 0
        self.numerator = []
        self.denominator = []
        self.spectral_index = 0
        self.spectral_amplitude = 0
        self.covariance = []
        self.fit_cutoff_rigidity = 0
    def add_nuclei(self,num,den,spline_steps):
        self.numerator=num
        self.denominator=den
        self.rigidity,self.ratio_rigidity=flux_ratio_versus_x(num.rigidity,num.flux_rigidity,den.rigidity,den.flux_rigidity,spline_steps)
        self.energy,self.ratio_energy=flux_ratio_versus_x(num.energy,num.flux_energy,den.energy,den.flux_energy,spline_steps)
        self.energy_per_nucleon,self.ratio_energy_per_nucleon=flux_ratio_versus_x(num.energy_per_nucleon,num.flux_energy_per_nucleon,
                                                              den.energy_per_nucleon,den.flux_energy_per_nucleon,spline_steps)
        self.rigidity_modulated,self.ratio_rigidity_modulated=flux_ratio_versus_x(num.rigidity_modulated,num.flux_rigidity_modulated
                                                              ,den.rigidity_modulated,den.flux_rigidity_modulated,spline_steps)
        self.energy_modulated,self.ratio_energy_modulated=flux_ratio_versus_x(num.energy_modulated,num.flux_energy_modulated
                                                              ,den.energy_modulated,den.flux_energy_modulated,spline_steps)
        self.energy_per_nucleon_modulated,self.ratio_energy_per_nucleon_modulated=flux_ratio_versus_x(
                                                       num.energy_per_nucleon_modulated,num.flux_energy_per_nucleon_modulated,
                                                       den.energy_per_nucleon_modulated,den.flux_energy_per_nucleon_modulated,spline_steps)
        self.phi=self.numerator.phi
    def fit_ratio(self,R_cutoff):
        self.cutoff=R_cutoff
        popt_ratio, pcov_ratio = curve_fit(rigidity_power_law,self.rigidity[self.rigidity>R_cutoff],
                                           self.ratio_rigidity[self.rigidity>R_cutoff],[10,-0.75],bounds=(-2,[1000,2]))
        self.covariance=np.array(pcov_ratio)
        self.spectral_index=popt_ratio[1]
        self.spectral_amplitude=popt_ratio[0]
#no energy per nucleon for a nucleus
class Nuclei:
    def __init__(self, name, charge):
        self.name = name
        self.charge=charge
        self.energy = []
        self.rigidity = []
        self.energy_per_nucleon = []
        self.flux_energy_per_nucleon = []
        self.flux_rigidity = []
        self.flux_energy = []
        self.rigidity_modulated = []
        self.energy_modulated = []
        self.energy_per_nucleon_modulated=[]
        self.flux_energy_per_nucleon_modulated = []
        self.flux_rigidity_modulated = []
        self.flux_energy_modulated = []
        self.phi=0
        # and the isotopes
        self.list_isotopes=[]
        self.spectral_index=0

    def add_isotopes(self,name_iso,mass_iso,charge_iso):
        self.list_isotopes.append(self.isotope(name_iso,mass_iso,charge_iso))

    def add_isotope_fluxes(self):
        i=0
        while i<len(self.list_isotopes):
            if i==0: 
                self.flux_energy_per_nucleon=np.array(self.list_isotopes[i].flux.copy())
                self.energy_per_nucleon=np.array(self.list_isotopes[i].energy_per_nucleon.copy())
            else:
                self.flux_energy_per_nucleon+=np.array(self.list_isotopes[i].flux.copy())
            i+=1
    #        def find_total_flux(self):
    def calc_total_flux(self,num_steps):
        if len(self.list_isotopes)>=2:
            # now flux_per_rigidity is different
            # find the min and max R common between the isotopes
            # then interpolate that flux on that range of R. add the fluxes at the end
            rigidities=[]
            energies=[]
            fluxes=[]
            rigidities_mod=[]
            energies_mod=[]
            fluxes_mod=[]
            energies_per_nuc_mod=[]
            i=0
            # now that the fluxes will be added together, this is last chance to adjust their units to not have nucleons
            # so divide by A (mass/# nucleons) and while thats going on lets just multiply 
            #in the 10^7 for converting cm^-2 to m^-2 and MeV^-1 to GeV^-1
            while i<len(self.list_isotopes):
                #fluxes.append(log_energy(np.true_divide(self.list_isotopes[i].flux.copy(),
                                                        #self.list_isotopes[i].mass*(10**(-7)))))
                fluxes.append(log_energy(self.list_isotopes[i].flux.copy()))
                rigidities.append(log_energy(self.list_isotopes[i].rigidity.copy()))
                energies.append(log_energy(self.list_isotopes[i].energy.copy()))
                fluxes_mod.append(log_energy(self.list_isotopes[i].flux_modulated.copy()))
                rigidities_mod.append(log_energy(self.list_isotopes[i].rigidity_modulated.copy()))
                energies_mod.append(log_energy(self.list_isotopes[i].energy_modulated.copy()))
                energies_per_nuc_mod.append(log_energy(self.list_isotopes[i].energy_modulated_per_nucleon.copy()))
                i+=1
            spline_min_R,spline_max_R=find_interpolation_range(*rigidities)
            spline_min_E,spline_max_E=find_interpolation_range(*energies)
            spline_min_R_mod,spline_max_R_mod=find_interpolation_range(*rigidities_mod)
            spline_min_E_mod,spline_max_E_mod=find_interpolation_range(*energies_mod)
            spline_min_En_mod,spline_max_En_mod=find_interpolation_range(*energies_per_nuc_mod)
            #now interpolate the isotopes log flux along the log rigidity in this common rig range
            i=0
            while i<len(self.list_isotopes):
                rig_rtn,flux_rtnr=spline(rigidities[i],fluxes[i], num_steps,spline_min_R,spline_max_R)
                ene_rtn,flux_rtne=spline(energies[i],fluxes[i], num_steps,spline_min_E,spline_max_E)
                flux_rtnr=np.array(undo_log_energy(flux_rtnr))
                flux_rtne=np.array(undo_log_energy(flux_rtne))
                rig_rtn_mod,flux_rtnr_mod=spline(rigidities_mod[i],fluxes_mod[i], num_steps,spline_min_R_mod,spline_max_R_mod)
                ene_rtn_mod,flux_rtne_mod=spline(energies_mod[i],fluxes_mod[i], num_steps,spline_min_E_mod,spline_max_E_mod)
                enen_rtn_mod,flux_rtnen_mod=spline(energies_per_nuc_mod[i],fluxes_mod[i], num_steps,spline_min_En_mod,spline_max_En_mod)
                flux_rtnr_mod=np.array(undo_log_energy(flux_rtnr_mod))
                flux_rtne_mod=np.array(undo_log_energy(flux_rtne_mod))
                flux_rtnen_mod=np.array(undo_log_energy(flux_rtnen_mod))
                if i==0:
                    self.flux_rigidity=np.array(flux_rtnr.copy())
                    self.flux_energy=np.array(flux_rtne.copy())
                    self.flux_rigidity_modulated=np.array(flux_rtnr_mod.copy())
                    self.flux_energy_modulated=np.array(flux_rtne_mod.copy())
                    self.flux_energy_per_nucleon_modulated=np.array(flux_rtnen_mod.copy())
                else:
                    self.flux_rigidity= self.flux_rigidity+np.array(flux_rtnr.copy())
                    self.flux_energy= self.flux_energy+np.array(flux_rtne.copy())
                    self.flux_rigidity_modulated= self.flux_rigidity_modulated+np.array(flux_rtnr_mod.copy())                
                    self.flux_energy_modulated= self.flux_energy_modulated+np.array(flux_rtne_mod.copy())
                    self.flux_energy_per_nucleon_modulated= self.flux_energy_per_nucleon_modulated+np.array(flux_rtnen_mod.copy())                
                i+=1
            self.rigidity=np.array(undo_log_energy(rig_rtn.copy()))
            self.energy=np.array(undo_log_energy(ene_rtn.copy()))
            self.rigidity_modulated=np.array(undo_log_energy(rig_rtn_mod.copy()))
            self.energy_modulated=np.array(undo_log_energy(ene_rtn_mod.copy()))
            self.energy_per_nucleon_modulated=np.array(undo_log_energy(enen_rtn_mod.copy()))
            self.phi=self.list_isotopes[0].phi
        else:
            print("add more isotopes\n")
    
    class isotope:
        def __init__(self, name,mass,charge):
            self.name = name    # instance variable unique to each instance
            self.energy_per_nucleon = []    # creates a new empty list
            self.energy = []
            self.rigidity = []
            self.flux = []
            self.mass = mass
            self.charge = charge
            self.energy_modulated = []
            self.energy_modulated_per_nucleon = []
            self.rigidity_modulated = []
            self.flux_modulated = []
            self.phi = 0
        ## NO LOGS    
        def add_energy_per_nucleon(self, energy_per_nuc):
            self.energy_per_nucleon=np.array(energy_per_nuc)
            # now recalculate the energy, and the rigidity, 
            self.rigidity=np.array(rigidity_calc(energy_per_nuc.copy(),self.mass,self.charge))
            self.energy=np.array(energy_per_nuc.copy())*self.mass
    
        ## NO LOGS
        def add_flux(self,flux_in):
            #correct the flux?
            mev_nuc=np.array(self.energy_per_nucleon.copy()*10**3)
            self.flux=np.array(np.true_divide(flux_in,mev_nuc*mev_nuc*self.mass*(10**(-7))))
            #fluxes.append(log_energy(np.true_divide(self.list_isotopes[i].flux.copy(),
                                      #self.list_isotopes[i].mass*(10**(-7)))))
        def add_modulation(self,solar):
            # once the fluxes+phi have been input then the modulation can occur as well, given a solar phi
            self.phi=solar
            self.flux_modulated,self.energy_modulated=force_field_approx(self.flux.copy(),self.charge,self.mass,self.phi,self.energy.copy())
            self.energy_modulated_per_nucleon=np.array(self.energy_modulated.copy()/self.mass)
            self.rigidity_modulated=np.array(rigidity_calc(self.energy_modulated_per_nucleon.copy(),self.mass,self.charge))
            
