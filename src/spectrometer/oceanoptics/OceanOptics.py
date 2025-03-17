import seabreeze.spectrometers as sb
from time import sleep, localtime, strftime
import numpy as np
import os
import sys
import LivePlot as ulp

class spectrometers():
    def __init__(self):
        self.devices = sb.list_devices()

class spectrum():
    def __init__(self,devices,device_ID=0,integration_time=100000,averages=1):
        self.device = sb.Spectrometer(devices[device_ID])
        self.device.integration_time_micros(integration_time)
        self.wavelength = self.device.wavelengths()
        self.averages = averages
        self.background = np.zeros(self.device.pixels)

    def collect_intensities(self):
        self.intensities = np.zeros(self.device.pixels)
        for i in range(self.averages):
            self.intensities += (self.device.intensities() - self.background)/self.averages

    def collect_background(self):
        for i in range(self.averages):
            self.background += self.device.intensities()/self.averages

    def save(self,save_dir,save_name = None):
        if save_name is None:
            save_name = strftime('%Y%m%d_%H%M%S',localtime())
        self.file = os.path.join(save_dir, save_name) + '.txt'

        with open(self.file, 'w') as f:
            for i, wl in enumerate(self.wavelength):
                f.write('{},{}\n'.format(wl, self.intensities[i]))

    def plot(self,wavelength_limits = None,intensity_limits = None):
        if hasattr(self,'intensities'):
            ulp.plot_creation(self.wavelength,self.intensities,x_lim=wavelength_limits, y_lim=intensity_limits,  x_label = 'Wavelength (nm)', y_label = 'Intensity (counts)', title = 'Spectrum', live=False)
        else:
            print('No spectrum found.')

    def plot_live(self, wavelength_limits = None, intensity_limits = None, rescale = True):
        self.rolling_averages = self.averages
        self.averages = 1
        self.collect_intensities()
        conn, fig, keep_alive = ulp.plot_creation(self.wavelength, self.intensities,  x_lim = wavelength_limits, y_lim = intensity_limits,  x_label = 'Wavelength (nm)', y_label = 'Intensity (counts)', title = 'Spectrum', rescale_y = rescale)
        self.all_intensities = self.intensities
        update = True
        sleep(0.5)

        while update:
            self.update = update
            self.collect_intensities()
            self.all_intensities = np.vstack((self.all_intensities,self.intensities))
            if self.all_intensities.shape[0]>self.rolling_averages:
                self.all_intensities = self.all_intensities[1:,:]
            self.average_intensities = self.all_intensities.mean(axis=0)
            
            wavelengths = self.wavelength
            intensities = self.average_intensities
            if wavelength_limits is not None:
                filt_1 = wavelengths>=wavelength_limits[0]
                filt_2 = wavelengths<=wavelength_limits[1]
                filt_12 = np.logical_and(filt_1,filt_2)
            else:
                filt_12 = np.ones_like(wavelengths)
            if intensity_limits is not None:
                filt_3 = intensities>=intensity_limits[0]
                filt_4 = intensities<=intensity_limits[1]
                filt_34 = np.logical_and(filt_3,filt_4)
            else:
                filt_34 = np.ones_like(intensities)
            filt = np.logical_and(filt_12,filt_34)
            wavelengths_filt = wavelengths[filt]
            intensities_filt = intensities[filt]
            
            update = ulp.plot_update(conn, wavelengths_filt, intensities_filt)

        self.averages = self.rolling_averages
        keep_alive.terminate()

def main():
    wavelength_limits = [500, 1000]
    wavelength_limits = [300, 1000]
    # wavelength_limits = [350, 500]
    # intensity_limits = [-1000, 30000]
    # wavelength_limits = [700, 900]
    # intensity_limits = [2500, 15000]
    
    averages = 1
    integration_time=50000

    specs = spectrometers()
    spec = spectrum(specs.devices,integration_time=integration_time,averages=averages)
    
    # spec.collect_background()
    spec.collect_intensities()
    
    # spec.plot()

    # save_path = 'F:\\AaronData\\2023\\04\\28'
    # save_name = '300mW_1pt8bar_Ar_NCLInt_Trans'
    # print(save_name)
    # spec.save(save_path, save_name=save_name)
    spec.plot(wavelength_limits = wavelength_limits)
    # spec.plot(wavelength_limits = wavelength_limits, intensity_limits = intensity_limits)

    # spec.plot_live()

if __name__ == "__main__":
    main()
