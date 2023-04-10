import src.hermite_interpolation.hermite_filter as hi
import numpy as np
import matplotlib.pyplot as plt

def main():
    time = np.linspace(0, 2.5, 251)
    wave = 2 / np.pi * np.arcsin(np.sin(np.pi * time))
    wave = wave*wave*wave
    index = np.array([0,50,100,150,200,251], dtype=int)
    nelem, map, mat = hi.assemble_hermite_mat(time, index, 4)
    pars = np.einsum("ij,j->i", mat, wave)
    filtered = np.zeros_like(wave)
    for m, k, i, j in zip(map, nelem, index, index[1:]):

      filtered[i:j+1] = hi.hermite_interpolation(time[i:j+1], pars[m], time[i], time[min(j,len(time)-1)], n=k)
    fig, axs = plt.subplots(2,1,figsize=(4,6), dpi=300)
    axs[0].plot(wave)
    axs[0].plot(filtered)
    axs[1].plot(wave-filtered)
    plt.tight_layout()
    plt.show()
    plt.close()



if __name__=='__main__':
    main()