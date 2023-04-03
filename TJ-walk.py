import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy import stats 
from scipy import signal 
from scipy.fft import fft 
from scipy.fft import fftfreq 

#FUNCTIONS

def plot_lines(X,Y,Z):                                                  #xaxis is 0 - N/F seconds elapsed over walk
    xaxis = np.linspace(0,N/F,N)
    plt.figure(figsize=(15,10))
    pltx = plt.subplot(3,1,1)
    plty = plt.subplot(3,1,2)
    pltz = plt.subplot(3,1,3)
    pltx.plot(xaxis,xAccel,label="Raw")
    pltx.plot(xaxis,X,label="Filtered")
    pltx.set_title("X Axis")
    plty.plot(xaxis,yAccel,label="Raw")
    plty.plot(xaxis,Y,label="Filtered")
    plty.set_title("Y Axis")
    plty.set_ylabel("Acceleration (m/s^2)",fontsize=20)
    pltz.plot(xaxis,zAccel,label="Raw")
    pltz.plot(xaxis,Z,label="Filtered")
    pltz.set_title("Z Axis")
    pltz.set_xlabel("Time (seconds)",fontsize=20)
    pltx.legend(loc='upper left')
    plty.legend()
    pltz.legend(loc="upper left")
    #plt.show()
    plt.savefig("TJ-acceleration.png")
    #plt.save later to add to report

def plot_hist(x,y,z):
    plt.figure(figsize=(12,5))
    pltx = plt.subplot(1,3,1)
    plty = plt.subplot(1,3,2)
    pltz = plt.subplot(1,3,3)
    pltx.hist(x)
    pltx.set_title("X Axis")
    pltx.set_ylabel("Count",fontsize=20)
    plty.hist(y)
    plty.set_title("Y Axis")
    plty.set_xlabel("Acceleration (m/s^2)",fontsize=20)
    pltz.hist(z)
    pltz.set_title("Z Axis")
    #plt.show()

def applyGravity(ax):
    ax*= GRAVITY
    return ax

def cleanData(ax):
    b,a = signal.butter(3,0.1,btype='lowpass',analog=False)            #clean the data using butterworth filter
    clean = signal.filtfilt(b,a,ax)
    return clean

def applyFFT(ax):
    transform = fft(ax)
    return transform

#argmax will tell us the approximate frequency of the sinusoid in the form of an index
#we can then use this value to index the DFT sample frequencies created by fftfreq. 
#this value corresponds to cycles/second because of the spacing we defined for the FFT data to be in seconds
#step frequency only makes sense in the x axis because of me walking forward 
#and applying force in the x direction and orientation of the phone (ankle, screen facing out)

def stepFrequency(x):                                                   #x is already FFT'd
    sampleFreqs = fftfreq(N)                                            #where N is the size of the x axis (and y and z)
    absxf = np.abs(x)                                                   #need to absolute value the FFT'd data to find the peak with the largest magnitude. Can't just cut graph in half because it is not symmetrical
    value = np.argmax(absxf)
    frequency = sampleFreqs[value]
    return frequency


#CONSTANTS
F = 20      #Frequency of accelerometer, measured in Hz
GRAVITY = 9.81  #m/s^2

#MAIN CODE
df = pd.read_csv('TJwalk.txt',sep=';',index_col = 'index')              #read in the data
df = df[(df.index >24) & (df.index <2150)]                              #shave off entries that may not be periodic to un-bias the readings

N = df['x'].size                                                        #get size of data entries
#print(N)
xAccel = applyGravity(df['x'])
yAccel = applyGravity(df['y'])
zAccel = applyGravity(df['z'])                                          #z axis doesn't really help to answer any analysis, so we will exclude it

filtX = cleanData(xAccel)
filtY = cleanData(yAccel)
filtZ = cleanData(zAccel)

plot_lines(filtX,filtY,filtZ)                                           #plot the smooth data

xf = applyFFT(filtX)                                                    #Apply a Fourier Transform to the cleaned data
yf = applyFFT(filtY)
zf = applyFFT(filtZ)

stepFreq = stepFrequency(xf)                                            #inspiration from https://stackoverflow.com/questions/3694918/how-to-extract-frequency-associated-with-fft-values-in-python
stepsMin = stepFreq * F * 60 * 2                                            #multiply by F to get steps/second
timeElapsed = N / F
numSteps = stepsMin * (timeElapsed/60)
print("Step Frequency: "+ str(stepFreq) + " steps/second")
print("Step Per Minute: "+ str(stepsMin) + " steps/minute")
print("Time Elapsed: "+str(timeElapsed) + " seconds")
print("Steps Taken: "+str(numSteps) + " steps")

#CLT                                                                    #Normal tests for acceleration 
plot_hist(filtX,filtY,filtZ)                                            #pvalues indicate non-normalitity, but with CLT we can assume normality
print("Normal Tests' p-value for Accelerations")                        #p-value way under 0.05
print("X " + str(stats.normaltest(filtX).pvalue))
print("Y " + str(stats.normaltest(filtY).pvalue))
print("Z " + str(stats.normaltest(filtZ).pvalue)) 
xspace = np.linspace(0,N/F,N)

print("Levene Tests' p-value for Accelerations :" + str(stats.levene(filtX,filtY,filtZ).pvalue))                        #p-value way under 0.05 as well (numeric underflow)

#Adding to CSV file
tempDF = pd.DataFrame(columns = ['name','age','gender','height','steps_min','time','steps','pace'])
new_row = {'name':'TJ','age':22,'gender':'male','height':172,'steps_min':stepsMin,'time':timeElapsed,'steps':numSteps,'pace':stepsMin}
tempDF = tempDF.append(new_row,ignore_index=True)
tempDF.to_csv('walkdata.csv',index=False)

#frequencies for ANOVA 
total = filtX + filtY + filtZ
temp = pd.DataFrame(columns=['TJ','Hana','Angus','Barry','Kevin','Anna','Adam'])
temp['TJ'] = total
temp.to_csv('anovafreq.csv',index=False)