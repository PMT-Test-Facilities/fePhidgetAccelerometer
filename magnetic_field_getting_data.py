#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:18:24 2020

@author: vincentgousy-leblanc
"""

from Phidget22.PhidgetException import *
from Phidget22.Phidget import *
from Phidget22.Devices.Log import *
from Phidget22.LogLevel import *
from Phidget22.Devices.Magnetometer import *
import traceback
import time

mgx=[]
mgy=[]
mgz=[]
timelist1=[]

#Declare any event handlers here. These will be called every time the associated event occurs.

def onMagneticFieldChange(self, magneticField, timestamp):
	#print("MagneticField: \t"+ str(magneticField[0])+ "  |  "+ str(magneticField[1])+ "  |  "+ str(magneticField[2]))
	#print("Timestamp: " + str(timestamp))
    mgx.append(magneticField[0])
    mgy.append(magneticField[1])
    mgz.append(magneticField[2])
    print("----------")

def onAttach(self):
	print("Attach!")

def onDetach(self):
	print("Detach!")

def onError(self, code, description):
	print("Code: " + ErrorEventCode.getName(code))
	print("Description: " + str(description))
	print("----------")

def main():
	try:
		Log.enable(LogLevel.PHIDGET_LOG_INFO, "phidgetlog.log")
		#Create your Phidget channels
		magnetometer0 = Magnetometer()

		#Set addressing parameters to specify which channel to open (if any)

		#Assign any event handlers you need before calling open so that no events are missed.
		magnetometer0.setOnMagneticFieldChangeHandler(onMagneticFieldChange)
		magnetometer0.setOnAttachHandler(onAttach)
		magnetometer0.setOnDetachHandler(onDetach)
		magnetometer0.setOnErrorHandler(onError)

		#Open your Phidgets and wait for attachment
		magnetometer0.openWaitForAttachment(5000)

		#Do stuff with your Phidgets here or in your event handlers.

		try:
			input("Press Enter to Stop\n")
		except (Exception, KeyboardInterrupt):
			pass

		#Close your Phidgets once the program is done.
		magnetometer0.close()

	except PhidgetException as ex:
		#We will catch Phidget Exceptions here, and print the error informaiton.
		traceback.print_exc()
		print("")
		print("PhidgetException " + str(ex.code) + " (" + ex.description + "): " + ex.details)

main()



#%%
x=np.linspace(0,118,len(mgx))

plt.plot(x,np.array([mgx])[0]*1000,'r.',label='mgx (mG)')
plt.plot(x,np.array([mgy])[0]*1000,'b.',label='mgy (mG)')
plt.plot(x,np.array([mgz])[0]*1000,'g.',label='mgz (mG)')
plt.legend()
plt.xlabel("lenght (cm)")
plt.ylabel("magnetic field ")
