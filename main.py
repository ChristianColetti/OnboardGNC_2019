#!/usr/bin/env python

from __future__ import print_function
from pymavlink import mavutil
import multiprocessing, time
import numpy as np
import air
from CCimport import *

# define indices of xh for easier access.
x, y, z, vt, alpha, beta, phi, theta, psi, p, q, r = range(12)

# define indices of y for easier access.
ax, ay, az, gyro_p, gyro_q, gyro_r, mag_x, mag_y, mag_z = range(9)  #z
mag_psi = 6
pres_baro = 9
gps_posn_n, gps_posn_e, gps_posn_d, gps_vel_n, gps_vel_e, gps_vel_d = range(10,16)

# define indices of servo for easier access.
mode_flag = 0
rcin_0, rcin_1, rcin_2, rcin_3, rcin_4, rcin_5 = range(1,7)
servo_0, servo_1, servo_2, servo_3, servo_4, servo_5 = range(7,13)
throttle, aileron, elevator, rudder, none, flaps = range(7,13)

# define indices of cmd for easier access.
psi_c, h_c = range(2)

def estimator_loop(y,xh,servo):
    global u
    # get sensors for read_sensor function call.
    adc,imu,baro,ubl = air.initialize_sensors()
    time.sleep(3)
    count=0
    #Sensor installation details
    Rba=np.array([[0,-1,0], [-1,0,0], [0,0,1]]) #acc frame to body (eg, imu to body)
    #Environmental parameters
    declination=+3.233*np.pi/180 #rad, mag declination is +3deg14' (eg, east) in Stillwater 
    pres_sl=1010		#millibars, sea level pressure for the day. Update me! 1mb is 100Pa
    rhoSL=1.225			#kg/m^2, sea level standard density
    g=9.8065 #m/s^2, gravity

    print('Coletti Estimator-Controller V0.9, remember to check V_e and V_d sign conv for H on ins steps')

    #bias calculate
    print('Performing bias calibration...')
    #master.mav.statustext_send(mavutil.mavlink.MAV_SEVERITY_NOTICE, 'WARNING! LEVEL AIRCRAFT UNTIL FURTHER NOTICE!')
    time.sleep(2)
    # Give 10 seconds of warmup
    t1 = time.time()
    gyro = np.array([[0, 0, 0]])
    accel = np.array([[0, 0, 0]])
    mag = np.array([[0, 0, 0]])
    while time.time() - t1 < 10:
        m9a, m9g, m9m = imu.getMotion9()
        accel = np.append(accel, [m9a], axis=0)
        gyro = np.append(gyro, [m9g], axis=0)
        mag = np.append(mag, [m9m], axis=0)
        time.sleep(0.05)
    gyro_bias = [np.average(gyro[:, 0]), np.average(gyro[:, 1]), np.average(gyro[:, 2])]
    accel_bias = [np.average(accel[:, 0]), np.average(accel[:, 1]), np.average(accel[:, 2])]
    mag_bias = [np.average(mag[:, 0]), np.average(mag[:, 1]), np.average(mag[:, 2])]
    print('bias calibration complete')

    # ==========================================================================
    # Logging Initialization
    # POC: Charlie
    now = datetime.now()
    date_time = now.strftime('%y_%m_%d__%H_%M_%S')
    os.chdir('/home/pi/')
    f_logfile = open('log_' + date_time + '.csv', 'w+')
    est_log_string = 'p_n, p_e, -h_b, Vt, alpha, beta, phi_a, theta_a, psi_m, p, q, r, rcin_0, rcin_1, rcin_2, rcin_3, rcin_4, rcin_5, rcin_6, servo_0, servo_1, servo_2, servo_3, servo_4, servo_5, ax, ay, az, gyro_q, gyro_p, gyro_r, mag_x, mag_y, mag_z, pres_baro, gps_posn_n, gps_posn_e, gps_posn_d, gps_vel_n, gps_vel_e, gps_vel_d\n'
    f_logfile.write(est_log_string)
    # ==========================================================================

    # Define Q here
    Q = np.eye(12)
                                                                                    #cov psi_mag                                                     #cov baro, GPS cov: n,           e,            d,         vn,        ve           vd
    R_INS = np.diag([np.cov(accel[:, 0]),np.cov(accel[:, 1]),np.cov(accel[:, 2]),2354.336792,np.cov(gyro[:, 0]),np.cov(gyro[:, 1]),np.cov(gyro[:, 2]),524071.8484, 532.9199581, 23.98810422, 677884.7773, 2.580815226, 2.214463558, 13.63108936,])

    R_AHRS = np.diag([np.cov(accel[:, 0]),np.cov(accel[:, 1]),np.cov(accel[:, 2]),2354.336792,np.cov(gyro[:, 0]),np.cov(gyro[:, 1]),np.cov(gyro[:, 2])])
                                                                                    #cov psi_mag
    accel = 0
    gyro = 0
    mag = 0

    controlsDer = np.array([0, 1, -23.448, 0, -50.313, -0.104, 8.169, 1137, 30.394, -14.904, -167.131])
    Xde, XdT, Zde, ZdT, MdeMwZde, MdTMwZdT, Ydr, Lda, Ldr, Nda, Ndr = controlsDer

    B=np.array([[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [Xde, XdT, 0, 0],
               [0, 0, 0, Ydr],
               [Zde, ZdT, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, Lda, Ldr],
               [MdeMwZde, MdTMwZdT, 0, 0],
               [0, 0, Nda, Ndr]])

    while True:
        initialEstTime = time.time()
        new_gps = air.read_sensor(y,adc,imu,baro,ubl) # updates values in y
        #initiate
        if count == 0:
            xh_old = [0,0,0,0,0,0,0,0,np.arctan2(y[mag_y], y[mag_x])+declination,0,0,0]
            P_old = numpy.eye(len(xh))
            
        #First compute sensor-specific estimates for imu/mag sensors
        #Raw sensor data needs to be rotated to stability axes
        acc=Rba.dot( np.array([y[ax]-accel_bias[0],y[ay]-accel_bias[1],y[az]-accel_bias[2]]) )
        mag=Rba.dot( np.array([y[mag_x]-mag_bias[0],y[mag_y]-mag_bias[1],y[mag_z]-mag_bias[2]]))
        
        #Magnetic heading (psi_m)
        Rhb=np.array([[np.cos(xh_old[theta]), np.sin(xh_old[theta])*np.sin(xh_old[phi]), np.sin(xh_old[theta])*np.cos(xh_old[phi])], [0, np.cos(xh_old[phi]), -np.sin(xh_old[phi])], [-np.sin(xh_old[theta]), np.cos(xh_old[theta])*np.sin(xh_old[phi]), np.cos(xh_old[theta])*np.cos(xh_old[phi])]]) #rotation from body to horizontal plane
        magh=Rhb.dot(mag) #mag vector in horizontal plane components
        psi_m = np.arctan2(magh[1], magh[0]) + declination
        Rbf = np.array([[cos(xh_old[theta]) * cos(psi_m), cos(xh_old[theta]) * sin(psi_m), -sin(xh_old[theta])], [sin(xh_old[phi]) * sin(xh_old[theta]) * cos(psi_m) - cos(xh_old[phi]) * sin(psi_m),sin(xh_old[phi]) * sin(xh_old[theta]) * sin(psi_m) + cos(xh_old[phi]) * cos(psi_m), sin(xh_old[phi]) * cos(xh_old[theta])],[cos(xh_old[phi]) * sin(xh_old[theta]) * cos(psi_m) + sin(xh_old[phi]) * sin(psi_m), cos(xh_old[phi]) * sin(xh_old[theta]) * sin(psi_m) - sin(xh_old[phi]) * cos(psi_m), cos(xh_old[phi]) * cos(xh_old[theta])]]) #rotation from fixed to body frame

        #Pressure altitude  
        h_b= -(y[pres_baro]-pres_sl)*100 /(rhoSL*g)  #*100  from mb to Pa

        dt = 0.125
    
        #=====ESTIMATOR CODE STARTS HERE==================================
        xh = xh_old
        A = Afunc(xh)
        F = FindF(A, dt)
        G = FindG(A, F, B, dt)
        P = P_old
        [xhminus, Pminus] = priori(xh, u, P, F, G)
        
        #Handle GPS and then fuse all measurements
        if (new_gps):
            z = [y[ax]-accel_bias[0],y[ay]-accel_bias[1],y[az]-accel_bias[2], psi_m, y[gyro_p]-gyro_bias[0], y[gyro_q]-gyro_bias[1], y[gyro_r]-gyro_bias[2], y[gps_posn_n], y[gps_posn_e], h_b, y[gps_vel_n], y[gps_vel_e], y[gps_vel_d]]
            H = H_INS(xh)
            [xh, P] = posteriori(xhminus, Pminus, z, H, R_INS)
      
        else:
            z = [y[ax]-accel_bias[0],y[ay]-accel_bias[1],y[az]-accel_bias[2], psi_m, y[gyro_p]-gyro_bias[0], y[gyro_q]-gyro_bias[1], y[gyro_r]-gyro_bias[2]]
            H = H_AHRS(xh)
            [xh, P] = posteriori(xhminus, Pminus, z, H, R_AHRS)
        #OUTPUT: write estimated values to the xh array--------------------------------------
        xh_old = xh
        P_old = P

        alpha = atan(xh[5]/xh[3]) #w/u
        beta = asin(xh[4]/xh[3]) #v/u only good for mostly u flight
                # DONE: Log X Hat, Servos, RCs, Y to CSV
        f_logfile.write(
            ', '.join(map(str, xh)) + ', ' + ', '.join(map(str, servo)) + ', ' + ', '.join(map(str, y)) + '\n')

        count=count+1

	if (count % 8000)==0:
	    print("Phi=%3.0f, Theta=%3.0f, Psi=%3.0f" %  (xh_old[phi]*180/np.pi, xh_old[theta]*180/np.pi, psi_m*180/np.pi))
	#======ESTIMATOR CODE STOPS HERE===================================

	#if (0.0125- (time.time()-initialEstTime) < 0): print( 1/(time.time()-initialEstTime) )
        time.sleep(max(0.0125-(time.time()-initialEstTime),0) )

def controller_loop(xh,servo,cmd):

    while True:
    	initial_time=time.time()
        # print("total milliseconds between controller_loop iterations: {}".format(initial_time-last_time))
        if (servo[mode_flag] == 1):
            print(xh)
	    #======CONTROLLER CODE STARTS HERE===============================================
            # rewrite servo_out values to servo array based on their previous values and xh, cmd
            # if (servo[servo_1]<1.5): servo[servo_1] = 1.55
            # else: servo[servo_1] = 1.45
            # time.sleep(1)
            #Controller should assign values in range 1.25 to 1.75 to outputs;
            #WARNING, servo damage likely if values outside this range are assigned
            #Example: This is a manual passthrough function
            servo[throttle]=servo[rcin_0]
            servo[aileron]=servo[rcin_1]
            servo[elevator]=servo[rcin_2]
            servo[rudder]=servo[rcin_3]
            servo[servo_4]=servo[servo_4] #no servo; channel used for manual/auto switch
            servo[flaps]=servo[rcin_5]
	    #=======CONTROLLER CODE STOPS HERE ======================================
        time.sleep(max(0.0125-(time.time()-initial_time),0) )






def priori(xh, u, P, F, G):
    # do not forget to initialize xh and P.
    FT = F.T
    Pminus = np.dot(np.dot(F, P), FT)
    xhatminus = np.add(np.dot(F, xh), np.dot(G, u))
    return xhatminus, Pminus


def posteriori(xhatminus, Pminus, z, H, R):
    ss = len(xhatminus)  # state space size
    HT = H.T
    # calculate Kalman gain
    Knumerator = dot(Pminus, HT)
    Kdenominator = dot(dot(H, Pminus), HT) + R
    K = dot(Knumerator, np.linalg.inv(Kdenominator))  # Kalman gain

    residuals = z - dot(H, xhatminus)
    xhat = xhatminus + dot(K, residuals)
    one_minus_KC = numpy.eye(ss) - dot(K, H)

    # compute a posteriori estimate of errors
    P = dot(one_minus_KC, Pminus)

    return xhat, P


if __name__ == "__main__":

    master = mavutil.mavlink_connection('/dev/ttyAMA0', baud=57600, source_system=255)

    # initialize arrays for sharing sensor data.
    y = multiprocessing.Array('d', np.zeros(26)) # imu, baro, gps, adc
    xh = multiprocessing.Array('d', np.zeros(12)) # position, orientation, rates
    servo = multiprocessing.Array('d', np.zeros(13)) # mode_flag, rcin, servo_out
    cmd = multiprocessing.Array('d', np.zeros(2)) # psi_c, h_c

    # start processes for interpreting sensor data and setting servo pwm.
    estimator_process = multiprocessing.Process(target=estimator_loop, args=(y,xh,servo))
    estimator_process.daemon = True
    estimator_process.start()
    controller_process = multiprocessing.Process(target=controller_loop, args=(xh,servo,cmd))
    controller_process.daemon = True
    controller_process.start()
    servo_process = multiprocessing.Process(target=air.servo_loop, args=(servo,))
    servo_process.daemon = True
    servo_process.start()
    time.sleep(3)
    # start process for telemetry after other processes have initialized.
    telemetry_process = multiprocessing.Process(target=air.telemetry_loop, args=(y,xh,servo,master))
    telemetry_process.daemon = True
    telemetry_process.start()

    print("\nsending heartbeats to {} at 1hz.".format('/dev/ttyAMA0'))
    # loop for sending heartbeats and receiving messages from gcs.
    while True:
        # send heartbeat message periodically
        master.mav.heartbeat_send(1, 0, 0, 0, 4, 0)
        # still haven't figured out how to get mode to show up in mission planner.
        # print('heartbeat sent.')
        time.sleep(0.5)
	#=====WAYPOINT TRACKER STARTS HERE======================
	#Simple waypoint tracker
        #
	#=====WAYPOINT TRACKER STOPS HERE=======================

    # handle incoming commands over telemetry
        # try:
        #     msg = master.recv_match().to_dict()
        #     if (not (msg['mavpackettype'] == 'RADIO' or msg['mavpackettype'] == 'RADIO_STATUS' or msg['mavpackettype'] == 'HEARTBEAT')):
        #         print(msg)
        #         if (msg['mavpackettype'] == 'COMMAND_LONG'):
        #             master.mav.command_ack_send(msg['command'],4)
        #             print("acknowledge sent.")
        # except:
        #     pass
