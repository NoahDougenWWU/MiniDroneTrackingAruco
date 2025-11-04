import time
import cv2
import numpy as np
from kalmanquat import KalmanFilterQuaternion
import logging
import sys
from threading import Event


import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper
from cflib.crazyflie.log import LogConfig

URI = uri_helper.uri_from_env(default='radio://0/15/2M/E7E7E7E7E7')
deck_attached_event = Event()

DEFAULT_HEIGHT = 0.5 # meters


def param_deck_flow(_, value_str):
    value = int(value_str)
    print(value)
    if value:
        deck_attached_event.set()
        print('Deck is attached!')
    else:
        print('Deck is NOT attached!')

def rotmat_to_quat(R):
    """Convert a 3x3 rotation matrix to a quaternion [x, y, z, w]."""
    trace = np.trace(R)

    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2  # S = 4 * qw
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

    q = np.array([qx, qy, qz, qw])
    return q / np.linalg.norm(q)

def quat_to_rotmat(q):
    """Convert a quaternion [x, y, z, w] to a 3x3 rotation matrix."""
    x, y, z, w = q
    n = x*x + y*y + z*z + w*w
    if n < 1e-8:
        return np.eye(3)  # Identity for near-zero quaternion

    s = 2.0 / n
    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s

    R = np.array([
        [1.0 - (yy + zz),     xy - wz,         xz + wy],
        [xy + wz,             1.0 - (xx + zz), yz - wx],
        [xz - wy,             yz + wx,         1.0 - (xx + yy)]
    ])
    return R

def draw_pose_info(frame, measurement, marker_size, corners, ids, camera_matrix, dist_coeffs):
    """
    Draw 3D axes on detected markers to visualize pose.
    """
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, marker_size, camera_matrix, dist_coeffs
    )
    drone_state = [0,0,0,0]
    target_state = [0,0,0,0]
    if ids is None or rvecs is None or tvecs is None:
        return frame
    
    for i in range(len(ids)):
        # Draw the 3D axes
        cv2.drawFrameAxes(
            frame, camera_matrix, dist_coeffs, 
            rvecs[i], tvecs[i], 3
        )
        # Get the translation vector components
        x, y, z = tvecs[i][0]
        x_rot, y_rot, z_rot = rvecs[i][0]
            
        
        # Draw the distance text
        text = f"ID: {ids[i][0]}, X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}"
        cv2.putText(
            frame, 
            text, 
            (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 255), 
            2
        )
        # Draw the rotation text
        text = f"X_R: {x_rot:.2f}, Y_R: {y_rot:.2f}, Z_R: {z_rot:.2f}"
        cv2.putText(
            frame, 
            text, 
            (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 255), 
            2
        )

        if ids[i] == 2:
            target_state[0] = x
            target_state[1] = y
            target_state[2] = z
            target_state[3] = z_rot
        if ids[i] == 0:
            drone_state[0] = x
            drone_state[1] = y
            drone_state[2] = z
            drone_state[3] = z_rot
 

        rot_mat, _ = cv2.Rodrigues(rvecs[i])
        quat = rotmat_to_quat(rot_mat) 
        measurement = np.concatenate((tvecs[i].flatten(),quat))
    
    return frame, measurement, drone_state, target_state

def update_drone(mc, pid_dic, drone_state, target_state):
    
    # Get error for four states using offset
    
    offset = np.array((0,0,1,0))
    error = (target_state + offset) - drone_state

    
    # Add sum of errors for each state
    
    pid_dic["x_sum"] += error[0]
    pid_dic["y_sum"] += error[1]
    pid_dic["z_sum"] += error[2]

    
    # Use PID for each state to get new command
    
    x = (pid_dic["Px"] * error[0]) + (pid_dic["Ix"] * error[0]) + (pid_dic["Dx"] * (error[0] - pid_dic["prev_x"]))
    y = (pid_dic["Py"] * error[1]) + (pid_dic["Iy"] * error[1]) + (pid_dic["Dy"] * (error[1] - pid_dic["prev_y"]))
    z = (pid_dic["Pz"] * error[2]) + (pid_dic["Iz"] * error[2]) + (pid_dic["Dz"] * (error[2] - pid_dic["prev_z"]))

    
    # Store the new errors as the previous errors for derivative
    
    pid_dic["prev_x"] = error[0]
    pid_dic["prev_y"] = error[1]
    pid_dic["prev_z"] = error[2]
    #print(error)

    mc.start_linear_motion(z*0.001, -x*0.01, -y*0.01)
    return

def main():
    
    cflib.crtp.init_drivers()
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        scf.cf.param.add_update_callback(group='deck', name='bcFlow2',
                                         cb=param_deck_flow)
        time.sleep(1)

        if not deck_attached_event.wait(timeout=5):
            print('No flow deck detected!')
            sys.exit(1)

        # Arm the Crazyflie
        scf.cf.platform.send_arming_request(True)
        time.sleep(1.0)


        with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
            time.sleep(2.0)
            mc.stop
    
    if True:
            pid_dic = {
                "Px" : 1,
                "Py" : 1,
                "Pz" : 1,
                "Pzrot" : 1,
                "Ix" : 0,
                "Iy" : 0,
                "Iz" : 0,
                "Izrot" : 0,
                "Dx" : 0.0,
                "Dy" : 0.0,
                "Dz" : 0.0,
                "Dzrot" : 0.0,
                "x_sum" : 0,
                "y_sum" : 0,
                "z_sum" : 0,
                "zrot_sum" : 0,
                "prev_x" : 0,
                "prev_y" : 0,
                "prev_z" : 0,
                "prev_zrot" : 0
            }
            dt = 0.01
            update_timer = 0
            kf = KalmanFilterQuaternion(dt=dt)
            
            input_source = 0
            
            # Initialize webcam/video
            cap = cv2.VideoCapture(input_source)
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Video source: {width}x{height}")
            
            # Create ArUco dictionary and parameters
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters_create()
            
            # Generate a simple calibration matrix for visualization purposes
            focal_length = width
            center = (width / 2, height / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]], 
                dtype=np.float32
            )
            dist_coeffs = np.zeros((4, 1), dtype=np.float32)  # Assuming no lens distortion
            avg_drone_state = np.zeros((1,4))
            avg_target_state = np.zeros((1,4))
            
            try:
                while True:
                    
                    update_timer += 1
                    if update_timer == 10:
                        update_drone(mc, pid_dic, np.mean(avg_drone_state, axis=0), np.mean(avg_target_state, axis=0))
                        update_timer = 0
                        avg_drone_state = np.zeros((1,4))
                        avg_target_state = np.zeros((1,4))

                    drone_state = [0,0,0,0]
                    target_state = [0,0,0,0]
                    
                    measurement = None
                    ret, frame = cap.read()
                    if not ret:
                        print("End of video stream")
                        break
                    
                    # Convert to grayscale for detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Detect ArUco markers
                    corners, ids, rejected = cv2.aruco.detectMarkers(
                        gray, aruco_dict, parameters=aruco_params
                    )
                    
                    # Create a copy of the frame for visualization
                    output = frame.copy()
                    
                    if ids is not None and len(ids) > 0:
                        # Estimate and draw pose information
                        output, measurement, drone_state, target_state = draw_pose_info(output, measurement, 7.5, corners, ids, camera_matrix, dist_coeffs)

                    else:
                        cv2.putText(
                            output, 
                            "No markers detected", 
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (0, 0, 255), 
                            2
                        )

                    avg_drone_state = np.vstack((avg_drone_state, np.asarray(drone_state)))
                    avg_target_state = np.vstack((avg_target_state, np.asarray(target_state)))
                    
                    _ = kf.predict()
                    estimated = kf.update(measurement)
                    rvec, _ = cv2.Rodrigues(quat_to_rotmat(estimated[6:10].flatten()))
                    cv2.drawFrameAxes(
                        output, camera_matrix, dist_coeffs, 
                        rvec, estimated[:3].flatten(), 3
                    )

                    x, y, z = estimated[:3].flatten()
                    x_rot, y_rot, z_rot = rvec.flatten()
                    
                    text = f"Estimated position: X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}."
                    cv2.putText(
                        output, 
                        text, 
                        (0,420),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 0, 255), 
                        2
                        )
                    text = f"Estimated rotation: X: {x_rot:.2f}, Y: {y_rot:.2f}, Z: {z_rot:.2f}."
                    cv2.putText(
                        output, 
                        text, 
                        (0,445),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 0, 255), 
                        2
                        )
                    
                    # Show the output frame
                    cv2.imshow("ArUco Marker Tracker", output)
                    
                    # Exit on ESC key
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        break
                    time.sleep(dt)
                    
            finally:
                # Clean up
                #mc.stop()
                #time.sleep(1.0)
                #mc.land()
                cap.release()
                cv2.destroyAllWindows()
                print("Exiting...")


if __name__ == "__main__":
    main()