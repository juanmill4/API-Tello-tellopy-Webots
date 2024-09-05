from controller import Robot, Keyboard
from math import cos, sin, pi, radians, sqrt
import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import time
from pid_controller import pid_velocity_fixed_height_controller

FLYING_ATTITUDE = 1
MIN_TIME_BETWEEN_DIRECTION_CHANGES = 0.5  # tiempo mínimo en segundos entre cambios de dirección

# Cargar la matriz de la cámara y los coeficientes de distorsión desde un archivo .npz
with np.load('/home/user/tello-ai/calib_data/webots/crazy/MultiMatrix.npz') as X:
    mtx, dist = X['camMatrix'], X['distCoef']

# Tamaño de marcador ArUco en milímetros (ajusta según tus marcadores)
SIZE = 400

# Diccionario para almacenar distancias y ángulos de etiquetas detectadas
marker_data = {}
stored_distances = {'dist_1': [], 'dist_2': [], 'angle_1': [], 'angle_2': []}

# Preparar el diccionario y los parámetros de ArUco
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
parameters = aruco.DetectorParameters_create()

def detect_ArUco(img):
    Detected_ArUco_markers = {}
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if ids is not None:
        for i in range(len(ids)):
            Detected_ArUco_markers[str(ids[i][0])] = corners[i]
    return Detected_ArUco_markers

def Calculate_orientation_in_degree(Detected_ArUco_markers, camera_matrix, dist_coeffs):
    ArUco_marker_orientations = {}
    for aruco_id, corners in Detected_ArUco_markers.items():
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, SIZE, camera_matrix, dist_coeffs)
        rmat = cv.Rodrigues(rvec)[0]
        sy = np.sqrt(rmat[0,0] * rmat[0,0] + rmat[1,0] * rmat[1,0])
        singular = sy < 1e-6
        if not singular:
            pitch = np.arctan2(rmat[2,1], rmat[2,2])
            yaw = np.arctan2(-rmat[2,0], sy)
            roll = np.arctan2(rmat[1,0], rmat[0,0])
        else:
            pitch = np.arctan2(-rmat[1,2], rmat[1,1])
            yaw = np.arctan2(-rmat[2,0], sy)
            roll = 0
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)
        roll_deg = np.degrees(roll)
        pitch_deg = (pitch_deg + 180) % 360 - 180
        yaw_deg = (yaw_deg + 180) % 360 - 180
        roll_deg = (roll_deg + 180) % 360 - 180

        ArUco_marker_orientations[aruco_id] = {
            'yaw': yaw_deg,
            'pitch': pitch_deg,
            'roll': roll_deg,
            'rvec': rvec[0],
            'tvec': tvec[0]
        }
    return ArUco_marker_orientations

def clip_line_to_circle(center, radius, x1, y1, x2, y2):
    # Translate the line to the origin
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - center[0], y1 - center[1]

    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius

    discriminant = b * b - 4 * a * c
    if discriminant >= 0:
        discriminant = sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)

        if 0 <= t1 <= 1:
            x2 = int(x1 + t1 * dx)
            y2 = int(y1 + t1 * dy)
        elif 0 <= t2 <= 1:
            x2 = int(x1 + t2 * dx)
            y2 = int(y1 + t2 * dy)
    
    return x2, y2

def project_points(points, rvec, tvec, camera_matrix, dist_coeffs):
    projected_points, _ = cv.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
    return projected_points
    
def draw_infinite_line(img, point1, point2, color, thickness):
    # Tamaño de la imagen
    height, width = img.shape[:2]

    # Convertir los puntos a float para mayor precisión
    x1, y1 = point1
    x2, y2 = point2

    # Calcular la pendiente y la intersección
    if x2 - x1 != 0:
        m = (y1 - y2) / (x1 - x2)
        b = y1 - m * x1
    else:
        m = float('inf')
        b = y1  # La intersección en el eje y cuando x es constante

    # Función para calcular y dado x
    def calc_y(x):
        return int(-m * x + b)

    # Función para calcular x dado y
    def calc_x(y):
        return int((y - b) / m)

    # Determinar los puntos de intersección con los bordes de la imagen
    if m == float('inf'):
        p1 = (x1, 0)
        p2 = (x1, height)
    else:
        p1 = (0, calc_y(0))
        p2 = (width, calc_y(width))

    # Dibujar la línea en la imagen
    cv.line(img, p1, p2, color, thickness)


# Función para calcular los puntos de la línea extendida
def calculate_infinite_line_points(p1, p2, img_width):
    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    intercept = p1[1] - slope * p1[0]
    
    x1 = 0
    y1 = int(slope * x1 + intercept)
    
    x2 = img_width - 1
    y2 = int(slope * x2 + intercept)
    
    return (x1, y1), (x2, y2)

def draw_panel(img, markers_orientations=None, camera_matrix=None, dist_coeffs=None, yaw=0, mapping_mode=False, stored_distances=None, avg_dist_1 = None, avg_dist_2 = None):
    north_offset_angle = pi / 2
    img.fill(0)
    map_center = (500, 100)
    minimap_radius = 70
    outer_radius = minimap_radius + 10

    # Draw minimap circle
    cv.circle(img, map_center, minimap_radius, (255, 0, 0), 2)
    cv.circle(img, map_center, outer_radius, (255, 255, 255), 1)
    # Draw a red dot at the center of the minimap
    cv.circle(img, map_center, 5, (0, 0, 255), -1)

    if markers_orientations:
        for aruco_id, orientations in markers_orientations.items():
            rvec = orientations['rvec']
            tvec = orientations['tvec']
            roll_deg = orientations['roll']
            distance = np.linalg.norm(tvec)
            angle = np.arctan2(tvec[0][0], tvec[0][2])  # Ángulo en radianes utilizando X e Y
            
            
        
            marker_data[aruco_id] = {'distance': distance, 'angle': angle, 'tvec': tvec}
        
            max_distance = 8000
            scaled_distance = int((distance / max_distance) * minimap_radius)
            if distance < max_distance:
        
                # Ajustar la posición del marcador según el yaw
                north_corrected_angle =  angle - yaw - north_offset_angle
                marker_x = int(map_center[0] + scaled_distance * np.cos(north_corrected_angle))
                marker_y = int(map_center[1] - scaled_distance * np.sin(north_corrected_angle))
        
                # Proyectar puntos para las líneas perpendiculares
                perp_length = 400
                panel_points = np.float32([
                    [0, 0, 0], [0, SIZE, 0], [SIZE, SIZE, 0], [SIZE, 0, 0],
                    [0, 0, -SIZE/10], [0, SIZE, -SIZE/10], [SIZE, SIZE, -SIZE/10], [SIZE, 0, -SIZE/10]
                ])
                img_points = project_points(panel_points, rvec, tvec, camera_matrix, dist_coeffs)
                img_points = np.int32(img_points).reshape(-1, 2)
        
                x0, y0 = img_points[0]
                x1, y1 = img_points[3]
        
                # Calcular la pendiente solo si x1 != x0, de lo contrario establecer a infinito
                if x1 != x0:
                    slope = (y1 - y0) / (x1 - x0)
                else:
                    slope = float('inf')
        
                # Calcular los puntos finales de la línea solo una vez
                length = 100  # Aumentar la longitud para mayor sensibilidad
                if slope != float('inf'):
                    dx = length / np.sqrt(1 + slope ** 2)
                    dy = slope * dx
                else:
                    dx = 0
                    dy = length
        
                height, width, _ = img.shape
                # Calcular los puntos extendidos para las líneas
                line1_p1, line1_p2 = calculate_infinite_line_points(img_points[0], img_points[3], width)
                line2_p1, line2_p2 = calculate_infinite_line_points(img_points[1], img_points[2], width)
        
                # Definir el polígono que conecta los puntos extendidos
                pts = np.array([line1_p1, line1_p2, line2_p2, line2_p1], np.int32)
                pts = pts.reshape((-1, 1, 2))
        
                # Rellenar el polígono
                # cv.fillPoly(img, [pts], (255, 0, 0))
        
                perp_x1, perp_y1 = img_points[0]
                perp_x2, perp_y2 = img_points[3]
        
                # Invertir coordenadas para corregir el modo espejo
                line_x1 = int(marker_x + dx)
                line_y1 = int(marker_y - dy)  # Cambio aquí: -dy en lugar de +dy
                line_x2 = int(marker_x - dx)
                line_y2 = int(marker_y + dy)  # Cambio aquí: +dy en lugar de -dy
        
                # Clip lines to the circle's radius
                line_x1, line_y1 = clip_line_to_circle(map_center, minimap_radius, marker_x, marker_y, line_x1, line_y1)
                line_x2, line_y2 = clip_line_to_circle(map_center, minimap_radius, marker_x, marker_y, line_x2, line_y2)
        
                cv.line(img, (marker_x, marker_y), (line_x1, line_y1), (255, 0, 255), 2)
                cv.line(img, (marker_x, marker_y), (line_x2, line_y2), (255, 0, 255), 2)
        
                distance_text = f'{distance:.2f} mm'
                #cv.putText(img, distance_text, (marker_x + 10, marker_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                print(marker_data)

    # Dibuja el indicador de norte
    north_angle = -yaw - north_offset_angle
    north_x = int(map_center[0] + outer_radius * cos(north_angle))
    north_y = int(map_center[1] - outer_radius * sin(north_angle))
    cv.putText(img, 'N', (north_x, north_y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if mapping_mode:
        text = "Mapping"
        font = cv.FONT_HERSHEY_SIMPLEX
        text_size = cv.getTextSize(text, font, 1, 2)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = (img.shape[0] + text_size[1]) // 2
        cv.putText(img, text, (text_x, text_y), font, 1, (0, 0, 255), 2)

        if '0' in marker_data and '1' in marker_data:
            stored_distances['dist_1'].append(marker_data['0']['distance'])
            stored_distances['angle_1'].append(marker_data['0']['angle'])
            stored_distances['dist_2'].append(marker_data['1']['distance'])
            stored_distances['angle_2'].append(marker_data['1']['angle'])

    # Calcular las distancias medias si hay datos en stored_distances y no estamos en mapping_mode
    if not mapping_mode and stored_distances['dist_1'] and stored_distances['dist_2']:
        avg_dist_1 = np.mean(stored_distances['dist_1'])
        avg_angle_1 = np.mean(stored_distances['angle_1'])
        avg_dist_2 = np.mean(stored_distances['dist_2'])
        avg_angle_2 = np.mean(stored_distances['angle_2'])
        
        avg_dist_1 = avg_dist_1 / 50
        avg_dist_2 = avg_dist_2 / 50
        
        stored_distances['dist_1'].clear()
        stored_distances['dist_2'].clear()
        stored_distances['angle_1'].clear()
        stored_distances['angle_2'].clear()

    # Condición para dibujar el rectángulo
    if not mapping_mode and avg_dist_1 is not None and avg_dist_2 is not None:
        x_c, y_c = 250, 350

        vertices = [
            (x_c - avg_dist_1, y_c + avg_dist_2),
            (x_c + avg_dist_1, y_c + avg_dist_2),
            (x_c + avg_dist_1, y_c - avg_dist_2),
            (x_c - avg_dist_1, y_c - avg_dist_2),
            (x_c - avg_dist_1, y_c + avg_dist_2)
        ]

        vertices = [(int(x), int(y)) for x, y in vertices]

        for i in range(len(vertices) - 1):
            cv.line(img, vertices[i], vertices[i + 1], (255, 0, 255), 2)
            mid_x = (vertices[i][0] + vertices[i + 1][0]) // 2
            mid_y = (vertices[i][1] + vertices[i + 1][1]) // 2
            if i % 2 == 0:
                label = f'D1: {avg_dist_2:.2f}'
            else:
                label = f'D2: {avg_dist_1:.2f}'
            cv.putText(img, label, (mid_x, mid_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        drone_position = (int(x_c), int(y_c))
        cv.circle(img, drone_position, 5, (0, 0, 255), -1)
        
        # Actualizar la posición del dron en función de los datos de los marcadores detectados
        if marker_data and '0' in marker_data and '1' in marker_data:
            tvec_0 = marker_data['0']['tvec']
            tvec_1 = marker_data['1']['tvec']
            # Aquí se podría usar un promedio ponderado, o cualquier otro método de tu preferencia
            drone_x = (tvec_0[0][0] + tvec_1[0][0]) / 4
            drone_y = (tvec_0[0][2] + tvec_1[0][2]) / 4  # Usando la coordenada Z para y del plano

            # Convertir coordenadas de la posición del dron para dibujarlo en el panel (inverso)
            drone_position_x = int(x_c + drone_x / 50)  # Invertir la posición según tus necesidades
            drone_position_y = int(y_c - drone_y / 50)  # Invertir la posición según tus necesidades

            drone_position = (drone_position_x, drone_position_y)
        else:
            drone_position = (int(x_c), int(y_c))

        cv.circle(img, drone_position, 5, (0, 0, 255), -1)

    return avg_dist_1, avg_dist_2



if __name__ == '__main__':
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    m1_motor = robot.getDevice("m1_motor")
    m1_motor.setPosition(float('inf'))
    m1_motor.setVelocity(-1)
    m2_motor = robot.getDevice("m2_motor")
    m2_motor.setPosition(float('inf'))
    m2_motor.setVelocity(1)
    m3_motor = robot.getDevice("m3_motor")
    m3_motor.setPosition(float('inf'))
    m3_motor.setVelocity(-1)
    m4_motor = robot.getDevice("m4_motor")
    m4_motor.setPosition(float('inf'))
    m4_motor.setVelocity(1)

    imu = robot.getDevice("inertial_unit")
    imu.enable(timestep)
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)
    camera = robot.getDevice("camera")
    camera.enable(timestep)

    keyboard = Keyboard()
    keyboard.enable(timestep)

    past_x_global = 0
    past_y_global = 0
    past_time = 0
    first_time = True

    PID_crazyflie = pid_velocity_fixed_height_controller()
    PID_update_last_time = robot.getTime()
    sensor_read_last_time = robot.getTime()

    height_desired = FLYING_ATTITUDE
    autonomous_mode = False
    mapping_mode = False
    last_sideways_change_time = 0
    current_sideways = 0
    initial_yaw = None
    yaw_accumulated = 0
    avg_dist_1 = None
    avg_dist_2 = None

    panel_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    print("\n")
    print("====== Controls =======\n\n")
    print(" The Crazyflie can be controlled from your keyboard!\n")
    print(" All controllable movement is in body coordinates\n")
    print("- Use the up, back, right and left button to move in the horizontal plane\n")
    print("- Use Q and E to rotate around yaw\n ")
    print("- Use W and S to go up and down\n ")
    print("- Press A to start autonomous mode\n")
    print("- Press D to disable autonomous mode\n")
    print("- Press M to start mapping mode\n")

    while robot.step(timestep) != -1:
        dt = robot.getTime() - past_time
        actual_state = {}

        if first_time:
            past_x_global = gps.getValues()[0]
            past_y_global = gps.getValues()[1]
            past_time = robot.getTime()
            first_time = False

        roll = imu.getRollPitchYaw()[0]
        pitch = imu.getRollPitchYaw()[1]
        yaw = imu.getRollPitchYaw()[2]
        yaw1 = -((yaw + pi) % (2 * pi) - pi)  # Convertir yaw de [0, 2π) a [-π, π)
        yaw_rate = gyro.getValues()[2]
        x_global = gps.getValues()[0]
        v_x_global = (x_global - past_x_global)/dt
        y_global = gps.getValues()[1]
        v_y_global = (y_global - past_y_global)/dt
        altitude = gps.getValues()[2]

        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)
        v_x = v_x_global * cos_yaw + v_y_global * sin_yaw
        v_y = - v_x_global * sin_yaw + v_y_global * cos_yaw

        desired_state = [0, 0, 0, 0]
        forward_desired = 0
        sideways_desired = 0
        yaw_desired = 0
        height_diff_desired = 0

        key = keyboard.getKey()
        while key > 0:
            if key == Keyboard.UP:
                forward_desired += 0.5
            elif key == Keyboard.DOWN:
                forward_desired -= 0.5
            elif key == Keyboard.RIGHT or key == Keyboard.LEFT:
                new_direction = -1 if key == Keyboard.RIGHT else 1
                if current_sideways != new_direction:
                    if time.time() - last_sideways_change_time >= MIN_TIME_BETWEEN_DIRECTION_CHANGES:
                        sideways_desired = -0.5 if key == Keyboard.RIGHT else 0.5
                        last_sideways_change_time = time.time()
                        current_sideways = new_direction
                else:
                    sideways_desired = -0.5 if key == Keyboard.RIGHT else 0.5
            elif key == ord('Q'):
                yaw_desired = + 1
            elif key == ord('E'):
                yaw_desired = - 1
            elif key == ord('W'):
                height_diff_desired = 0.1
            elif key == ord('S'):
                height_diff_desired = - 0.1
            elif key == ord('A'):
                if not autonomous_mode:
                    autonomous_mode = True
                    print("Autonomous mode: ON")
            elif key == ord('D'):
                if autonomous_mode:
                    autonomous_mode = False
                    print("Autonomous mode: OFF")
            elif key == ord('M'):
                if not mapping_mode:
                    mapping_mode = True
                    initial_yaw = yaw1
                    yaw_accumulated = 0
                    print("Mapping mode: ON")
            key = keyboard.getKey()

        height_desired += height_diff_desired * dt

        camera_image = camera.getImage()
        height = camera.getHeight()
        width = camera.getWidth()
        image = np.frombuffer(camera_image, np.uint8).reshape((height, width, 4))
        image_bgr = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
        gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
        Detected_ArUco_markers = detect_ArUco(image_bgr)
        
        if Detected_ArUco_markers:
            print('aaaaaaaaaaaaaaaa')
        else:
            print('bbbbbbbbbbbbbbbb')

        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        if len(corners) > 0:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, SIZE, mtx, dist)

        if ids is not None:
            Detected_ArUco_markers = detect_ArUco(image_bgr)
            ArUco_marker_orientations = Calculate_orientation_in_degree(Detected_ArUco_markers, mtx, dist)

            for i, aruco_id in enumerate(ids.flatten()):
                aruco_id_str = str(aruco_id)
                aruco.drawDetectedMarkers(image_bgr, [corners[i]])
                tvec = ArUco_marker_orientations[aruco_id_str]['tvec']
                distancia_a_camara = np.linalg.norm(tvec)
                cv.putText(image_bgr, f"ID: {aruco_id}, Dist: {distancia_a_camara:.2f}",
                           (int(corners[i][0][0][0]), int(corners[i][0][0][1])),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            avg_dist_1, avg_dist_2 = draw_panel(panel_frame, ArUco_marker_orientations, mtx, dist, yaw1, mapping_mode, stored_distances, avg_dist_1, avg_dist_2)
        else:
            avg_dist_1, avg_dist_2 = draw_panel(panel_frame, yaw=yaw1, mapping_mode=mapping_mode, stored_distances=stored_distances, avg_dist_1 = avg_dist_1, avg_dist_2 = avg_dist_2)


        if mapping_mode:
            text = "Mapping"
            font = cv.FONT_HERSHEY_SIMPLEX
            text_size = cv.getTextSize(text, font, 1, 2)[0]
            text_x = (image_bgr.shape[1] - text_size[0]) // 2
            text_y = (image_bgr.shape[0] + text_size[1]) // 2
            cv.putText(image_bgr, text, (text_x, text_y), font, 1, (0, 0, 255), 2)
            yaw_desired = -1
            if initial_yaw is not None:
                delta_yaw = yaw1 - initial_yaw
                if delta_yaw > pi:
                    delta_yaw -= 2 * pi
                elif delta_yaw < -pi:
                    delta_yaw += 2 * pi
                yaw_accumulated += abs(delta_yaw)
                initial_yaw = yaw1
                print(yaw_accumulated)

                if yaw_accumulated >= 4 * pi :
                    mapping_mode = False
                    yaw_desired = 0
                    print("Mapping mode: OFF")
                    # Aquí puedes realizar la acción de dibujar el rectángulo basado en las distancias almacenadas

        motor_power = PID_crazyflie.pid(dt, forward_desired, sideways_desired,
                                        yaw_desired, height_desired,
                                        roll, pitch, yaw_rate,
                                        altitude, v_x, v_y)

        m1_motor.setVelocity(-motor_power[0])
        m2_motor.setVelocity(motor_power[1])
        m3_motor.setVelocity(-motor_power[2])
        m4_motor.setVelocity(motor_power[3])

        past_time = robot.getTime()
        past_x_global = x_global
        past_y_global = y_global
        
        cv.imshow("Crazyflie Camera View", image_bgr)
        cv.imshow('Panel 3D', panel_frame)
        cv.waitKey(1)

    cv.destroyAllWindows()