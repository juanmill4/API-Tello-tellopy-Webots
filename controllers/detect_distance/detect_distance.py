from controller import Robot, Keyboard
from math import cos, sin, pi, radians, sqrt
import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import time
from pid_controller import pid_velocity_fixed_height_controller

FLYING_ATTITUDE = 1.4
MIN_TIME_BETWEEN_DIRECTION_CHANGES = 0.5  # tiempo mínimo en segundos entre cambios de dirección
MARKER_LOST_TIMEOUT = 1  # tiempo en segundos antes de considerar el marcador perdido

# Cargar la matriz de la cámara y los coeficientes de distorsión desde un archivo .npz
with np.load('/home/user/tello-ai/calib_data/webots/crazy/MultiMatrix.npz') as X:
    mtx, dist = X['camMatrix'], X['distCoef']

# Tamaño de marcador ArUco en milímetros (ajusta según tus marcadores)
SIZE = 300

# Diccionario para almacenar distancias y ángulos de etiquetas detectadas
marker_data = {}
stored_distances = {'dist_1': [], 'dist_2': [], 'angle_1': [], 'angle_2': []}
initial_yaw1 = None
yaw_accumulated = 0
yaw_in_range_count = 0
pass_start_time = None
yaw_target = None  # Variable global para mantener el yaw target inicial
drawn_rectangles = []

# Preparar el diccionario y los parámetros de ArUco
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
parameters = aruco.DetectorParameters_create()

# Ajustar valor dentro del rango -1.0 a 1.0
def fix_range(value):
    return max(min(value, 1.0), -1.0)

def set_throttle(value):
    global height_diff_desired, height_desired
    height_diff_desired = fix_range(value)
    height_desired += height_diff_desired * dt

def set_yaw(value):
    global yaw_desired
    yaw_desired = fix_range(value)

def set_pitch(value):
    global forward_desired
    forward_desired = fix_range(value)

def set_roll(value):
    global sideways_desired
    sideways_desired = fix_range(value)

def land():
    global height_desired, past_time, past_x_global, past_y_global
    height_desired = 0  # Set desired height to 0 to land
    while True:
        dt = robot.getTime() - past_time
        actual_state = {}
        
        # Get sensor data
        roll = imu.getRollPitchYaw()[0]
        pitch = imu.getRollPitchYaw()[1]
        yaw = imu.getRollPitchYaw()[2]
        yaw_rate = gyro.getValues()[2]
        x_global = gps.getValues()[0]
        v_x_global = (x_global - past_x_global) / dt
        y_global = gps.getValues()[1]
        v_y_global = (y_global - past_y_global) / dt
        altitude = gps.getValues()[2]

        # Get body fixed velocities
        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)
        v_x = v_x_global * cos_yaw + v_y_global * sin_yaw
        v_y = -v_x_global * sin_yaw + v_y_global * cos_yaw

        # PID velocity controller with fixed height
        motor_power = PID_crazyflie.pid(dt, 0, 0, 0, height_desired, roll, pitch, yaw_rate, altitude, v_x, v_y)

        m1_motor.setVelocity(-motor_power[0])
        m2_motor.setVelocity(motor_power[1])
        m3_motor.setVelocity(-motor_power[2])
        m4_motor.setVelocity(motor_power[3])

        past_time = robot.getTime()
        past_x_global = x_global
        past_y_global = y_global

        if altitude <= 0.05:  # Check if the drone is close enough to the ground
            m1_motor.setVelocity(0)
            m2_motor.setVelocity(0)
            m3_motor.setVelocity(0)
            m4_motor.setVelocity(0)
            print("Drone has landed.")
            break
        robot.step(timestep)

def takeoff():
    global height_desired, past_time, past_x_global, past_y_global
    height_desired = FLYING_ATTITUDE  # Set desired height to the flying attitude
    while True:
        dt = robot.getTime() - past_time
        actual_state = {}
        
        # Get sensor data
        roll = imu.getRollPitchYaw()[0]
        pitch = imu.getRollPitchYaw()[1]
        yaw = imu.getRollPitchYaw()[2]
        yaw_rate = gyro.getValues()[2]
        x_global = gps.getValues()[0]
        v_x_global = (x_global - past_x_global) / dt
        y_global = gps.getValues()[1]
        v_y_global = (y_global - past_y_global) / dt
        altitude = gps.getValues()[2]

        # Get body fixed velocities
        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)
        v_x = v_x_global * cos_yaw + v_y_global * sin_yaw
        v_y = -v_x_global * sin_yaw + v_y_global * cos_yaw

        # PID velocity controller with fixed height
        motor_power = PID_crazyflie.pid(dt, 0, 0, 0, height_desired, roll, pitch, yaw_rate, altitude, v_x, v_y)

        m1_motor.setVelocity(-motor_power[0])
        m2_motor.setVelocity(motor_power[1])
        m3_motor.setVelocity(-motor_power[2])
        m4_motor.setVelocity(motor_power[3])

        past_time = robot.getTime()
        past_x_global = x_global
        past_y_global = y_global

        if altitude >= FLYING_ATTITUDE - 0.05:  # Check if the drone has reached the desired altitude
            print("Drone has taken off.")
            break
        robot.step(timestep)

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

def calculate_infinite_line_points(p1, p2, img_width):
    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    intercept = p1[1] - slope * p1[0]
    
    x1 = 0
    y1 = int(slope * x1 + intercept)
    
    x2 = img_width - 1
    y2 = int(slope * x2 + intercept)
    
    return (x1, y1), (x2, y2)
    

def check_overlap(new_vertices, existing_rectangles):
    for rect in existing_rectangles:
        for vertex in new_vertices:
            if rect[0][0] <= vertex[0] <= rect[2][0] and rect[0][1] <= vertex[1] <= rect[2][1]:
                return True
    return False


def draw_panel(img, markers_orientations=None, camera_matrix=None, dist_coeffs=None, yaw=0, mapping_mode=False, stored_distances=None, avg_distances=None):
    
    global drawn_rectangles
    
    if avg_distances is None:
        avg_distances = {}


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
                yaw_d = orientations['yaw']
                roll_deg = orientations['roll']
                distance = np.linalg.norm(tvec)
                angle = np.arctan2(tvec[0][0], tvec[0][2])  # Ángulo en radianes utilizando X e Z
                
                marker_data[aruco_id] = {'distance': distance, 'angle': angle, 'tvec': tvec, 'rvec': rvec, 'yaw': yaw_d}
            
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

        for marker_id in marker_data.keys():
            dist_key = f'dist_{marker_id}'
            angle_key = f'angle_{marker_id}'
            if dist_key not in stored_distances:
                stored_distances[dist_key] = []
            if angle_key not in stored_distances:
                stored_distances[angle_key] = []

            stored_distances[dist_key].append(marker_data[marker_id]['distance'])
            stored_distances[angle_key].append(marker_data[marker_id]['angle'])

    # Calcular las distancias medias si hay datos en stored_distances y no estamos en mapping_mode
    # Calcular las distancias medias si hay datos en stored_distances y no estamos en mapping_mode
    if not mapping_mode and any(stored_distances.values()):
        for marker_id in marker_data.keys():
            marker_id_str = str(marker_id)
            print(f'marker_id: {marker_id_str}, int(marker_id): {int(marker_id)}, bits: {bin(int(marker_id))}')
            if marker_id_str.endswith('0') or marker_id_str.endswith('1'):
                print('Match found for marker_id:', marker_id)
                dist_key = f'dist_{marker_id}'
                angle_key = f'angle_{marker_id}'
    
                if dist_key in stored_distances and angle_key in stored_distances and stored_distances[dist_key] and stored_distances[angle_key]:
                    avg_dist = np.mean(stored_distances[dist_key])
                    avg_angle = np.mean(stored_distances[angle_key])
                    
                    avg_dist /= 50
                    avg_angle /= 50
    
                    avg_distances[marker_id] = (avg_dist, avg_angle)
                    
                    stored_distances[dist_key].clear()
                    stored_distances[angle_key].clear()
    
    print(avg_distances)
    # Condición para dibujar el rectángulo
    x_c, y_c = 250, 350
    # Condición para dibujar el rectángulo
    if not mapping_mode and avg_distances:
        valid_pairs = []
        sorted_keys = sorted(avg_distances.keys(), key=lambda x: int(x))
        for i in range(len(sorted_keys) - 1):
            id1, id2 = sorted_keys[i], sorted_keys[i + 1]
            if (int(id1) % 10 == 0 and int(id2) % 10 == 1) and (abs(int(id1) // 10 - int(id2) // 10) <= 1):
                valid_pairs.append((id1, id2))
        
        print(valid_pairs)
        
        for id1, id2 in valid_pairs:
            avg_dist_1, avg_angle_1 = avg_distances[id1]
            avg_dist_2, avg_angle_2 = avg_distances[id2]
    
            while True:
                vertices = [
                    (x_c - avg_dist_1, y_c + avg_dist_2),
                    (x_c + avg_dist_1, y_c + avg_dist_2),
                    (x_c + avg_dist_1, y_c - avg_dist_2),
                    (x_c - avg_dist_1, y_c - avg_dist_2),
                    (x_c - avg_dist_1, y_c + avg_dist_2)
                ]
    
                vertices = [(int(x), int(y)) for x, y in vertices]
    
                if not check_overlap(vertices, drawn_rectangles):
                    break
    
                # Desplazar el centro en función de las distancias
                x_c += 2 * max(avg_dist_1, avg_dist_2)
                y_c += 2 * max(avg_dist_1, avg_dist_2)
    
            drawn_rectangles.append(vertices[:4])
    
            for i in range(len(vertices) - 1):
                cv.line(img, vertices[i], vertices[i + 1], (255, 0, 255), 2)
                mid_x = (vertices[i][0] + vertices[i + 1][0]) // 2
                mid_y = (vertices[i][1] + vertices[i + 1][1]) // 2
                dist_1 = avg_dist_1 * 50
                dist_2 = avg_dist_2 * 50
                if i % 2 == 0:
                    label = f'D1: {dist_2:.2f}'
                else:
                    label = f'D2: {dist_1:.2f}'
                cv.putText(img, label, (mid_x, mid_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    
            drone_position = (int(x_c), int(y_c))
            cv.circle(img, drone_position, 5, (0, 0, 255), -1)

            # Actualizar la posición del dron en función de los datos de los marcadores detectados
            tvec_1 = marker_data[id1]['tvec']
            tvec_2 = marker_data[id2]['tvec']
            # Aquí se podría usar un promedio ponderado, o cualquier otro método de tu preferencia
            drone_x = (tvec_1[0][0] + tvec_2[0][0]) / 4
            drone_y = (tvec_1[0][2] + tvec_2[0][2]) / 4  # Usando la coordenada Z para y del plano

            # Convertir coordenadas de la posición del dron para dibujarlo en el panel (inverso)
            drone_position_x = int(x_c + drone_x / 50)  # Invertir la posición según tus necesidades
            drone_position_y = int(y_c - drone_y / 50)  # Invertir la posición según tus necesidades

            drone_position = (drone_position_x, drone_position_y)
            #cv.circle(img, drone_position, 5, (0, 0, 255), -1)

    return avg_distances


last_marker_4_seen_time = None

def follow_marker(center_x, center_y, frame_center_x, frame_center_y, distancia_a_camara, pitch_actual, roll_actual, throttle_actual, marker_detected, tvec, estado, id, marker_data, yaw1):
    global contador_pitch_cero, roll_contador_cero, contador_throttle_cero, last_marker_seen_time, last_marker_4_seen_time
    global last_valid_pitch, last_valid_roll, last_valid_throttle, yaw_desired, initial_yaw1, yaw_accumulated, yaw_in_range_count, pass_start_time, yaw_target
    global mapping_mode, mapping_completed
    
    pitch = 0
    roll = 0
    throttle = 0
    yaw = 0
    current_time = robot.getTime()
    pi = np.pi
    
    print(estado)

    if '5' in marker_data:
        x = marker_data['5']['tvec'][0][0]
        y = marker_data['5']['tvec'][0][1]
        z = marker_data['5']['tvec']
        z = np.linalg.norm(z)


    # Actualizar el tiempo de detección de la etiqueta 4
    if id == 4:
        last_marker_4_seen_time = current_time

    # Gestión del estado de centrado en roll
    if estado == "centrado_roll":
        if marker_detected and (id & 0b111) == 0b101:
            distancia_minima_roll = -200
            distancia_maxima_roll = 150
            roll_max = 0.3
            roll_min = 0.1

            if roll_contador_cero >= 16:
                print("Roll mantenido en 0 después de 16 veces consecutivas. Iniciando control de throttle.")
                roll_objetivo = 0
                roll_contador_cero = 0
                estado = "centrado_throttle"
            else:
                if x > distancia_maxima_roll:
                    diferencia = x - distancia_maxima_roll
                    roll_objetivo = -max(roll_min, min(roll_max, roll_max * (diferencia / 100.0)))  # Control proporcional suavizado
                elif x < distancia_minima_roll:
                    diferencia = distancia_minima_roll - x
                    roll_objetivo = max(roll_min, min(roll_max, roll_max * (diferencia / 100.0)))  # Control proporcional suavizado
                else:
                    roll_objetivo = 0

                if distancia_minima_roll <= x <= distancia_maxima_roll:
                    roll_contador_cero += 1
                    print("Dron centrado en el eje X, roll ajustado a 0.")
                else:
                    roll_contador_cero = 0

            last_marker_seen_time = current_time        
            last_valid_roll = roll_objetivo  # Guardar el último roll válido
        else:
            roll_objetivo = last_valid_roll

        # Suavizar la transición del roll actual al roll objetivo
        suavizado = 0.05  # Factor de suavizado, ajusta según sea necesario
        roll = roll_actual + (roll_objetivo - roll_actual) * suavizado

        if (current_time - last_marker_seen_time) < MARKER_LOST_TIMEOUT:
            set_roll(roll)
            print(f"Dron moviéndose con un roll de {roll:.3f}.")

    elif estado == "centrado_throttle":
        if marker_detected and (id & 0b111) == 0b101:
            distancia_minima_throttle = -300
            distancia_maxima_throttle = -100
            throttle_max = 0.2
            throttle_min = 0.1

            if contador_throttle_cero >= 16:
                print("Throttle mantenido en 0 después de 16 veces consecutivas. Iniciando control de pitch.")
                throttle_objetivo = 0
                contador_throttle_cero = 0
                estado = "control_pitch"
            else:
                if y > distancia_minima_throttle:
                    diferencia = y - distancia_minima_throttle
                    throttle_objetivo = -max(throttle_min, min(throttle_max, throttle_max * (diferencia / 100.0)))  # Control proporcional suavizado
                elif y < distancia_maxima_throttle:
                    diferencia = distancia_maxima_throttle - y
                    throttle_objetivo = max(throttle_min, min(throttle_max, throttle_max * (diferencia / 100.0)))  # Control proporcional suavizado
                else:
                    throttle_objetivo = 0

                if distancia_minima_throttle <= y <= distancia_maxima_throttle:
                    contador_throttle_cero += 1
                    print("Dron centrado en el eje Y, throttle ajustado a 0.")
                else:
                    contador_throttle_cero = 0

            last_marker_seen_time = current_time        
            last_valid_throttle = throttle_objetivo  # Guardar el último throttle válido
        else:
            throttle_objetivo = last_valid_throttle

        # Suavizar la transición del throttle actual al throttle objetivo
        suavizado = 0.05  # Factor de suavizado, ajusta según sea necesario
        throttle = throttle_actual + (throttle_objetivo - throttle_actual) * suavizado

        if (current_time - last_marker_seen_time) < MARKER_LOST_TIMEOUT:
            set_throttle(throttle)
            print(f"Dron moviéndose con un throttle de {throttle:.3f}.")


    # Gestión del estado de control de pitch
    elif estado == "control_pitch":
        if marker_detected and (id & 0b111) == 0b101:
            distancia_minima = 1500
            distancia_maxima = 1700
            pitch_max = 0.3  # Reducción del pitch máximo para suavizar el movimiento
            pitch_min = 0.1  # Pitch mínimo

            if contador_pitch_cero >= 16:
                print("Pitch mantenido en 0 después de 16 veces consecutivas.")
                pitch_objetivo = 0
                contador_pitch_cero = 0
                yaw_target = None
                estado = "control_yaw"
            else:
                if z > distancia_maxima:
                    diferencia = z - distancia_maxima
                    pitch_objetivo = max(pitch_min, min(pitch_max, pitch_max * (diferencia / 1000.0)))  # Control proporcional suavizado
                elif z < distancia_minima:
                    diferencia = distancia_minima - z
                    pitch_objetivo = -max(pitch_min, min(pitch_max, pitch_max * (diferencia / 1000.0)))  # Control proporcional suavizado
                else:
                    pitch_objetivo = 0

                if distancia_minima <= z <= distancia_maxima:
                    contador_pitch_cero += 1
                    print("Dron dentro del rango objetivo, pitch ajustado a 0.")
                else:
                    contador_pitch_cero = 0

            last_marker_seen_time = current_time
            last_valid_pitch = pitch_objetivo  # Guardar el último pitch válido
        else:
            pitch_objetivo = last_valid_pitch  # Mantener el último pitch válido

        # Suavizar la transición del pitch actual al pitch objetivo
        suavizado = 0.05  # Factor de suavizado, ajusta según sea necesario
        pitch = pitch_actual + (pitch_objetivo - pitch_actual) * suavizado

        if (current_time - last_marker_seen_time) < MARKER_LOST_TIMEOUT:
            set_pitch(pitch)
            print(f"Dron moviéndose con un pitch de {pitch:.3f}.")


    elif estado == "control_yaw":
        yaw_speed = 0.3

        if '5' in marker_data:
            yaw5 = marker_data['5']['yaw']

            if initial_yaw1 is None:
                initial_yaw1 = yaw1

            delta_yaw = yaw1 - initial_yaw1
            if delta_yaw > pi:
                delta_yaw -= 2 * pi
            elif delta_yaw < -pi:
                delta_yaw += 2 * pi
            yaw_accumulated += abs(delta_yaw)
            initial_yaw1 = yaw1

            if yaw_target is None:
                yaw_target = np.deg2rad(abs(yaw5) - 7)
            print(f'yaw target {yaw_target} yaw acummulated {yaw_accumulated}')
            

            if yaw_in_range_count >= 2:
                set_yaw(0)
                set_roll(0)
                set_pitch(0)
                set_throttle(0)
                print("Yaw en rango deseado por más de 10 veces. Dron detenido.")
                estado = 'stop'
                yaw_in_range_count = 0

            if yaw_accumulated < yaw_target:
                if yaw5 < 0:
                    yaw = yaw_speed
                else:
                    yaw = -yaw_speed
                set_yaw(yaw)
                print(f"Ajustando yaw: {yaw:.3f}")
            else:
                set_yaw(0)
                if yaw5 < 0:
                    set_roll(-0.3)
                else:
                    set_roll(0.3)
                print("Yaw ajustado. Iniciando ajuste de roll.")
                if marker_detected and (id & 0b111) == 0b101:
                    yaw_accumulated = 0
                    print('Marcador detectado nuevamente. Volviendo a estado centrado_roll.')
                    estado = 'centrado_roll'
                    yaw_in_range_count += 1


    elif estado == 'mapping':
        set_pitch(0)
        set_roll(0)
        set_throttle(0)
        yaw_d = 0.6  # Ajuste continuo del yaw
        set_yaw(yaw_d)
        print("Marcador no detectado por más de 3 segundos. Ajustando yaw para buscar marcador.")
        if marker_detected and (id & 0b111) == 0b101:
            print('Marcador detectado nuevamente.')
            estado = 'centrado_roll'
            set_yaw(0)

    elif estado == 'stop':
        set_yaw(0)
        set_roll(0)
        set_pitch(0)
        set_throttle(0)
        if id == 5:
            estado = "control_pitch_door"
        elif id == 4:
            print('Close Door')

    elif estado == "control_pitch_door":
        if marker_detected and (id & 0b111) == 0b101:
            distancia_minima = 1500
            distancia_maxima = 1700
            pitch_max = 0.3  # Reducción del pitch máximo para suavizar el movimiento
            pitch_min = 0.1  # Pitch mínimo

            if contador_pitch_cero >= 16:
                print("Pitch mantenido en 0 después de 16 veces consecutivas.")
                pitch_objetivo = 0
                contador_pitch_cero = 0
                estado = "centrado_roll_door"
            else:
                if z > distancia_maxima:
                    diferencia = z - distancia_maxima
                    pitch_objetivo = max(pitch_min, min(pitch_max, pitch_max * (diferencia / 1000.0)))  # Control proporcional suavizado
                elif z < distancia_minima:
                    diferencia = distancia_minima - z
                    pitch_objetivo = -max(pitch_min, min(pitch_max, pitch_max * (diferencia / 1000.0)))  # Control proporcional suavizado
                else:
                    pitch_objetivo = 0

                if distancia_minima <= z <= distancia_maxima:
                    contador_pitch_cero += 1
                    print("Dron dentro del rango objetivo, pitch ajustado a 0.")
                else:
                    contador_pitch_cero = 0

            last_marker_seen_time = current_time
            last_valid_pitch = pitch_objetivo  # Guardar el último pitch válido
        else:
            pitch_objetivo = last_valid_pitch  # Mantener el último pitch válido

        # Suavizar la transición del pitch actual al pitch objetivo
        suavizado = 0.05  # Factor de suavizado, ajusta según sea necesario
        pitch = pitch_actual + (pitch_objetivo - pitch_actual) * suavizado

        if (current_time - last_marker_seen_time) < MARKER_LOST_TIMEOUT:
            set_pitch(pitch)
            print(f"Dron moviéndose con un pitch de {pitch:.3f}.")


    elif estado == "centrado_roll_door":
        if marker_detected and (id & 0b111) == 0b101:
            distancia_minima_roll = -150
            distancia_maxima_roll = 50
            roll_max = 0.3
            roll_min = 0.1

            if roll_contador_cero >= 16:
                print("Roll mantenido en 0 después de 16 veces consecutivas. Iniciando estado 'pass'.")
                roll_objetivo = 0
                roll_contador_cero = 0
                estado = "stop_pass"
            else:
                if x > distancia_maxima_roll:
                    diferencia = x - distancia_maxima_roll
                    roll_objetivo = -max(roll_min, min(roll_max, roll_max * (diferencia / 100.0)))  # Control proporcional suavizado
                elif x < distancia_minima_roll:
                    diferencia = distancia_minima_roll - x
                    roll_objetivo = max(roll_min, min(roll_max, roll_max * (diferencia / 100.0)))  # Control proporcional suavizado
                else:
                    roll_objetivo = 0

                if distancia_minima_roll <= x <= distancia_maxima_roll:
                    roll_contador_cero += 1
                    print("Dron centrado en el eje X, roll ajustado a 0.")
                else:
                    roll_contador_cero = 0

            last_marker_seen_time = current_time        
            last_valid_roll = roll_objetivo  # Guardar el último roll válido
        else:
            roll_objetivo = last_valid_roll

        # Suavizar la transición del roll actual al roll objetivo
        suavizado = 0.05  # Factor de suavizado, ajusta según sea necesario
        roll = roll_actual + (roll_objetivo - roll_actual) * suavizado

        if (current_time - last_marker_seen_time) < MARKER_LOST_TIMEOUT:
            set_roll(roll)
            print(f"Dron moviéndose con un roll de {roll:.3f}.")



    elif estado == "stop_pass":
        if marker_detected and (id & 0b111) == 0b101:
            throttle_objetivo = -0.3
    
            last_marker_seen_time = current_time        
            last_valid_throttle = throttle_objetivo  # Guardar el último throttle válido
        else:
            throttle_objetivo = last_valid_throttle
            # Cambiar a estado 'centrado_roll_door' si la etiqueta 5 no es detectada por más de 0.5 segundos
            if (current_time - last_marker_seen_time) >= 0.7:
                print("Etiqueta 5 no detectada por más de 0.7 segundos. Cambiando a estado 'centrado_roll_door'.")
                estado = "pass"
                last_marker_seen_time = 0  # Resetear el tiempo de última detección
    
        # Suavizar la transición del throttle actual al throttle objetivo
        suavizado = 0.15  # Factor de suavizado, ajusta según sea necesario
        throttle = throttle_actual + (throttle_objetivo - throttle_actual) * suavizado
    
        if (current_time - last_marker_seen_time) < MARKER_LOST_TIMEOUT:
            set_throttle(throttle)
            print(f"Dron moviéndose con un throttle de {throttle:.3f}.")



    elif estado == 'pass':
        if last_marker_4_seen_time is not None and (current_time - last_marker_4_seen_time) < 0.3:
            print("Etiqueta 4 detectada recientemente. Esperando.")
            set_yaw(0)
            set_roll(0)
            set_pitch(0)
            set_throttle(0)
        else:
            pitch_speed = 0.6  # Velocidad de pitch deseada
        
            if pass_start_time is None:
                pass_start_time = current_time
        
            if current_time - pass_start_time < 6:
                set_pitch(pitch_speed)
                print(f"Manteniendo pitch a {pitch_speed} durante {current_time - pass_start_time:.2f} segundos.")
            else:
                set_pitch(0)
                print("Estado 'pass' completado. Volviendo a estado 'stop'.")
                estado = 'mapping'
                mapping_mode = True
                mapping_completed = False
                pass_start_time = None  # Resetear el tiempo de inicio para la próxima vez

    return pitch, roll, throttle, estado

if __name__ == '__main__':
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # Initialize motors
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


    # Initialize Sensors
    imu = robot.getDevice("inertial_unit")
    imu.enable(timestep)
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)
    camera = robot.getDevice("camera")
    camera.enable(timestep)

    # Get keyboard
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # Initialize variables
    past_x_global = 0
    past_y_global = 0
    past_time = 0
    first_time = True

    # Crazyflie velocity PID controller
    PID_crazyflie = pid_velocity_fixed_height_controller()
    PID_update_last_time = robot.getTime()
    sensor_read_last_time = robot.getTime()

    height_desired = FLYING_ATTITUDE
    autonomous_mode = False
    landing_mode = False
    takeoff_mode = False
    mapping_mode = False
    mapping_completed = False
    contador_pitch_cero = 0
    roll_contador_cero = 0
    contador_throttle_cero = 0
    last_sideways_change_time = 0
    current_sideways = 0
    pitch_actual = 0
    roll_actual = 0
    throttle_actual = 0
    last_marker_seen_time = 0
    last_valid_pitch = 0  # Inicializar el último pitch válido
    last_valid_roll = 0
    last_valid_throttle = 0
    estado = "mapping"  # Estado inicial
    avg_distances = None

    
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
    print("- Press T to take off\n")
    print("- Press L to land the drone\n")
    print("- Press M to start mapping mode\n")

    while robot.step(timestep) != -1:
        if not landing_mode and not takeoff_mode:
            dt = robot.getTime() - past_time
            actual_state = {}
    
            if first_time:
                past_x_global = gps.getValues()[0]
                past_y_global = gps.getValues()[1]
                past_time = robot.getTime()
                first_time = False
    
            # Get sensor data
            roll = imu.getRollPitchYaw()[0]
            pitch = imu.getRollPitchYaw()[1]
            yaw = imu.getRollPitchYaw()[2]
            yaw_rate = gyro.getValues()[2]
            yaw1 = -((yaw + pi) % (2 * pi) - pi)  # Convertir yaw de [0, 2π) a [-π, π)
            x_global = gps.getValues()[0]
            v_x_global = (x_global - past_x_global) / dt
            y_global = gps.getValues()[1]
            v_y_global = (y_global - past_y_global) / dt
            altitude = gps.getValues()[2]
    
            # Get body fixed velocities
            cos_yaw = cos(yaw)
            sin_yaw = sin(yaw)
            v_x = v_x_global * cos_yaw + v_y_global * sin_yaw
            v_y = -v_x_global * sin_yaw + v_y_global * cos_yaw
    
            # Initialize values
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
                elif key == ord('L'):
                    print("Landing initiated.")
                    landing_mode = True
                    break
                elif key == ord('T'):
                    print("Takeoff initiated.")
                    takeoff_mode = True
                    break
                elif key == ord('M'):
                    if not mapping_mode:
                        mapping_mode = True
                        initial_yaw = yaw1
                        yaw_accumulated = 0
                        mapping_completed = False  # Reset mapping completion
                        print("Mapping mode: ON")
                key = keyboard.getKey()
    
            height_desired += height_diff_desired * dt
    
            # Process camera image with OpenCV
            camera_image = camera.getImage()
            height = camera.getHeight()
            width = camera.getWidth()
            image = np.frombuffer(camera_image, np.uint8).reshape((height, width, 4))
            image_bgr = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
            gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    
            # Detección de marcadores ArUco
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            marker_detected = len(corners) > 0
    
            if mapping_mode:
                if marker_detected:
                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, SIZE, mtx, dist)
                    if ids is not None:
                        for i, aruco_id in enumerate(ids.flatten()):
                            Detected_ArUco_markers = detect_ArUco(image_bgr)
                            ArUco_marker_orientations = Calculate_orientation_in_degree(Detected_ArUco_markers, mtx, dist)
                            aruco.drawDetectedMarkers(image_bgr, [corners[i]])
                            cv.putText(image_bgr, f"ID: {aruco_id}",
                                       (int(corners[i][0][0][0]), int(corners[i][0][0][1])),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            avg_distances = draw_panel(panel_frame, ArUco_marker_orientations, mtx, dist, yaw1, mapping_mode, stored_distances, avg_distances)
                else:
                    avg_distances = draw_panel(panel_frame, yaw=yaw1, mapping_mode=mapping_mode, stored_distances=stored_distances, avg_distances=avg_distances)
    
                text = "Mapping"
                font = cv.FONT_HERSHEY_SIMPLEX
                text_size = cv.getTextSize(text, font, 1, 2)[0]
                text_x = (image_bgr.shape[1] - text_size[0]) // 2
                text_y = (image_bgr.shape[0] + text_size[1]) // 2
                cv.putText(image_bgr, text, (text_x, text_y), font, 1, (0, 0, 255), 2)
                set_yaw(-1)
                if initial_yaw is not None:
                    delta_yaw = yaw1 - initial_yaw
                    if delta_yaw > pi:
                        delta_yaw -= 2 * pi
                    elif delta_yaw < -pi:
                        delta_yaw += 2 * pi
                    yaw_accumulated += abs(delta_yaw)
                    initial_yaw = yaw1
    
                    if yaw_accumulated >= 4 * pi:
                        mapping_mode = False
                        mapping_completed = True
                        set_yaw(0)
                        yaw_accumulated = 0
                        print("Mapping mode: OFF")
    
            if mapping_completed and not mapping_mode:
                if marker_detected:
                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, SIZE, mtx, dist)
                    if ids is not None:
                        Detected_ArUco_markers = detect_ArUco(image_bgr)
                        ArUco_marker_orientations = Calculate_orientation_in_degree(Detected_ArUco_markers, mtx, dist)
                        for i, id in enumerate(ids):
                            print(id)
                            aruco.drawDetectedMarkers(image_bgr, corners, ids)
                            tvec = tvecs[i][0]
                            distancia_a_camara = np.linalg.norm(tvec)
                            cv.putText(image_bgr, f"ID: {id[0]}, Dist: {distancia_a_camara:.2f}", 
                                    (int(corners[i][0][0][0]), int(corners[i][0][0][1])), 
                                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cX, cY = int(np.mean(corners[i][0][:, 0])), int(np.mean(corners[i][0][:, 1]))
                            frame_center_x = gray.shape[1] / 2
                            frame_center_y = gray.shape[0] / 2
                            pitch_actual, roll_actual, throttle_actual, estado = follow_marker(cX, cY, frame_center_x, frame_center_y, distancia_a_camara, pitch_actual, roll_actual, throttle_actual, marker_detected, tvec, estado, id, marker_data, yaw1)
                            avg_distances = draw_panel(panel_frame, ArUco_marker_orientations, mtx, dist, yaw1, mapping_mode, stored_distances, avg_distances)
                else:
                    pitch_actual, roll_actual, throttle_actual, estado = follow_marker(0, 0, 0, 0, 0, pitch_actual, roll_actual, throttle_actual, marker_detected, [0, 0, 0], estado, id=None, marker_data=marker_data, yaw1=yaw1)
                    avg_distances = draw_panel(panel_frame, yaw=yaw1, mapping_mode=mapping_mode, stored_distances=stored_distances, avg_distances=avg_distances)
            # PID velocity controller with fixed height
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
        elif landing_mode:
            land()
            landing_mode = False
        elif takeoff_mode:
            takeoff()
            takeoff_mode = False
        
        cv.imshow("Crazyflie Camera View", image_bgr)
        cv.imshow('Panel 3D', panel_frame)
        cv.waitKey(1)
    
    cv.destroyAllWindows()
    