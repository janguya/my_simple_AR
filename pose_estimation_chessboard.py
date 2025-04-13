import numpy as np
import cv2 as cv
import pyrender
import trimesh
import os

screenshot_count = 1

def rotation_x(degrees):
    radians = np.radians(degrees)
    cos = np.cos(radians)
    sin = np.sin(radians)
    R = np.array([
        [1, 0, 0, 0],
        [0, cos, -sin, 0],
        [0, sin, cos, 0],
        [0, 0, 0, 1]
    ])
    return R

def rotation_y(degrees):
    radians = np.radians(degrees)
    cos = np.cos(radians)
    sin = np.sin(radians)
    R = np.array([
        [cos, 0, sin, 0],
        [0, 1, 0, 0],
        [-sin, 0, cos, 0],
        [0, 0, 0, 1]
    ])
    return R

def rotation_z(degrees):
    radians = np.radians(degrees)
    cos = np.cos(radians)
    sin = np.sin(radians)
    R = np.array([
        [cos, -sin, 0, 0],
        [sin, cos, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return R

# ========== Calibration Data ==========
video_file = 'data/chessboard.mp4'
model_path = 'data/hand.obj'

K = np.array([[1.16604299e+03, 0.00000000e+00, 6.59299947e+02],
              [0.00000000e+00, 1.16211608e+03, 3.68545490e+02],
              [0, 0, 1]])
dist_coeff = np.array([0.311, -2.171, 0.004, 0.004, 4.87])
board_pattern = (7, 4)
board_cellsize = 0.03
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# ========== Load 3D model ==========
scene_or_mesh = trimesh.load(model_path)

model_nodes = []

# Scene이면 geometry 리스트로 분리
if isinstance(scene_or_mesh, trimesh.Scene):
    mesh_nodes = []
    print('scene loaded')
    for name, geometry in scene_or_mesh.geometry.items():
        mesh = pyrender.Mesh.from_trimesh(geometry, smooth=False)
        mesh_nodes.append(mesh)
    print(mesh_nodes)
else:
    mesh_nodes = [pyrender.Mesh.from_trimesh(scene_or_mesh)]

scene_model = pyrender.Scene()
for mesh in mesh_nodes:
    node = scene_model.add(mesh)
    model_nodes.append(node)


# ========== Camera Intrinsic ==========
cam = pyrender.IntrinsicsCamera(
    fx=K[0, 0], fy=K[1, 1],
    cx=K[0, 2], cy=K[1, 2]
)
scene_model.add(cam, pose=np.eye(4))

# ========== Lighting ==========
light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
scene_model.add(light, pose=np.eye(4))

# ========== Prepare object points ==========
obj_points = board_cellsize * np.array(
    [[c, r, 2] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
)

# ========== Renderer ==========
renderer = pyrender.OffscreenRenderer(viewport_width=1280, viewport_height=720)

# ========== Read Video ==========
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read video: ' + video_file

while True:
    valid, img = video.read()
    if not valid:
        break

    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Rotation vector to matrix
        R, _ = cv.Rodrigues(rvec)
        t = tvec.reshape(3) * 2

        # 체스보드 중앙쪽에 모델을 위치시키기 위한 offset
        cols, rows = board_pattern
        center_col = (cols - 1) / 2
        center_row = (rows - 1) / 2
        target_offset = np.array([center_col, center_row, 0]) * board_cellsize

        # 기존 pose 행렬 구성
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t + R @ target_offset

        # 좌표계 보정 및 스케일/회전 적용
        flip_z = np.diag([1, -1, -1, 1])
        scale = 0.01
        S = np.diag([scale, scale, scale, 1.0])
        R_x = rotation_x(-90)

        model_pose = flip_z @ T @ R_x @ S

        for node in model_nodes:
            scene_model.set_pose(node, pose=model_pose)

        # Render with depth
        color, depth = renderer.render(scene_model)

        # Resize both to match video frame
        color = cv.resize(color, (img.shape[1], img.shape[0]))
        depth = cv.resize(depth, (img.shape[1], img.shape[0]))

        # mask: 모델이 그려진 부분만 True
        mask = depth > 0

        # 원본 복사
        output = img.copy()

        # 모델이 그려진 부분만 덧씌움 (OpenCV 방식)
        output[mask] = color[mask]

        # 카메라 위치 정보 표시
        p = (-R.T @ t).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(output, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

        # 결과 보여주기
        cv.imshow('AR Hand Pose Estimation', output)

    else:
        cv.imshow('AR Hand Pose Estimation', img)

    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    elif key == 13:  # 엔터 키 (Enter = ASCII 13)
        filename = f"data/screenshot_{screenshot_count:02d}.png"
        while os.path.exists(filename):
            screenshot_count += 1
            filename = f"data/screenshot_{screenshot_count:02d}.png"
        cv.imwrite(filename, output)
        print(f"📸 Screenshot saved to: {filename}")
        screenshot_count += 1
    elif key == 27:
        break

video.release()
cv.destroyAllWindows()